"""Build the auditable local-delivery payload for the ModelNet40 cloud teacher run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


CONDITIONS = ("clean", "illumination_1.0", "defocus_0.2", "sensor_noise_0.4")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--baseline-checkpoint", required=True)
    p.add_argument("--delivery-dir", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260809)
    return p.parse_args()


def read_predictions(path):
    rows = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            rows[(item["condition"], int(item["index"]))] = item
    return rows


def paired_ci(rows, baseline, method, samples, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray([row["label"] for row in rows])
    a = np.asarray([row["predictions"][baseline] for row in rows]) == y
    b = np.asarray([row["predictions"][method] for row in rows]) == y
    delta = b.astype(np.int8) - a.astype(np.int8)
    draws = rng.integers(0, len(delta), size=(samples, len(delta)))
    values = delta[draws].mean(axis=1) * 100
    return [round(float(x), 3) for x in np.quantile(values, [.025, .975])]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    a = parse_args(); run = Path(a.run_dir); delivery = Path(a.delivery_dir)
    delivery.mkdir(parents=True, exist_ok=True)
    originals_path = run / "official_test_predictions.jsonl"
    tuned_path = run / "illumination_tuned_test_predictions.jsonl"
    # The clean-guard rerun evaluates only the final deployable adapter.  Keep
    # the old five-method delivery when available, but support a truthful
    # baseline-vs-final package instead of inventing unavailable ablations.
    if originals_path.exists():
        merged = read_predictions(originals_path)
        tuned = read_predictions(tuned_path)
        for key, row in tuned.items():
            merged[key]["predictions"].update(row["predictions"])
        methods = ("edge_baseline", "cloud_unlabeled", "label_only", "hybrid",
                   "cloud_unlabeled_illumination_tuned")
    else:
        merged = read_predictions(tuned_path)
        methods = ("edge_baseline", "cloud_unlabeled_illumination_tuned")
    conditions = []
    for condition in CONDITIONS:
        rows = [row for (name, _), row in sorted(merged.items()) if name == condition]
        entry = {"condition": condition, "samples": len(rows), "methods": {}}
        for method in methods:
            accuracy = np.mean([row["predictions"][method] == row["label"] for row in rows]) * 100
            value = {"accuracy_pct": round(float(accuracy), 3)}
            if method != "edge_baseline":
                baseline = entry["methods"]["edge_baseline"]["accuracy_pct"]
                value["gain_over_edge_pp"] = round(float(accuracy - baseline), 3)
                value["paired_bootstrap_95ci_gain_pp"] = paired_ci(
                    rows, "edge_baseline", method, a.bootstrap_samples,
                    a.seed + CONDITIONS.index(condition) * 101 + methods.index(method))
            entry["methods"][method] = value
        conditions.append(entry)
    drift_names = CONDITIONS[1:]
    mean_drift = {}
    for method in methods:
        values = [next(c for c in conditions if c["condition"] == name)["methods"][method]["accuracy_pct"]
                  for name in drift_names]
        mean_drift[method] = round(sum(values) / len(values), 3)
    metrics = {
        "artifact": "ModelNet40 InternViT-6B cloud-teacher AdaptFormer full official-test results",
        "artifact_version": "2026-08-12", "official_test_tracks": 2468,
        "teacher_protocol": "frozen InternViT-6B; new 40-class head; 9043 labelled offline-head tracks",
        "refresh_protocol": "480 disjoint tracks; cloud_unlabeled losses do not use refresh labels",
        "development_tracks": 320, "conditions": conditions,
        "mean_drift_accuracy_pct": mean_drift,
        "selection_note": "illumination profiles selected on train-internal development before their official-test evaluation",
    }
    with open(delivery / "modelnet40_recovery_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    copy_names = [
        "teacher_head.pth", "cloud_unlabeled_adapter.pth", "label_only_adapter.pth",
        "hybrid_adapter.pth", "cloud_unlabeled_illumination_tuned_adapter.pth",
        "refresh_teacher_cache.pt", "teacher_gate.json", "teacher_head_metrics.json",
        "split_manifest.json", "summary.json", "illumination_tuned_summary.json",
        "official_test_predictions.jsonl", "illumination_tuned_test_predictions.jsonl",
        "run.log", "adapter_test.log", "illumination_tune.log",
    ]
    for name in copy_names:
        source = run / name
        if source.exists(): shutil.copy2(source, delivery / name)
    shutil.copy2(a.baseline_checkpoint, delivery / "edge_baseline_modelnet40.pth")
    shutil.copy2(Path(__file__).with_name("modelnet_cloud_teacher_refresh.py"),
                 delivery / "modelnet_cloud_teacher_refresh.py")
    readme = f"""# ModelNet40 大 ViT 云教师完整交付

这是完整 official `test` split（2,468 条四视图轨迹）的结果，不是 quick subset。
冻结的 InternViT-6B 主干没有被改写；`teacher_head.pth` 是重新用 ModelNet40 训练的 40 类头。
正式 `cloud_unlabeled` Adapter 的 refresh loss 不读取 480 条上传轨迹的标签。

## Teacher gate

- Edge mean drift: 91.979%
- InternViT-6B teacher mean drift: 95.417%
- Teacher gain: +3.437 pp（通过）

## Full official test

| 方法 | Clean | Illumination | Defocus | Sensor noise | Mean drift |
|---|---:|---:|---:|---:|---:|
"""
    lookup = {c["condition"]: c for c in conditions}
    labels = {"edge_baseline": "Edge baseline", "cloud_unlabeled": "Cloud unlabeled",
              "label_only": "Label-only upper bound", "hybrid": "Hybrid",
              "cloud_unlabeled_illumination_tuned": "Cloud unlabeled (clean-guard tuned)"}
    for method in methods:
        nums = [lookup[name]["methods"][method]["accuracy_pct"] for name in CONDITIONS]
        readme += f"| {labels[method]} | {nums[0]:.3f}% | {nums[1]:.3f}% | {nums[2]:.3f}% | {nums[3]:.3f}% | {mean_drift[method]:.3f}% |\n"
    readme += """

逐条件的 paired-bootstrap 95% CI、所有方法的原始精度见
`modelnet40_recovery_metrics.json`；逐样本标签和预测见两个 JSONL 文件。
`split_manifest.json` 固化了 head/refresh/dev 的互斥索引。
"""
    with open(delivery / "README.md", "w") as f: f.write(readme)
    files = sorted(path for path in delivery.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    with open(delivery / "SHA256SUMS", "w") as f:
        for path in files: f.write(f"{sha256(path)}  {path.name}\n")


if __name__ == "__main__":
    main()
