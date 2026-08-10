"""Fast, fixed-checkpoint sanity evaluation on an independent BoxCars test subset.

This is intentionally an *evaluation-only* script.  It never trains a head or
an Adapter, and it does not select a checkpoint from test results.  The default
is a deterministic, class-stratified 256-track subset of BoxCars' official test
split.  BoxCars' official test split is highly imbalanced (one make has only
three tracks), so equal per-class sampling would discard too much test data.
The sampler therefore preserves the broad class mix while retaining every make
that can be represented.  This is sufficient for a quick independent direction check, but explicitly not
a replacement for a full-test statistical report.

It evaluates the models needed for the system claim:
  - frozen Edge MV-ViT-S baseline;
  - the existing label-free cloud-guided AdaptFormer refresh;
  - optionally, frozen InternViT-6B plus the already-selected BoxCars task
    head.  Omit ``--teacher-model-path``/``--teacher-head-checkpoint`` to skip
    the expensive 6B pass; the Edge-versus-Adapter paired claim does not need it.

For every condition it writes the exact sampled indices and per-track
predictions, so the Edge-versus-Adapter comparison is paired and reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from adaptformer import attach_adaptformer, load_adapter_checkpoint, set_adapter_enabled
from boxcars_camera_drift_dataset import PairedBoxCarsCameraDrift
from boxcars_dataset import BoxCarsMultiView
from model import EarlyFusionMultiViewViT


# BoxCars images are first converted from ImageNet normalization back to
# [0, 1], then normalized with InternViT's image statistics.
IM = torch.tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
IS = torch.tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
CM = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 1, 3, 1, 1)
CS = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 1, 3, 1, 1)

SPECS = {
    "clean": (None, 0.0),
    "illumination_1.0": ("illumination", 1.0),
    "motion_blur_0.8": ("motion_blur", 0.8),
    "sensor_noise_0.6": ("sensor_noise", 0.6),
}


class TaskHead(nn.Module):
    """Exact small head layout used by retrain_boxcars_cloud_teacher_head_from_cache."""

    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 0):
        super().__init__()
        self.network = (
            nn.Linear(feature_dim, num_classes)
            if not hidden_dim
            else nn.Sequential(
                nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes),
            )
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick paired BoxCars official-test evaluation")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--teacher-model-path", default="",
                        help="leave empty to skip the 6B teacher evaluation")
    parser.add_argument("--teacher-head-checkpoint", default="",
                        help="leave empty to skip the 6B teacher evaluation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=256,
                        help="deterministic class-stratified official-test sample; 0 means the full test split")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--edge-batch-size", type=int, default=16)
    parser.add_argument("--teacher-batch-size", type=int, default=1,
                        help="keep 1 on a 24GB GPU for the 6B teacher")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def base_dataset(path: str) -> BoxCarsMultiView:
    return BoxCarsMultiView(
        path, "test", "make", 4,
        transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
    )


def stratified_indices(dataset: BoxCarsMultiView, requested: int, seed: int) -> list[int]:
    """Choose exactly ``requested`` tracks, preserving the test class mix.

    Every class receives one item if the budget permits; the remaining budget
    is allocated proportionally over its residual capacity.  This handles the
    official split's extremely rare Porsche class without pretending that a
    256-track quick check can be class-balanced.
    """
    if requested <= 0 or requested >= len(dataset):
        return list(range(len(dataset)))
    groups: dict[int, list[int]] = defaultdict(list)
    for index, (_vehicle, label) in enumerate(dataset.samples):
        groups[int(label)].append(index)
    if len(groups) != len(dataset.classes):
        raise RuntimeError("official test split does not contain every requested class")
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    labels = sorted(groups)
    if requested < len(labels):
        # The caller asked for fewer rows than classes.  A global deterministic
        # sample is less misleading than silently dropping arbitrary classes.
        return sorted(rng.sample(range(len(dataset)), requested))
    allocation = {label: 1 for label in labels}
    remaining = requested - len(labels)
    residual_capacity = {label: len(groups[label]) - 1 for label in labels}
    total_capacity = sum(residual_capacity.values())
    raw_extra = {label: remaining * residual_capacity[label] / total_capacity for label in labels}
    for label in labels:
        allocation[label] += min(residual_capacity[label], int(raw_extra[label]))
    unallocated = requested - sum(allocation.values())
    # Largest-remainder allocation is deterministic and cannot overdraw a
    # small class.  Repeat only in the unlikely event a class hits capacity.
    ordered = sorted(labels, key=lambda label: (raw_extra[label] % 1, residual_capacity[label]), reverse=True)
    while unallocated:
        progressed = False
        for label in ordered:
            if allocation[label] < len(groups[label]):
                allocation[label] += 1
                unallocated -= 1
                progressed = True
                if not unallocated:
                    break
        if not progressed:
            raise RuntimeError("cannot allocate requested test sample")
    selected: list[int] = []
    for label in labels:
        selected.extend(groups[label][:allocation[label]])
    rng.shuffle(selected)
    return selected


def make_condition_dataset(base: BoxCarsMultiView, indices: list[int], drift: str | None, severity: float):
    selected = drift or "illumination"
    paired = PairedBoxCarsCameraDrift(
        base, (selected,), clean_probability=0.0, seed=123,
        fixed_drift=selected, fixed_severity=severity, return_metadata=True,
    )
    return Subset(paired, indices)


def loader(dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )


def load_edge_model(baseline_path: str, adapter_path: str, classes: int, device: torch.device):
    model = EarlyFusionMultiViewViT("vit_small_patch16_224", 4, classes, pretrained=False)
    checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
    checkpoint_classes = checkpoint.get("classes") if isinstance(checkpoint, dict) else None
    if checkpoint_classes is not None and len(checkpoint_classes) != classes:
        raise ValueError("baseline checkpoint class count does not match BoxCars test")
    model.load_state_dict(checkpoint.get("model", checkpoint.get("state_dict", checkpoint)), strict=True)
    attach_adaptformer(model, r=32)
    _missing, unexpected = load_adapter_checkpoint(model, adapter_path, device="cpu")
    if unexpected:
        raise RuntimeError(f"unexpected Adapter checkpoint keys: {unexpected}")
    return model.to(device).eval()


@torch.inference_mode()
def evaluate_edge(model, data_loader: DataLoader, condition: str, device: torch.device):
    rows = []
    correct = {"Edge baseline": 0, "Cloud unlabeled Adapter": 0}
    count = 0
    for batch in data_loader:
        clean, corrupt, view_mask, labels = batch[:4]
        images = clean if condition == "clean" else corrupt
        images, view_mask = images.to(device, non_blocking=True), view_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        set_adapter_enabled(model, False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            baseline = model(images, view_mask=view_mask).argmax(1)
        set_adapter_enabled(model, True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            adapted = model(images, view_mask=view_mask).argmax(1)
        # ``Subset`` leaves PairedBoxCars metadata unchanged, so obtain the
        # original index from the final item returned by the paired dataset.
        indices = batch[-1].tolist()
        for local, index in enumerate(indices):
            truth, p0, p1 = int(labels[local]), int(baseline[local]), int(adapted[local])
            correct["Edge baseline"] += int(p0 == truth)
            correct["Cloud unlabeled Adapter"] += int(p1 == truth)
            rows.append({"dataset_index": int(index), "condition": condition,
                         "label": truth, "model": "Edge baseline", "prediction": p0,
                         "correct": bool(p0 == truth)})
            rows.append({"dataset_index": int(index), "condition": condition,
                         "label": truth, "model": "Cloud unlabeled Adapter", "prediction": p1,
                         "correct": bool(p1 == truth)})
        count += len(labels)
    return {name: value / max(count, 1) for name, value in correct.items()}, rows


@torch.inference_mode()
def teacher_features(model, images: torch.Tensor, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    pixels = ((images * IS + IM).clamp(0, 1) - CM) / CS
    batch, views = pixels.shape[:2]
    output = model(pixels.reshape(batch * views, *pixels.shape[2:]).to(device, torch.bfloat16))
    hidden = output[0] if isinstance(output, (tuple, list)) else output.last_hidden_state
    hidden = hidden[:, 0] if hidden.ndim == 3 else hidden
    hidden = hidden.reshape(batch, views, -1).float()
    weights = mask.to(device, hidden.dtype).unsqueeze(-1)
    return F.normalize((hidden * weights).sum(1) / weights.sum(1).clamp_min(1), dim=-1)


def load_teacher(model_path: str, head_path: str, device: torch.device):
    # Some older transformers versions used on a4 need this compatibility
    # property for the InternViT remote-code model to load.
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = property(lambda _self: {})
    from transformers import AutoModel
    teacher = AutoModel.from_pretrained(
        model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()
    payload = torch.load(head_path, map_location="cpu", weights_only=False)
    head = TaskHead(int(payload["feature_dim"]), int(payload["num_classes"]), int(payload.get("hidden_dim", 0)))
    head.load_state_dict(payload["state_dict"], strict=True)
    return teacher, head.to(device).eval(), payload


@torch.inference_mode()
def evaluate_teacher(teacher, head, data_loader: DataLoader, condition: str, device: torch.device):
    rows, correct, count = [], 0, 0
    for batch in data_loader:
        clean, corrupt, view_mask, labels = batch[:4]
        images = clean if condition == "clean" else corrupt
        features = teacher_features(teacher, images, view_mask, device)
        prediction = head(features).argmax(1).cpu()
        indices = batch[-1].tolist()
        for local, index in enumerate(indices):
            truth, pred = int(labels[local]), int(prediction[local])
            correct += int(pred == truth)
            rows.append({"dataset_index": int(index), "condition": condition,
                         "label": truth, "model": "Cloud teacher", "prediction": pred,
                         "correct": bool(pred == truth)})
        count += len(labels)
    return correct / max(count, 1), rows


def paired_bootstrap(records: list[dict], resamples: int, seed: int):
    """Paired CI for Adapter minus Edge accuracy; CPU-only and deterministic."""
    if not resamples:
        return None
    grouped: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row["model"] in ("Edge baseline", "Cloud unlabeled Adapter"):
            grouped[row["condition"]][row["model"]].append(bool(row["correct"]))
    answer = {}
    generator = torch.Generator().manual_seed(seed)
    for condition, values in grouped.items():
        base = torch.tensor(values["Edge baseline"], dtype=torch.float32)
        adapter = torch.tensor(values["Cloud unlabeled Adapter"], dtype=torch.float32)
        if len(base) != len(adapter):
            raise RuntimeError("paired prediction rows are not aligned")
        indices = torch.randint(len(base), (resamples, len(base)), generator=generator)
        differences = (adapter[indices].mean(1) - base[indices].mean(1)).sort().values
        answer[condition] = {
            "point_difference": float(adapter.mean() - base.mean()),
            "ci95": [float(differences[int(.025 * resamples)]),
                     float(differences[min(resamples - 1, int(.975 * resamples))])],
            "resamples": resamples,
        }
    return answer


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("quick cloud-teacher test requires CUDA")
    if args.samples < 0 or args.edge_batch_size < 1 or args.teacher_batch_size < 1:
        raise ValueError("sample and batch sizes must be valid")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")
    base = base_dataset(args.dataset_path)
    indices = stratified_indices(base, args.samples, args.seed)
    is_full = len(indices) == len(base)
    run_teacher = bool(args.teacher_model_path and args.teacher_head_checkpoint)
    if bool(args.teacher_model_path) != bool(args.teacher_head_checkpoint):
        raise ValueError("--teacher-model-path and --teacher-head-checkpoint must be given together")
    datasets = {name: make_condition_dataset(base, indices, *spec) for name, spec in SPECS.items()}
    started = time.time()
    records: list[dict] = []
    metrics: dict[str, dict[str, float]] = {
        "Edge baseline": {}, "Cloud unlabeled Adapter": {},
    }

    edge = load_edge_model(args.baseline_checkpoint, args.adapter_checkpoint, len(base.classes), device)
    for name, dataset in datasets.items():
        score, rows = evaluate_edge(edge, loader(dataset, args.edge_batch_size, args.num_workers), name, device)
        records.extend(rows)
        for model_name, value in score.items():
            metrics[model_name][name] = value
        print(f"edge {name}: baseline={score['Edge baseline']:.4%} adapter={score['Cloud unlabeled Adapter']:.4%}", flush=True)
    del edge
    torch.cuda.empty_cache()

    head_payload = None
    if run_teacher:
        metrics["Cloud teacher"] = {}
        teacher, head, head_payload = load_teacher(args.teacher_model_path, args.teacher_head_checkpoint, device)
        for name, dataset in datasets.items():
            score, rows = evaluate_teacher(teacher, head, loader(dataset, args.teacher_batch_size, args.num_workers), name, device)
            records.extend(rows)
            metrics["Cloud teacher"][name] = score
            print(f"teacher {name}: {score:.4%}", flush=True)
        del teacher, head
        torch.cuda.empty_cache()

    for model_metrics in metrics.values():
        model_metrics["mean_drift"] = sum(model_metrics[name] for name in SPECS if name != "clean") / 3
    scope = {
        "split": "official BoxCars116k test",
        "evaluation_type": ("full official test split" if is_full
                            else "fixed deterministic class-stratified quick subset"),
        "samples": len(indices),
        "full_test_samples": len(base),
        "selection_statement": "No checkpoint, head, epoch, or loss weight was selected from these test results.",
    }
    if not is_full:
        scope["limitation"] = "This is a rapid independent test check, not a full-test final statistical report."
    result = {
        "scope": scope,
        "conditions": {name: {"drift": spec[0] or "normal", "severity": spec[1]} for name, spec in SPECS.items()},
        "sample_indices": indices,
        "sample_class_counts": {base.classes[label]: sum(base.samples[index][1] == label for index in indices)
                                for label in range(len(base.classes))},
        "models": metrics,
        "paired_bootstrap_adapter_minus_edge": paired_bootstrap(records, args.bootstrap_resamples, args.seed + 1),
        "teacher_head": ({key: head_payload.get(key) for key in ("teacher", "head_mode", "feature_dim", "hidden_dim", "best_epoch")}
                         if head_payload is not None else None),
        "checkpoints": {
            "baseline": os.path.abspath(args.baseline_checkpoint),
            "cloud_unlabeled_adapter": os.path.abspath(args.adapter_checkpoint),
            "teacher_head": os.path.abspath(args.teacher_head_checkpoint) if run_teacher else None,
        },
        "elapsed_seconds": time.time() - started,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with open(os.path.join(args.output_dir, "predictions.jsonl"), "w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    with open(os.path.join(args.output_dir, "summary.md"), "w", encoding="utf-8") as handle:
        if is_full:
            handle.write("# BoxCars official test full evaluation\n\n")
            handle.write("This run evaluates the complete official test split with all checkpoints fixed in advance.\n\n")
        else:
            handle.write("# Quick independent BoxCars test check\n\n")
            handle.write("This is a fixed, deterministic, stratified subset of the official test split. "
                         "It is not the full-test final report.\n\n")
        handle.write("| Model | Clean | Illumination | Blur | Noise | Mean drift |\n|---|---:|---:|---:|---:|---:|\n")
        for model_name, value in metrics.items():
            handle.write(f"| {model_name} | {value['clean']:.2%} | {value['illumination_1.0']:.2%} | "
                         f"{value['motion_blur_0.8']:.2%} | {value['sensor_noise_0.6']:.2%} | {value['mean_drift']:.2%} |\n")
        handle.write("\nExact indices and per-track predictions are stored in `summary.json` and `predictions.jsonl`.\n")
    print(json.dumps(result["models"], indent=2), flush=True)


if __name__ == "__main__":
    main()
