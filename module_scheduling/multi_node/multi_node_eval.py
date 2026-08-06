"""multi_node/multi_node_eval.py — 真实多节点多视角冲突仲裁评估。

4 节点各按不同 view_mask 看同一 BoxCars test 样本的 4 视角子集（各看 3 缺 1），
各自跑 MV-ViT+adapter 推理 → pred/conf → arbiter 仲裁 → 统计冲突率/解决率。

4 节点共用同一套 baseline+adapter（控制变量，冲突只来自视角差异）。
复用：boxcars_dataset / model / adaptformer / arbiter。
在 AutoDL 跑（真模型+数据+adapter 权重）。

对应硬指标：冲突率≤5% / 解决率≥90%。

诚实边界（答辩口径）：
- 4 视角非 4 摄像头（BoxCars 是同摄像头同车 4 时间观测，作"多视角"用，不能说"4 物理摄像头"）
- 4 节点都看同一样本（同目标 4 视角），非多路口多目标
- target_id=样本序号（同一样本天然同目标，非 ReID）
- 冲突率反映"视角差异导致的预测分歧"，不是"多路口多目标冲突"
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms


SCRIPT_DIR = Path(__file__).resolve().parent
PERCEPTION_DIR = SCRIPT_DIR.parent.parent / "module_edge_perception"
for _p in (str(SCRIPT_DIR), str(PERCEPTION_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from boxcars_dataset import BoxCarsMultiView, VALID_TASKS
from model import EarlyFusionMultiViewViT
from adaptformer import attach_adaptformer, load_adapter_checkpoint
from arbiter import (
    Arbiter,
    WeightedVoteFusion,
    BayesianFusion,
    Report,
    CONFLICT_CONF_THRESHOLD,
)


# 4 节点视角分配：各看 3 视角，缺 1 不同（重叠 2）
DEFAULT_VIEW_MASKS = [
    [1, 1, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1],
]


def parse_args():
    p = argparse.ArgumentParser(description="真实多节点多视角冲突仲裁评估")
    p.add_argument("--dataset-path", required=True, help="BoxCars116k 根目录")
    p.add_argument("--baseline-checkpoint", required=True, help="baseline.pth（EarlyFusionMultiViewViT 主干）")
    p.add_argument("--adapter-checkpoint", required=True, help="adapter_best.pth（u=1 下发口径）")
    p.add_argument("--task", default="make", choices=VALID_TASKS)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--model-name", default="vit_small_patch16_224")
    p.add_argument("--adapter-r", type=int, default=32)
    p.add_argument("--split", default="test", choices=("train", "validation", "test"))
    p.add_argument("--output", default=str(SCRIPT_DIR / "multi_node_eval.csv"))
    p.add_argument("--fusion", default="weighted", choices=("weighted", "bayesian"))
    p.add_argument("--conf-threshold", type=float, default=CONFLICT_CONF_THRESHOLD)
    p.add_argument("--max-samples", type=int, default=0, help="0=test 全量；>0 子集验证")
    p.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def build_model(args, num_classes, device):
    """加载 baseline 主干 + attach adapter + 加载 adapter 权重（4 节点共用这一套）。"""
    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=args.num_views,
        num_classes=num_classes,
        pretrained=False,
    ).to(device)
    baseline_state = torch.load(args.baseline_checkpoint, map_location="cpu")
    if isinstance(baseline_state, dict) and "model" in baseline_state:
        baseline_state = baseline_state["model"]
    missing, unexpected = model.load_state_dict(baseline_state, strict=False)
    print(f"baseline 主干加载: missing={len(missing)}, unexpected={len(unexpected)}")
    attach_adaptformer(model, r=args.adapter_r)
    model.to(device)  # attach 后新模块在 CPU，再迁移一次
    miss2, unexp2 = load_adapter_checkpoint(model, args.adapter_checkpoint, device=device)
    print(f"adapter 权重加载: missing={len(miss2)}, unexpected={len(unexp2)}")
    model.eval()
    return model


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("多节点评估需要 CUDA（在 AutoDL 跑）")
    device = torch.device("cuda")
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = BoxCarsMultiView(
        args.dataset_path, args.split, args.task, args.num_views, transform
    )
    num_classes = len(dataset.classes)
    print(f"BoxCars {args.split}: {len(dataset)} 样本, {num_classes} 类")

    model = build_model(args, num_classes, device)
    view_masks = DEFAULT_VIEW_MASKS
    num_nodes = len(view_masks)
    print(f"节点数: {num_nodes}, 视角分配: {view_masks}")
    print(f"融合策略: {args.fusion}, conf 阈值: {args.conf_threshold}")

    fusion = WeightedVoteFusion() if args.fusion == "weighted" else BayesianFusion()
    arb = Arbiter(fusion=fusion, conf_threshold=args.conf_threshold)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.amp_dtype]
    use_amp = args.amp_dtype != "fp32"

    limit = min(len(dataset), args.max_samples or len(dataset))
    samples = []
    for idx in range(limit):
        views, _full_mask, label, _meta = dataset[idx]
        views = views.unsqueeze(0).to(device)  # [1, V, C, H, W]
        preds, confs, reports = [], [], []
        for node_id, vm in enumerate(view_masks):
            vm_tensor = torch.tensor([vm], dtype=torch.float32, device=device)
            with torch.no_grad():
                if use_amp:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        logits = model(views, view_mask=vm_tensor)
                else:
                    logits = model(views, view_mask=vm_tensor)
            probs = torch.softmax(logits.float(), dim=1)[0]
            conf, pred = probs.max(dim=0)
            preds.append(int(pred.item()))
            confs.append(float(conf.item()))
            reports.append(Report(
                node_id=node_id,
                target_id=idx,
                pred=int(pred.item()),
                conf=float(conf.item()),
                softmax=probs.cpu().tolist(),  # 贝叶斯融合用
            ))
        arb.receive(reports)
        samples.append({
            "target_id": idx,
            "label": int(label),
            "preds": preds,
            "confs": confs,
        })
        if (idx + 1) % args.log_every == 0 or idx + 1 == limit:
            print(f"{idx + 1}/{limit} 已推理", flush=True)

    # 全部样本推理完，统一仲裁
    arb.detect_conflicts()
    arb.arbitrate()
    arb.dispatch_and_rollback()
    stats = arb.stats()

    # 写每样本明细 CSV
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    conflict_targets = {c.target_id for c in arb.conflicts}
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = (
            ["target_id", "label"]
            + [f"pred_n{n + 1}" for n in range(num_nodes)]
            + [f"conf_n{n + 1}" for n in range(num_nodes)]
            + ["conflict", "final_pred", "resolved"]
        )
        writer.writerow(header)
        for sm in samples:
            tid = sm["target_id"]
            is_conflict = tid in conflict_targets
            final = arb.decisions.get(tid)
            final_val = final if final is not None else -1
            resolved = int(is_conflict and final is not None)
            row = (
                [tid, sm["label"]]
                + sm["preds"]
                + sm["confs"]
                + [int(is_conflict), final_val, resolved]
            )
            writer.writerow(row)

    print("\n" + "=" * 56)
    print("多节点冲突仲裁评估结果")
    print("=" * 56)
    print(f"总样本数: {limit}")
    print(f"重叠观测组数: {stats['total_overlaps']}")
    print(f"冲突数: {stats['num_conflicts']}  冲突率: {stats['conflict_rate']:.2%}  (硬指标≤5%)")
    print(f"解决率: {stats['resolve_rate']:.2%}  (硬指标≥90%)")
    print(f"回滚数: {stats['num_rollbacks']}")
    print(f"明细 CSV: {out}")


if __name__ == "__main__":
    main()
