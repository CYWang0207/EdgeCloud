"""Train ModelNet40 conditional AdaptFormer under calibrated camera drift."""
import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from adaptformer import attach_adaptformer, count_adapter_parameters, freeze_backbone, save_adapter_checkpoint, set_adapter_enabled
from dataset import ModelNet40MultiView
from model import EarlyFusionMultiViewViT
from modelnet_camera_drift_dataset import DRIFTS, PairedModelNetDrift


def drift_types(value):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or set(values) - set(DRIFTS):
        raise argparse.ArgumentTypeError(f"drift types must come from {DRIFTS}")
    return values


def fixed_severities(value):
    result = {}
    for item in value.split(","):
        name, separator, raw = item.partition("=")
        if not separator or name not in DRIFTS or not 0 <= float(raw) <= 1:
            raise argparse.ArgumentTypeError("use TYPE=0..1 entries")
        result[name] = float(raw)
    return result


def args_parser():
    parser = argparse.ArgumentParser(description="ModelNet40 calibrated conditional Adapter")
    parser.add_argument("--dataset-path", required=True); parser.add_argument("--baseline-checkpoint", required=True); parser.add_argument("--save-dir", required=True)
    parser.add_argument("--drift-types", type=drift_types, default=("illumination", "defocus", "sensor_noise"))
    parser.add_argument("--drift-weights", type=float, nargs="+", default=(.30, .30, .40))
    parser.add_argument("--fixed-severities", type=fixed_severities, default={"illumination": 1., "defocus": .2, "sensor_noise": .4})
    parser.add_argument("--clean-probability", type=float, default=.2); parser.add_argument("--condition-dim", type=int, default=128)
    parser.add_argument("--vlm-condition-cache", default=""); parser.add_argument("--vlm-weight", type=float, default=.10); parser.add_argument("--teacher-condition-probability", type=float, default=.5); parser.add_argument("--null-condition-probability", type=float, default=.2)
    parser.add_argument("--feature-weight", type=float, default=.25); parser.add_argument("--consistency-weight", type=float, default=.20); parser.add_argument("--quality-weight", type=float, default=0.)
    parser.add_argument("--model-name", default="vit_small_patch16_224"); parser.add_argument("--num-views", type=int, default=4); parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--lr", type=float, default=2e-4); parser.add_argument("--weight-decay", type=float, default=.05); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16"); parser.add_argument("--max-train-batches", type=int, default=0); parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def load_baseline(model, path):
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload, strict=True)


def dataset(args, split, clean_probability):
    base = ModelNet40MultiView(args.dataset_path, "train" if split == "train" else "test", transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), args.num_views)
    return PairedModelNetDrift(base, args.drift_types, clean_probability=clean_probability, seed=args.seed + (0 if split == "train" else 1_000_003), drift_weights=args.drift_weights, fixed_severities=args.fixed_severities, return_metadata=True)


def load_cache(path, expected_length, args):
    if not path: return None
    payload = torch.load(path, map_location="cpu")
    conditions = payload.get("train") if isinstance(payload, dict) else None
    source = payload.get("source_metadata") if isinstance(payload, dict) else None
    expected = {"drift_generator": "modelnet_camera_drift_dataset", "drift_types": list(args.drift_types), "fixed_severities": args.fixed_severities, "seed": args.seed}
    if not isinstance(conditions, torch.Tensor) or conditions.shape != (expected_length, args.condition_dim): raise ValueError("VLM cache shape must be [train_samples, condition_dim]")
    if not isinstance(source, dict) or any(source.get(k) != v for k, v in expected.items()): raise ValueError("VLM cache provenance does not match calibrated ModelNet drift")
    return F.normalize(conditions.float(), dim=-1)


@torch.no_grad()
def evaluate(model, loader, device, dtype, limit):
    model.eval(); total = [0., 0., 0.]
    for step, batch in enumerate(loader):
        if limit and step >= limit: break
        clean, corrupt, labels = (item.to(device, non_blocking=True) for item in batch[:3])
        with torch.autocast("cuda", dtype=dtype): clean_logits = model(clean); corrupt_logits = model(corrupt)
        total[0] += (clean_logits.argmax(1) == labels).sum().item(); total[1] += (corrupt_logits.argmax(1) == labels).sum().item(); total[2] += labels.numel()
    return total[0] / max(total[2], 1), total[1] / max(total[2], 1)


def main():
    args = args_parser()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    if len(args.drift_types) != len(args.drift_weights): raise ValueError("drift-weights length must match drift-types")
    if not 0 <= args.teacher_condition_probability <= 1 or not 0 <= args.null_condition_probability <= 1 or args.teacher_condition_probability + args.null_condition_probability > 1: raise ValueError("invalid condition probabilities")
    random.seed(args.seed); torch.manual_seed(args.seed); device = torch.device("cuda"); dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    train, validation = dataset(args, "train", args.clean_probability), dataset(args, "test", 0.)
    loaders = [DataLoader(data, batch_size=args.batch_size, shuffle=(i == 0), num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0) for i, data in enumerate((train, validation))]
    model = EarlyFusionMultiViewViT(args.model_name, args.num_views, len(train.classes), pretrained=False); load_baseline(model, args.baseline_checkpoint); model.attach_drift_conditioner(args.condition_dim); attach_adaptformer(model, args.r, condition_dim=args.condition_dim); freeze_backbone(model); model.to(device)
    cache = load_cache(args.vlm_condition_cache, len(train), args); opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs); scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")
    os.makedirs(args.save_dir, exist_ok=True); best = -1.
    print(f"ModelNet40 conditional camera adapter: train={len(train)} test={len(validation)} adapter={count_adapter_parameters(model)} drifts={args.drift_types} fixed={args.fixed_severities}", flush=True)
    for epoch in range(args.epochs):
        model.train(); sums = [0., 0., 0.]
        for step, batch in enumerate(loaders[0]):
            if args.max_train_batches and step >= args.max_train_batches: break
            clean, corrupt, labels = (item.to(device, non_blocking=True) for item in batch[:3]); indices = batch[-1].long(); teacher = cache[indices].to(device, non_blocking=True) if cache is not None else None
            choice = random.random(); active = teacher if teacher is not None and choice < args.teacher_condition_probability else (torch.zeros(labels.shape[0], args.condition_dim, device=device) if choice < args.teacher_condition_probability + args.null_condition_probability else None)
            set_adapter_enabled(model, False)
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype): clean_logits, clean_features = model(clean, return_features=True, apply_quality_gate=False)
            set_adapter_enabled(model, True); opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                logits, aux = model(corrupt, condition_vector=active, return_aux=True); classification = F.cross_entropy(logits, labels); consistency = F.kl_div(F.log_softmax(logits.float(), 1), F.softmax(clean_logits.float(), 1), reduction="batchmean"); alignment = 1 - F.cosine_similarity(aux["features"].float(), clean_features.float(), dim=1).mean(); vlm = 1 - F.cosine_similarity(aux["edge_condition"].float(), teacher.float(), dim=1).mean() if teacher is not None else logits.new_zeros(()); loss = classification + args.consistency_weight * consistency + args.feature_weight * alignment + args.vlm_weight * vlm
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sums[0] += loss.item() * labels.numel(); sums[1] += (logits.argmax(1) == labels).sum().item(); sums[2] += labels.numel()
        scheduler.step(); clean_acc, drift_acc = evaluate(model, loaders[1], device, dtype, args.max_val_batches)
        meta = {"epoch": epoch + 1, "dataset": "modelnet40", "drift_types": list(args.drift_types), "drift_weights": list(args.drift_weights), "fixed_severities": args.fixed_severities, "drift_generator": "modelnet_camera_drift_dataset", "validation_clean_accuracy": clean_acc, "validation_drift_accuracy": drift_acc, "condition_dim": args.condition_dim, "args": vars(args)}
        print(f"epoch={epoch + 1}/{args.epochs} loss={sums[0]/max(sums[2],1):.4f} train={sums[1]/max(sums[2],1):.4f} clean={clean_acc:.4f} drift={drift_acc:.4f}", flush=True); save_adapter_checkpoint(os.path.join(args.save_dir, "latest.pth"), model, **meta)
        if drift_acc > best:
            best = drift_acc; save_adapter_checkpoint(os.path.join(args.save_dir, "best.pth"), model, **meta)
            with open(os.path.join(args.save_dir, "manifest.json"), "w", encoding="utf-8") as handle: json.dump(meta, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__": main()
