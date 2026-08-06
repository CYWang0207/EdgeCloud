"""Train a small AdaptFormer expert for synthetic BoxCars visual drift.

This is deliberately label-supervised rather than VLM-class-logit distilled:
BoxCars supplies the real make label, while a generic VLM is not a trustworthy
16-way fine-grained vehicle-make teacher.  Each sample yields an aligned clean
and drifted pair.  The objective combines drifted-image classification with
clean/drift feature consistency and clean-baseline output consistency.
"""

import argparse
import json
import os
import random
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from adaptformer import (
    attach_adaptformer,
    count_adapter_parameters,
    freeze_backbone,
    save_adapter_checkpoint,
    set_adapter_enabled,
)
from boxcars_dataset import BoxCarsMultiView, VALID_TASKS
from boxcars_drift_dataset import PairedBoxCarsDriftDataset, SUPPORTED_TRAIN_DRIFTS
from model import EarlyFusionMultiViewViT


def parse_drift_types(value):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(values) - set(SUPPORTED_TRAIN_DRIFTS))
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            f"drift types must be comma-separated values from "
            f"{SUPPORTED_TRAIN_DRIFTS}; unknown={unknown}"
        )
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="BoxCars synthetic-drift adapter training")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--task", choices=VALID_TASKS, default="make")
    parser.add_argument("--expert-name", default="general")
    parser.add_argument(
        "--drift-types", type=parse_drift_types,
        default=SUPPORTED_TRAIN_DRIFTS,
        help="comma-separated expert domain; one type for a specialist",
    )
    parser.add_argument("--severity-min", type=float, default=0.35)
    parser.add_argument("--severity-max", type=float, default=1.0)
    parser.add_argument("--clean-probability", type=float, default=0.15)
    parser.add_argument("--independent-view-drifts", action="store_true")
    parser.add_argument("--feature-weight", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=0.20)
    parser.add_argument("--condition-dim", type=int, default=128)
    parser.add_argument("--vlm-condition-cache", default="")
    parser.add_argument("--vlm-weight", type=float, default=0.10)
    parser.add_argument("--quality-weight", type=float, default=0.10)
    parser.add_argument("--teacher-condition-probability", type=float, default=0.50)
    parser.add_argument("--null-condition-probability", type=float, default=0.20)
    parser.add_argument("--train-norm", action="store_true")
    parser.add_argument("--train-head", action="store_true")
    parser.add_argument("--save-dir", default="./checkpoints/boxcars_drift_adapters")
    parser.add_argument("--model-name", default="vit_small_patch16_224")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def setup_distributed():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return True, local_rank, dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return False, 0, 0, 1


def reduce(values, device, distributed):
    result = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(result)
    return result.tolist()


def load_baseline(model, path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        state = payload.get("model", payload.get("state_dict", payload))
    else:
        state = payload
    model.load_state_dict(state, strict=True)


def make_dataset(args, split, clean_probability):
    transform_items = [transforms.Resize((224, 224))]
    if split == "train":
        transform_items.append(transforms.RandomHorizontalFlip())
    transform_items.append(transforms.ToTensor())
    base = BoxCarsMultiView(
        args.dataset_path, split, args.task, args.num_views,
        transforms.Compose(transform_items),
    )
    return PairedBoxCarsDriftDataset(
        base,
        drift_types=args.drift_types,
        severity_min=args.severity_min,
        severity_max=args.severity_max,
        clean_probability=clean_probability,
        independent_view_drifts=args.independent_view_drifts,
        seed=args.seed + (0 if split == "train" else 1_000_003),
    )


def make_loaders(args, distributed, rank, world_size):
    train_set = make_dataset(args, "train", args.clean_probability)
    # Validation measures the requested drift domain only; clean retention is
    # reported separately by forwarding the paired clean image.
    val_set = make_dataset(args, "validation", 0.0)
    train_sampler = DistributedSampler(
        train_set, world_size, rank, shuffle=True, seed=args.seed
    ) if distributed else None
    val_sampler = DistributedSampler(
        val_set, world_size, rank, shuffle=False
    ) if distributed else None
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_set, sampler=train_sampler, shuffle=train_sampler is None, **common
    )
    val_loader = DataLoader(
        val_set, sampler=val_sampler, shuffle=False, **common
    )
    return train_set, train_loader, val_loader, train_sampler


def paired_loss(
    model, clean, drifted, view_mask, labels, quality_targets,
    teacher_condition, active_condition, feature_weight, consistency_weight,
    vlm_weight, quality_weight,
):
    raw = model.module if isinstance(model, DDP) else model
    # The disabled pass is the frozen clean-domain reference.  norm/head are
    # detached here, so only the single drifted adapted pass receives gradients.
    set_adapter_enabled(raw, False)
    with torch.no_grad():
        clean_logits, clean_target = raw(
            clean, view_mask=view_mask, return_features=True,
            apply_quality_gate=False,
        )
    set_adapter_enabled(raw, True)
    drift_logits, aux = model(
        drifted, view_mask=view_mask, condition_vector=active_condition,
        return_aux=True,
    )
    drift_features = aux["features"]
    classification = F.cross_entropy(drift_logits, labels)
    consistency = F.kl_div(
        F.log_softmax(drift_logits.float(), dim=1),
        F.softmax(clean_logits.float(), dim=1),
        reduction="batchmean",
    )
    alignment = 1.0 - F.cosine_similarity(
        drift_features.float(), clean_target.float(), dim=1
    ).mean()
    vlm_distill = drift_features.new_zeros(())
    if teacher_condition is not None:
        vlm_distill = 1.0 - F.cosine_similarity(
            aux["edge_condition"].float(), teacher_condition.float(), dim=1
        ).mean()
    quality = F.mse_loss(
        aux["view_quality"].float(), quality_targets.float()
    )
    total = (
        classification
        + consistency_weight * consistency
        + feature_weight * alignment
        + vlm_weight * vlm_distill
        + quality_weight * quality
    )
    return (
        total, classification.detach(), consistency.detach(),
        alignment.detach(), vlm_distill.detach(), quality.detach(), drift_logits,
    )


def load_condition_cache(path, split, expected_length, condition_dim):
    if not path:
        return None
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        payload = payload.get(split, payload.get("conditions"))
    if not isinstance(payload, torch.Tensor) or payload.ndim != 2:
        raise ValueError("VLM cache must contain a [samples, condition_dim] tensor")
    if payload.shape != (expected_length, condition_dim):
        raise ValueError(
            f"VLM cache shape must be {(expected_length, condition_dim)}, "
            f"got {tuple(payload.shape)}"
        )
    return F.normalize(payload.float(), dim=-1)


def choose_active_condition(batch_size, teacher_condition, args, device):
    unit = random.random()
    teacher_probability = (
        args.teacher_condition_probability if teacher_condition is not None else 0.0
    )
    if teacher_condition is not None and unit < teacher_probability:
        return teacher_condition
    if unit < teacher_probability + args.null_condition_probability:
        return torch.zeros(
            batch_size, args.condition_dim, device=device, dtype=torch.float32
        )
    return None


@torch.no_grad()
def evaluate(model, loader, device, distributed, max_batches, amp_dtype):
    model.eval()
    totals = [0.0, 0.0, 0.0, 0.0]
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        clean, drifted, view_mask, labels = batch[:4]
        clean = clean.to(device, non_blocking=True)
        drifted = drifted.to(device, non_blocking=True)
        view_mask = view_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            drift_logits = model(drifted, view_mask=view_mask)
            clean_logits = model(clean, view_mask=view_mask)
        totals[0] += F.cross_entropy(drift_logits, labels, reduction="sum").item()
        totals[1] += (drift_logits.argmax(1) == labels).sum().item()
        totals[2] += (clean_logits.argmax(1) == labels).sum().item()
        totals[3] += labels.numel()
    loss_sum, drift_correct, clean_correct, count = reduce(totals, device, distributed)
    return loss_sum / max(count, 1), drift_correct / max(count, 1), clean_correct / max(count, 1)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("drift adapter training requires CUDA")
    if min(args.feature_weight, args.consistency_weight, args.vlm_weight,
           args.quality_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    if not 0 <= args.teacher_condition_probability <= 1:
        raise ValueError("teacher condition probability must be in [0, 1]")
    if not 0 <= args.null_condition_probability <= 1:
        raise ValueError("null condition probability must be in [0, 1]")
    if args.teacher_condition_probability + args.null_condition_probability > 1:
        raise ValueError("teacher + null condition probabilities must not exceed 1")
    distributed, local_rank, rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    train_set, train_loader, val_loader, train_sampler = make_loaders(
        args, distributed, rank, world_size
    )
    model = EarlyFusionMultiViewViT(
        args.model_name, args.num_views, len(train_set.classes), pretrained=False
    )
    load_baseline(model, args.baseline_checkpoint)
    model.attach_drift_conditioner(condition_dim=args.condition_dim)
    attach_adaptformer(model, r=args.r, condition_dim=args.condition_dim)
    freeze_backbone(model)
    if not args.train_norm:
        for parameter in model.norm.parameters():
            parameter.requires_grad = False
    if not args.train_head:
        for parameter in model.head.parameters():
            parameter.requires_grad = False
    model.to(device)
    if distributed:
        # The clean reference forwards with adapters disabled, so some adapter
        # parameters can be absent from a given backward graph.  This is
        # expected for the paired objective and must be declared to DDP.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw = model.module if distributed else model
    condition_cache = load_condition_cache(
        args.vlm_condition_cache, "train", len(train_set), args.condition_dim
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")
    expert_dir = os.path.join(args.save_dir, args.expert_name)
    if rank == 0:
        os.makedirs(expert_dir, exist_ok=True)
        print(
            f"expert={args.expert_name} drifts={args.drift_types} "
            f"adapter={count_adapter_parameters(raw):,} train={len(train_set)}",
            flush=True,
        )
    best_drift_accuracy = -1.0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = [0.0] * 7
        limit = min(len(train_loader), args.max_train_batches or len(train_loader))
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            clean, drifted, view_mask, labels = [
                item.to(device, non_blocking=True) for item in batch[:4]
            ]
            quality_targets = batch[-2].to(device, non_blocking=True)
            sample_indices = batch[-1].long()
            teacher_condition = None
            if condition_cache is not None:
                teacher_condition = condition_cache[sample_indices].to(
                    device, non_blocking=True
                )
            active_condition = choose_active_condition(
                labels.shape[0], teacher_condition, args, device
            )
            should_step = (batch_index + 1) % args.accumulation_steps == 0 or batch_index + 1 == limit
            sync = model.no_sync() if distributed and not should_step else nullcontext()
            with sync:
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss, cls, consistency, alignment, vlm_distill, quality, logits = paired_loss(
                        model, clean, drifted, view_mask, labels, quality_targets,
                        teacher_condition, active_condition,
                        args.feature_weight, args.consistency_weight,
                        args.vlm_weight, args.quality_weight,
                    )
                    scaled_loss = loss / args.accumulation_steps
                scaler.scale(scaled_loss).backward()
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            n = labels.numel()
            totals[0] += loss.item() * n
            totals[1] += cls.item() * n
            totals[2] += consistency.item() * n
            totals[3] += alignment.item() * n
            totals[4] += vlm_distill.item() * n
            totals[5] += quality.item() * n
            totals[6] += n
        scheduler.step()
        totals = reduce(totals, device, distributed)
        val_loss, drift_acc, clean_acc = evaluate(
            model, val_loader, device, distributed, args.max_val_batches, amp_dtype
        )
        if rank == 0:
            n = max(totals[6], 1)
            print(
                f"epoch={epoch + 1}/{args.epochs} loss={totals[0]/n:.4f} "
                f"cls={totals[1]/n:.4f} consistency={totals[2]/n:.4f} "
                f"align={totals[3]/n:.4f} vlm={totals[4]/n:.4f} "
                f"quality={totals[5]/n:.4f} val_loss={val_loss:.4f} "
                f"val_drift_acc={drift_acc:.4f} val_clean_acc={clean_acc:.4f}",
                flush=True,
            )
            metadata = {
                "epoch": epoch,
                "expert_name": args.expert_name,
                "drift_types": list(args.drift_types),
                "severity_range": [args.severity_min, args.severity_max],
                "validation_drift_accuracy": drift_acc,
                "validation_clean_accuracy": clean_acc,
                "r": args.r,
                "condition_dim": args.condition_dim,
                "task": args.task,
                "classes": train_set.classes,
                "args": vars(args),
            }
            save_adapter_checkpoint(os.path.join(expert_dir, "latest.pth"), raw, **metadata)
            if drift_acc > best_drift_accuracy:
                best_drift_accuracy = drift_acc
                save_adapter_checkpoint(os.path.join(expert_dir, "best.pth"), raw, **metadata)
                with open(os.path.join(expert_dir, "manifest.json"), "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, ensure_ascii=False, indent=2)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
