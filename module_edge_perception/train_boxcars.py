"""Train the multi-view ViT baseline on BoxCars116k, with optional DDP."""

import argparse
import os
import random
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from boxcars_dataset import BoxCarsMultiView, VALID_TASKS
from model import EarlyFusionMultiViewViT


def parse_args():
    parser = argparse.ArgumentParser(description="BoxCars116k MV-ViT baseline (DDP)")
    parser.add_argument(
        "--dataset-path",
        default="/root/autodl-tmp/EdgeCloud/data/BoxCars116k_kaggle/BoxCars116k",
    )
    parser.add_argument("--task", choices=VALID_TASKS, default="make")
    parser.add_argument("--save-dir", default="./checkpoints/boxcars_make_baseline")
    parser.add_argument("--model-name", default="vit_small_patch16_224")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="per-GPU batch size")
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="0 means all batches; useful for a smoke test")
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def setup_distributed():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, local_rank, dist.get_rank(), dist.get_world_size()
    device_index = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)
    return False, device_index, 0, 1


def reduce_totals(values, device, distributed):
    totals = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals.tolist()


def make_loaders(args, distributed, rank, world_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    train_set = BoxCarsMultiView(
        args.dataset_path, "train", args.task, args.num_views, train_transform
    )
    val_set = BoxCarsMultiView(
        args.dataset_path, "validation", args.task, args.num_views, eval_transform
    )
    train_sampler = DistributedSampler(
        train_set, world_size, rank, shuffle=True, seed=args.seed
    ) if distributed else None
    val_sampler = DistributedSampler(
        val_set, world_size, rank, shuffle=False
    ) if distributed else None
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_set, sampler=train_sampler, shuffle=train_sampler is None, **common
    )
    val_loader = DataLoader(
        val_set, sampler=val_sampler, shuffle=False, **common
    )
    return train_set, val_set, train_loader, val_loader, train_sampler


def evaluate(model, loader, device, distributed, max_batches, amp_dtype):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for batch_index, (images, view_mask, labels, _) in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            view_mask = view_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(images, view_mask=view_mask)
                loss_sum += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    loss_sum, correct, count = reduce_totals(
        (loss_sum, correct, count), device, distributed
    )
    return loss_sum / count, correct / count


def main():
    args = parse_args()
    distributed, local_rank, rank, world_size = setup_distributed()
    if not torch.cuda.is_available():
        raise RuntimeError("baseline training requires CUDA")
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    train_set, val_set, train_loader, val_loader, train_sampler = make_loaders(
        args, distributed, rank, world_size
    )
    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=args.num_views,
        num_classes=len(train_set.classes),
        pretrained=not args.no_pretrained,
    ).to(device)
    # The baseline never calls the soft-pruning path, so this parameter cannot
    # receive gradients. Freeze it instead of enabling costly DDP unused-parameter scans.
    model.token_mask_token.requires_grad_(False)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    start_epoch = 0
    best_accuracy = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        target = model.module if distributed else model
        target.load_state_dict(checkpoint.get("model", checkpoint))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            best_accuracy = checkpoint.get("best_accuracy", 0.0)

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(
            f"BoxCars baseline: task={args.task}, classes={len(train_set.classes)}, "
            f"train={len(train_set)}, validation={len(val_set)}, GPUs={world_size}, "
            f"per_gpu_batch={args.batch_size}, "
            f"effective_batch={args.batch_size * world_size * args.accumulation_steps}"
        )

    criterion = nn.CrossEntropyLoss()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        correct = 0
        count = 0
        batches_this_epoch = min(
            len(train_loader), args.max_train_batches or len(train_loader)
        )
        for batch_index, (images, view_mask, labels, _) in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            view_mask = view_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            should_step = (
                (batch_index + 1) % args.accumulation_steps == 0
                or (batch_index + 1) == batches_this_epoch
            )
            sync_context = nullcontext()
            if distributed and not should_step:
                sync_context = model.no_sync()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(images, view_mask=view_mask)
                    loss = criterion(outputs, labels) / args.accumulation_steps
                scaler.scale(loss).backward()
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item() * args.accumulation_steps * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
        scheduler.step()
        loss_sum, correct, count = reduce_totals(
            (loss_sum, correct, count), device, distributed
        )
        val_loss, val_accuracy = evaluate(
            model, val_loader, device, distributed, args.max_val_batches, amp_dtype
        )
        if rank == 0:
            print(
                f"epoch={epoch + 1}/{args.epochs} lr={scheduler.get_last_lr()[0]:.6g} "
                f"train_loss={loss_sum / count:.4f} train_acc={correct / count:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}",
                flush=True,
            )
            raw_model = model.module if distributed else model
            state = {
                "epoch": epoch,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_accuracy": max(best_accuracy, val_accuracy),
                "classes": train_set.classes,
                "args": vars(args),
            }
            torch.save(state, os.path.join(args.save_dir, "latest.pth"))
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(state, os.path.join(args.save_dir, "best.pth"))

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
