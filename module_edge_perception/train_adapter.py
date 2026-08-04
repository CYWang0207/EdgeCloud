"""云端 VLM 软标签蒸馏 → 冻结主干只训 AdaptFormer adapter（张晨 8/4-8/5 任务）。

对应 CLAUDE.md 第八节 + 接口契约接口3（VLM Oracle 输出软标签）。
teacher 用预计算软标签文件：云端 VLM（或先 baseline 自蒸馏占位）离线把每个样本的
logits 落盘成 .npz，本脚本只读文件训练，不依赖任何 VLM 推理资源。

训练口径：
- 主干：加载 BoxCars 基线 checkpoint（EarlyFusionMultiViewViT），attach_adaptformer(r=32)
  后 freeze_backbone，只放开 adapter(含 scale) + norm + head。
- 损失：软标签蒸馏
  loss = (1-alpha)*CE(student, label) + alpha*T^2*KL(softmax(student/T) || softmax(teacher/T))
- 产物：adapter-only checkpoint（adaptformer.save_adapter_checkpoint），仅含
  blocks.*.mlp.adapter.* / blocks.*.mlp.scale / norm / head ≈ 0.3M 参 × 4B ≈ 1.2MB，
  对应 u=1 S_adapter 通信口径，可直接被 evaluate_rl_policy_on_mvvit.py 加载。

软标签文件格式（.npz，teacher 导出端生成，与数据集索引一一对齐）：
    train_logits: [N_train, C]   # C = num_classes（BoxCars make 任务 = 16）
    val_logits:   [N_val, C]
    test_logits:  [N_test, C]    # 可选
N_* 必须等于对应 split 的样本数。将来接真 VLM（InternVL/Qwen-VL）只改导出端。
"""

import argparse
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from boxcars_dataset import BoxCarsMultiView, VALID_TASKS
from model import EarlyFusionMultiViewViT
from adaptformer import (
    attach_adaptformer,
    count_adapter_parameters,
    freeze_backbone,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)


class TeacherSoftLabelDataset(torch.utils.data.Dataset):
    """把预计算 teacher logits 按索引对齐到 BoxCarsMultiView 上。

    软标签跟随样本索引，因此 shuffle / DDP 采样天然对齐，无需手工维护 permutation。
    __getitem__ 返回 (views, view_mask, label, teacher_logits)。
    """

    def __init__(self, base, logits):
        self.base = base
        self.logits = torch.as_tensor(np.asarray(logits, dtype=np.float32))
        if len(self.logits) != len(base):
            raise ValueError(
                f"soft label 条数 {len(self.logits)} != 数据集样本数 {len(base)}"
            )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        views, view_mask, label, _ = self.base[idx]
        return views, view_mask, label, self.logits[idx]


def load_soft_labels(path):
    if path is None:
        return {}
    data = np.load(path)
    out = {}
    for key in ("train_logits", "val_logits", "test_logits"):
        if key in data:
            out[key] = torch.as_tensor(data[key], dtype=torch.float32)
    if "train_logits" not in out:
        raise ValueError(f"soft label 文件缺少 train_logits: {path}")
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="BoxCars116k Adapter 蒸馏训练（冻结主干只训 adapter）"
    )
    parser.add_argument(
        "--dataset-path",
        default="/root/autodl-tmp/EdgeCloud/data/BoxCars116k_kaggle/BoxCars116k",
    )
    parser.add_argument("--soft-labels", required=True,
                        help="teacher 导出的 .npz（train_logits/val_logits，见文件头格式说明）")
    parser.add_argument("--baseline-checkpoint", required=True,
                        help="BoxCars baseline 权重（EarlyFusionMultiViewViT 主干）")
    parser.add_argument("--task", choices=VALID_TASKS, default="make")
    parser.add_argument("--save-dir", default="./checkpoints/boxcars_make_adapter")
    parser.add_argument("--model-name", default="vit_small_patch16_224")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--r", type=int, default=32, help="adapter 瓶颈维度")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4, help="per-GPU batch size")
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="KD 损失权重：loss=(1-alpha)*CE + alpha*T^2*KL")
    parser.add_argument("--temperature", type=float, default=4.0, help="蒸馏温度 T")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="0 表示跑完所有 batch；>0 用于冒烟测试")
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


def make_loaders(args, distributed, rank, world_size, soft_labels):
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
    if "train_logits" in soft_labels:
        train_set = TeacherSoftLabelDataset(train_set, soft_labels["train_logits"])
    if "val_logits" in soft_labels:
        val_set = TeacherSoftLabelDataset(val_set, soft_labels["val_logits"])

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


def distillation_loss(student_logits, teacher_logits, labels, alpha, temperature):
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, labels)
    return (1.0 - alpha) * ce_loss + alpha * kd_loss


def evaluate(model, loader, device, distributed, max_batches, amp_dtype):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            images, view_mask, labels, _ = batch
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
        raise RuntimeError("adapter 蒸馏训练需要 CUDA（在 AutoDL 上跑）")
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    soft_labels = load_soft_labels(args.soft_labels)
    train_set, val_set, train_loader, val_loader, train_sampler = make_loaders(
        args, distributed, rank, world_size, soft_labels
    )

    # --- 模型：基线主干 + 常挂 adapter + 冻结主干 ---
    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=args.num_views,
        num_classes=len(train_set.classes),
        pretrained=False,
    ).to(device)
    baseline_state = torch.load(args.baseline_checkpoint, map_location="cpu")
    if isinstance(baseline_state, dict) and "model" in baseline_state:
        baseline_state = baseline_state["model"]
    missing, unexpected = model.load_state_dict(baseline_state, strict=False)
    if rank == 0:
        print(f"基线主干加载: missing={len(missing)}, unexpected={len(unexpected)}")
    attach_adaptformer(model, r=args.r)
    freeze_backbone(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    n_adapter = count_adapter_parameters(model.module if distributed else model)
    if rank == 0:
        print(
            f"adapter 参数: {n_adapter:,} ({n_adapter / 1e6:.3f}M)  "
            f"≈ {n_adapter * 4 / 1e6:.2f}MB（u=1 下发口径）"
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    start_epoch = 0
    best_accuracy = 0.0
    if args.resume:
        raw = model.module if distributed else model
        load_adapter_checkpoint(raw, args.resume, device=device)
        resume_state = torch.load(args.resume, map_location="cpu")
        start_epoch = resume_state.get("epoch", 0) + 1
        best_accuracy = resume_state.get("best_accuracy", 0.0)

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(
            f"Adapter 蒸馏: task={args.task}, classes={len(train_set.classes)}, "
            f"r={args.r}, alpha={args.alpha}, T={args.temperature}, "
            f"train={len(train_set)}, validation={len(val_set)}, GPUs={world_size}"
        )

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
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            images, view_mask, labels, teacher_logits = batch
            images = images.to(device, non_blocking=True)
            view_mask = view_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            teacher_logits = teacher_logits.to(device, non_blocking=True)
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
                    loss = distillation_loss(
                        outputs, teacher_logits, labels,
                        args.alpha, args.temperature,
                    ) / args.accumulation_steps
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
            meta = {
                "epoch": epoch,
                "r": args.r,
                "task": args.task,
                "classes": train_set.classes,
                "best_accuracy": max(best_accuracy, val_accuracy),
                "alpha": args.alpha,
                "temperature": args.temperature,
                "teacher_soft_labels": args.soft_labels,
                "args": vars(args),
            }
            save_adapter_checkpoint(
                os.path.join(args.save_dir, "adapter_latest.pth"), raw_model, **meta
            )
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                save_adapter_checkpoint(
                    os.path.join(args.save_dir, "adapter_best.pth"), raw_model, **meta
                )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
