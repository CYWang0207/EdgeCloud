import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import ModelNet40MultiView
from model import EarlyFusionMultiViewViT


def parse_args():
    parser = argparse.ArgumentParser(description="Train pruning-aware multi-view ViT.")
    parser.add_argument("--dataset-path", type=str, default="./data/modelnet40v2png_ori4")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--model-name", type=str, default="vit_small_patch16_224")
    parser.add_argument("--num-classes", type=int, default=40)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--full-view-prob", type=float, default=0.70)
    parser.add_argument("--drop-one-view-prob", type=float, default=0.20)
    parser.add_argument("--keep-min-start", type=float, default=0.85)
    parser.add_argument("--keep-min-final", type=float, default=0.70)
    return parser.parse_args()


def sample_view_mask(batch_size, num_views, device, full_prob, drop_one_prob):
    view_mask = torch.ones(batch_size, num_views, device=device)
    probs = torch.rand(batch_size, device=device)
    drop_one_end = full_prob + drop_one_prob

    for b in range(batch_size):
        if probs[b] < full_prob:
            missing_count = 0
        elif probs[b] < drop_one_end:
            missing_count = 1
        else:
            missing_count = min(2, num_views - 1)

        if missing_count > 0:
            missing_views = torch.randperm(num_views, device=device)[:missing_count]
            view_mask[b, missing_views] = 0.0

    return view_mask


def sample_keep_ratios(batch_size, num_views, epoch_idx, total_epochs, device, keep_min_start, keep_min_final):
    if total_epochs <= 1:
        keep_floor = keep_min_final
    else:
        progress = epoch_idx / float(total_epochs - 1)
        keep_floor = keep_min_start + (keep_min_final - keep_min_start) * progress

    random_part = torch.rand(batch_size, num_views, device=device)
    keep_ratios = keep_floor + (1.0 - keep_floor) * random_part
    return keep_ratios.clamp(keep_min_final, 1.0)


def load_resume_if_needed(model, resume_path, device):
    if not resume_path:
        return
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"找不到 resume 权重: {resume_path}")

    state = torch.load(resume_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"已从 {resume_path} 加载权重。missing={len(missing)}, unexpected={len(unexpected)}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的设备: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        train_dataset = ModelNet40MultiView(
            root_dir=args.dataset_path,
            split="train",
            transform=train_transform,
            num_views=args.num_views
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        print(f"成功加载训练集，共有 {len(train_dataset)} 个 {args.num_views}视角样本。")
    except Exception as e:
        print(f"数据加载失败，请检查数据集路径 {args.dataset_path} 是否正确。\n错误信息: {e}")
        return

    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=args.num_views,
        num_classes=args.num_classes,
        pretrained=True
    ).to(device)
    load_resume_if_needed(model, args.resume, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("开始剪枝感知训练...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        keep_sum = 0.0
        active_view_sum = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)

            view_mask = sample_view_mask(
                batch_size,
                args.num_views,
                device,
                args.full_view_prob,
                args.drop_one_view_prob
            )
            keep_ratios = sample_keep_ratios(
                batch_size,
                args.num_views,
                epoch,
                args.epochs,
                device,
                args.keep_min_start,
                args.keep_min_final
            )

            optimizer.zero_grad()
            outputs = model(
                images,
                view_mask=view_mask,
                keep_ratios=keep_ratios,
                token_score_mode="random"
            )
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_size
            correct += (predicted == labels).sum().item()
            keep_sum += keep_ratios.sum().item()
            active_view_sum += view_mask.sum().item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                current_lr = optimizer.param_groups[0]["lr"]
                avg_keep = keep_sum / (total_samples * args.num_views)
                avg_active_views = active_view_sum / total_samples
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}], "
                    f"Step [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}, "
                    f"AvgKeep: {avg_keep:.3f}, AvgViews: {avg_active_views:.2f}"
                )

        scheduler.step()

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total_samples
        print(f"==> Epoch {epoch + 1} 完成! 平均 Loss: {epoch_loss:.4f}, 训练准确率: {epoch_acc:.2f}%")

        save_path = os.path.join(args.save_dir, f"mv_vit_token_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"已保存模型权重至: {save_path}\n")

    print("全部剪枝感知训练完成！")


if __name__ == "__main__":
    main()
