import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from dataset import ModelNet40MultiView
from model import EarlyFusionMultiViewViT


ROOT_DIR = Path(__file__).resolve().parent
COMMON_DIR = ROOT_DIR.parent / "common"
sys.path.insert(0, str(COMMON_DIR))

from drift_dataset import DeterministicDriftWrapper  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain token-aware MV-VIT on drifted data.")
    parser.add_argument("--dataset-path", type=str, default="./data/modelnet40v2png_ori4")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/mv_vit_token_epoch_19.pth")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--save-prefix", type=str, default="mv_vit_retrain_mixed")
    parser.add_argument("--model-name", type=str, default="vit_small_patch16_224")
    parser.add_argument("--num-classes", type=int, default=40)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--drift-schedule", type=str, default="mixed", choices=["none", "light", "mixed", "staged", "highfreq"])
    parser.add_argument("--drift-seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-token-mask", action="store_true")
    parser.add_argument("--keep-min", type=float, default=0.7)
    return parser.parse_args()


def build_loader(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    dataset = ModelNet40MultiView(
        root_dir=args.dataset_path,
        split="train",
        transform=transform,
        num_views=args.num_views,
    )
    dataset = DeterministicDriftWrapper(
        dataset,
        schedule=args.drift_schedule,
        seed=args.drift_seed,
        normalize=True,
    )

    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_model(args, device):
    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=args.num_views,
        num_classes=args.num_classes,
        pretrained=False,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"权重兼容加载: missing={len(missing)}, unexpected={len(unexpected)}")
    return model


def sample_keep_ratios(batch_size, num_views, keep_min, device):
    return keep_min + (1.0 - keep_min) * torch.rand(batch_size, num_views, device=device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的设备: {device}")

    loader = build_loader(args)
    model = load_model(args, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _drift_types, _severities, _struct_drifts in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            keep_ratios = None
            if args.train_token_mask:
                keep_ratios = sample_keep_ratios(
                    labels.shape[0],
                    args.num_views,
                    args.keep_min,
                    device,
                )

            optimizer.zero_grad()
            outputs = model(
                images,
                keep_ratios=keep_ratios,
                token_score_mode="random",
            )
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        epoch_acc = correct / max(total, 1)
        epoch_loss = total_loss / max(len(loader), 1)
        save_path = os.path.join(args.save_dir, f"{args.save_prefix}_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss={epoch_loss:.4f} Acc={epoch_acc:.4%}")
        print(f"已保存重训练权重至: {save_path}")

    print("漂移重训练完成！")


if __name__ == "__main__":
    main()
