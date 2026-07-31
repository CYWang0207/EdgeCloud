"""Fully retrain the BoxCars MV-ViT baseline on deterministic drift."""

import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from boxcars_dataset import BoxCarsMultiView, VALID_TASKS
from boxcars_drift_dataset import BoxCarsDriftWrapper
from model import EarlyFusionMultiViewViT


def parse_args():
    parser = argparse.ArgumentParser(description="BoxCars drift retraining")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--task", choices=VALID_TASKS, default="make")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-dir", default="./checkpoints/boxcars_make_drift")
    parser.add_argument("--save-prefix", default="boxcars_retrain_mixed")
    parser.add_argument("--model-name", default="vit_small_patch16_224")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--drift-schedule", choices=("none", "light", "mixed", "staged", "highfreq"), default="mixed")
    parser.add_argument("--drift-seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--train-token-mask", action="store_true")
    parser.add_argument("--keep-min", type=float, default=0.7)
    return parser.parse_args()


def load_state(model, checkpoint, device):
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    model.load_state_dict(state)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("BoxCars drift retraining requires CUDA")
    device = torch.device("cuda")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    base_dataset = BoxCarsMultiView(
        args.dataset_path, "train", args.task, args.num_views, transform
    )
    classes = base_dataset.classes
    dataset = BoxCarsDriftWrapper(
        base_dataset, args.drift_schedule, args.drift_seed, normalize=True
    )
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model = EarlyFusionMultiViewViT(
        args.model_name, args.num_views, len(classes), pretrained=False
    ).to(device)
    load_state(model, args.checkpoint, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        loss_sum = correct = count = 0
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            images, view_mask, labels = batch[0], batch[1], batch[2]
            images = images.to(device, non_blocking=True)
            view_mask = view_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            keep_ratios = None
            if args.train_token_mask:
                keep_ratios = args.keep_min + (1 - args.keep_min) * torch.rand(
                    labels.shape[0], args.num_views, device=device
                )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                images, view_mask=view_mask, keep_ratios=keep_ratios,
                token_score_mode="random",
            )
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            count += labels.size(0)
        scheduler.step()
        save_path = os.path.join(
            args.save_dir, f"{args.save_prefix}_epoch_{epoch + 1}.pth"
        )
        torch.save({
            "epoch": epoch, "model": model.state_dict(), "classes": classes,
            "args": vars(args),
        }, save_path)
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={loss_sum / max(count, 1):.4f} "
            f"acc={correct / max(count, 1):.4%} saved={save_path}", flush=True
        )


if __name__ == "__main__":
    main()
