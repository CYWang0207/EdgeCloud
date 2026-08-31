"""Train drift-conditioned prompt tokens for the BoxCars MV-ViT baseline."""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from boxcars_dataset import BoxCarsMultiView, VALID_TASKS
from boxcars_drift_dataset import BoxCarsDriftWrapper
from model import EarlyFusionMultiViewViT
from prompt_tuning.prompt_model import PromptGenerator


COMMON_DIR = Path(__file__).resolve().parent.parent / "common"
sys.path.insert(0, str(COMMON_DIR))
from drift_dataset import DRIFT_TYPE_TO_CONDITION, condition_id_for_drift  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="BoxCars prompt tuning under drift")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--task", choices=VALID_TASKS, default="make")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-path", default="./checkpoints/boxcars_make_prompt/best.pth")
    parser.add_argument("--model-name", default="vit_small_patch16_224")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-prompt-tokens", type=int, default=4)
    parser.add_argument("--drift-schedule", choices=("none", "light", "mixed", "staged", "highfreq"), default="mixed")
    parser.add_argument("--drift-seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--train-token-mask", action="store_true")
    parser.add_argument("--keep-min", type=float, default=0.7)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("BoxCars prompt training requires CUDA")
    device = torch.device("cuda")
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor()
    ])
    base_dataset = BoxCarsMultiView(
        args.dataset_path, "train", args.task, args.num_views, transform
    )
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
        args.model_name, args.num_views, len(base_dataset.classes), pretrained=False
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    model.load_state_dict(state)

    prompt_gen = PromptGenerator(
        model.cls_token.shape[-1], args.num_prompt_tokens,
        max(DRIFT_TYPE_TO_CONDITION.values()) + 1,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.norm.parameters():
        parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW([
        {"params": prompt_gen.parameters(), "lr": args.lr},
        {"params": model.norm.parameters(), "lr": args.head_lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0

    for epoch in range(args.epochs):
        model.train()
        prompt_gen.train()
        loss_sum = correct = count = 0
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            images, view_mask, labels, drift_types = batch[0], batch[1], batch[2], batch[4]
            images = images.to(device, non_blocking=True)
            view_mask = view_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            condition_ids = torch.tensor(
                [condition_id_for_drift(item) for item in drift_types],
                dtype=torch.long, device=device,
            )
            keep_ratios = None
            if args.train_token_mask:
                keep_ratios = args.keep_min + (1 - args.keep_min) * torch.rand(
                    labels.shape[0], args.num_views, device=device
                )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                images, view_mask=view_mask, keep_ratios=keep_ratios,
                token_score_mode="random", prompt_tokens=prompt_gen(condition_ids),
            )
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            count += labels.size(0)
        accuracy = correct / max(count, 1)
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={loss_sum / max(count, 1):.4f} "
            f"acc={accuracy:.4%}", flush=True
        )
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            save_dir = os.path.dirname(args.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save({
                "prompt_gen": prompt_gen.state_dict(),
                "vit_norm": model.norm.state_dict(),
                "vit_head": model.head.state_dict(),
                "classes": base_dataset.classes,
                "num_conditions": max(DRIFT_TYPE_TO_CONDITION.values()) + 1,
                "num_prompt_tokens": args.num_prompt_tokens,
                "drift_type_to_condition": DRIFT_TYPE_TO_CONDITION,
                "args": vars(args),
            }, args.save_path)
            print(f"saved={args.save_path}", flush=True)


if __name__ == "__main__":
    main()
