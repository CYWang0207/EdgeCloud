"""Export index-aligned BoxCars teacher logits from a frozen baseline.

The output is consumed by train_adapter.py and contains raw logits. Temperature
scaling is intentionally performed by the distillation loss, not here.
"""

import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from boxcars_dataset import BoxCarsMultiView, VALID_TASKS
from model import EarlyFusionMultiViewViT
from train_adapter import dataset_fingerprint


def parse_args():
    parser = argparse.ArgumentParser(description="导出 BoxCars baseline teacher logits")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output", default="soft_labels_boxcars.npz")
    parser.add_argument("--task", choices=VALID_TASKS, default="make")
    parser.add_argument("--model-name", default="vit_small_patch16_224")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def export_split(model, dataset, args, device):
    # shuffle=False is essential: output row i must correspond to dataset[i].
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    chunks = []
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    amp_enabled = args.amp_dtype != "fp32"
    with torch.inference_mode():
        for batch_index, (images, view_mask, _, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            view_mask = view_mask.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=dtype, enabled=amp_enabled):
                logits = model(images, view_mask=view_mask)
            chunks.append(logits.float().cpu())
            if (batch_index + 1) % 100 == 0:
                print(f"{dataset.split}: {(batch_index + 1) * args.batch_size}/{len(dataset)}", flush=True)
    logits = torch.cat(chunks).numpy()
    if logits.shape != (len(dataset), len(dataset.classes)):
        raise RuntimeError(f"导出形状错误: {logits.shape}")
    if not np.isfinite(logits).all():
        raise RuntimeError(f"{dataset.split} logits 包含 NaN 或 Inf")
    return logits


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("teacher 导出需要 CUDA")
    device = torch.device("cuda", 0)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    datasets = {
        split: BoxCarsMultiView(args.dataset_path, split, args.task, args.num_views, transform)
        for split in ("train", "validation")
    }
    model = EarlyFusionMultiViewViT(
        model_name=args.model_name, num_views=args.num_views,
        num_classes=len(datasets["train"].classes), pretrained=False,
    ).to(device)
    checkpoint = torch.load(args.baseline_checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    train_logits = export_split(model, datasets["train"], args, device)
    val_logits = export_split(model, datasets["validation"], args, device)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        args.output,
        train_logits=train_logits,
        val_logits=val_logits,
        train_fingerprint=dataset_fingerprint(datasets["train"]),
        val_fingerprint=dataset_fingerprint(datasets["validation"]),
    )
    print(f"saved {args.output}: train={train_logits.shape}, val={val_logits.shape}")


if __name__ == "__main__":
    main()
