"""Evaluate a BoxCars116k MV-ViT checkpoint on an official split."""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from boxcars_dataset import BoxCarsMultiView, VALID_TASKS
from model import EarlyFusionMultiViewViT


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BoxCars116k MV-ViT")
    parser.add_argument(
        "--dataset-path",
        default="data/BoxCars116k_kaggle/BoxCars116k",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=VALID_TASKS, default="make")
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    parser.add_argument("--model-name", default="vit_small_patch16_224")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="per-GPU batch size")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def main():
    args = parse_args()
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank, rank, world_size = 0, 0, 1
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    dataset = BoxCarsMultiView(
        args.dataset_path, args.split, args.task, args.num_views, transform
    )
    sampler = DistributedSampler(
        dataset, world_size, rank, shuffle=False
    ) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_classes = checkpoint.get("classes") if isinstance(checkpoint, dict) else None
    if checkpoint_classes is not None and checkpoint_classes != dataset.classes:
        raise ValueError("checkpoint classes do not match the requested dataset task")
    state_dict = checkpoint.get("model", checkpoint)
    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=args.num_views,
        num_classes=len(dataset.classes),
        pretrained=False,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    with torch.no_grad():
        for images, view_mask, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            view_mask = view_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(images, view_mask=view_mask)
                loss = F.cross_entropy(logits, labels, reduction="sum")
            top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
            totals[0] += loss
            totals[1] += (logits.argmax(dim=1) == labels).sum()
            totals[2] += (top5 == labels[:, None]).any(dim=1).sum()
            totals[3] += labels.numel()

    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if rank == 0:
        loss_sum, top1, top5, count = totals.tolist()
        print(f"split={args.split}")
        print(f"samples={int(count)}")
        print(f"loss={loss_sum / count:.6f}")
        print(f"top1={top1 / count:.6f} ({100 * top1 / count:.2f}%)")
        print(f"top5={top5 / count:.6f} ({100 * top5 / count:.2f}%)")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
