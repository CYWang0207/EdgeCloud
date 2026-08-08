"""Evaluate baseline and one or more drift adapters on identical corruptions."""

import argparse
import json

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from adaptformer import attach_adaptformer, load_adapter_checkpoint, set_adapter_enabled
from boxcars_dataset import BoxCarsMultiView, VALID_TASKS
from boxcars_camera_drift_dataset import DRIFTS, PairedBoxCarsCameraDrift
from model import EarlyFusionMultiViewViT


def adapter_spec(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("adapter must be NAME=CHECKPOINT")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("adapter must be NAME=CHECKPOINT")
    return name, path


def corruption_spec(value):
    name, separator, raw = value.partition("=")
    if not separator or name not in DRIFTS or not 0.0 <= float(raw) <= 1.0:
        raise argparse.ArgumentTypeError("corruption spec must be CAMERA_DRIFT=0..1")
    return name, float(raw)


def parse_args():
    parser = argparse.ArgumentParser(description="BoxCars drift adapter comparison")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--adapter", action="append", type=adapter_spec, default=[])
    parser.add_argument("--task", choices=VALID_TASKS, default="make")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--model-name", default="vit_small_patch16_224")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--corruption-spec", type=corruption_spec, action="append",
                        default=[], help="repeatable; defaults to calibrated BoxCars mixture")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def load_baseline(model, path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        payload = payload.get("model", payload.get("state_dict", payload))
    model.load_state_dict(payload, strict=True)


@torch.no_grad()
def accuracy(model, loader, device, amp_dtype, use_clean=False, conditioned=True):
    correct = count = 0
    loss_sum = 0.0
    for batch_index, batch in enumerate(loader):
        if loader.dataset.max_batches and batch_index >= loader.dataset.max_batches:
            break
        images = batch[0] if use_clean else batch[1]
        view_mask, labels = batch[2], batch[3]
        images = images.to(device, non_blocking=True)
        view_mask = view_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(
                images, view_mask=view_mask, apply_quality_gate=conditioned
            )
            loss_sum += torch.nn.functional.cross_entropy(
                logits, labels, reduction="sum"
            ).item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.numel()
    return {"loss": loss_sum / max(count, 1), "accuracy": correct / max(count, 1), "samples": count}


def make_loader(args, base, drift_type, severity):
    dataset = PairedBoxCarsCameraDrift(
        base,
        drift_types=(drift_type,),
        clean_probability=0.0,
        seed=args.seed,
        fixed_drift=drift_type,
        fixed_severity=severity,
    )
    # Keep the limit next to the dataset so accuracy() has a compact signature.
    dataset.max_batches = args.max_batches
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("drift adapter evaluation requires CUDA")
    device = torch.device("cuda")
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
    ])
    base = BoxCarsMultiView(
        args.dataset_path, args.split, args.task, args.num_views, transform
    )
    condition_dims = []
    for _name, path in args.adapter:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and payload.get("condition_dim"):
            condition_dims.append(int(payload["condition_dim"]))
    if condition_dims and len(set(condition_dims)) != 1:
        raise ValueError("all conditioned adapters must use the same condition_dim")
    condition_dim = condition_dims[0] if condition_dims else 0
    model = EarlyFusionMultiViewViT(
        args.model_name, args.num_views, len(base.classes), pretrained=False
    )
    load_baseline(model, args.baseline_checkpoint)
    if condition_dim:
        model.attach_drift_conditioner(condition_dim=condition_dim)
    attach_adaptformer(model, r=args.r, condition_dim=condition_dim)
    model.to(device).eval()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    specs = args.corruption_spec or [
        ("illumination", 1.0), ("motion_blur", .8), ("sensor_noise", .6),
    ]
    loaders = {f"{drift}_{severity}": make_loader(args, base, drift, severity)
               for drift, severity in specs}
    results = {
        "split": args.split,
        "camera_drift_generator": "boxcars_camera_drift_dataset",
        "calibrated_corruptions": [f"{drift}={severity}" for drift, severity in specs],
        "baseline": {},
        "adapters": {},
    }
    set_adapter_enabled(model, False)
    first_loader = next(iter(loaders.values()))
    results["baseline"]["clean"] = accuracy(
        model, first_loader, device, amp_dtype, use_clean=True, conditioned=False
    )
    for drift, loader in loaders.items():
        results["baseline"][drift] = accuracy(
            model, loader, device, amp_dtype, conditioned=False
        )
    for name, path in args.adapter:
        load_adapter_checkpoint(model, path, device="cpu")
        set_adapter_enabled(model, True)
        expert = {
            "clean": accuracy(model, first_loader, device, amp_dtype, use_clean=True)
        }
        for drift, loader in loaders.items():
            expert[drift] = accuracy(model, loader, device, amp_dtype)
        results["adapters"][name] = expert
    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
