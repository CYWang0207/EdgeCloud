"""Evaluate ModelNet40 baseline and conditional adapters on calibrated camera drift."""
import argparse
import json
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from adaptformer import attach_adaptformer, load_adapter_checkpoint, set_adapter_enabled
from dataset import ModelNet40MultiView
from model import EarlyFusionMultiViewViT
from modelnet_camera_drift_dataset import DRIFTS, PairedModelNetDrift


def corruption_spec(value):
    name, separator, raw = value.partition("=")
    if not separator or name not in DRIFTS or not 0 <= float(raw) <= 1:
        raise argparse.ArgumentTypeError("use TYPE=0..1")
    return name, float(raw)


def args_parser():
    parser = argparse.ArgumentParser(description="ModelNet40 conditional Adapter evaluation")
    parser.add_argument("--dataset-path", required=True); parser.add_argument("--baseline-checkpoint", required=True); parser.add_argument("--adapter-checkpoint", required=True); parser.add_argument("--output-json", required=True)
    parser.add_argument("--corruption-spec", type=corruption_spec, action="append", default=[]); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--model-name", default="vit_small_patch16_224"); parser.add_argument("--num-views", type=int, default=4); parser.add_argument("--r", type=int, default=32); parser.add_argument("--condition-dim", type=int, default=128); parser.add_argument("--seed", type=int, default=123); return parser.parse_args()


def load_baseline(model, path):
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload, strict=True)


@torch.no_grad()
def paired_accuracy(model, loader, device, corrupted):
    """Evaluate baseline and conditional adapter on the same loaded batches."""
    baseline_correct = adapter_correct = total = 0
    for clean, corrupt, labels in loader:
        images, labels = (corrupt if corrupted else clean).to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            set_adapter_enabled(model, False)
            baseline_logits = model(images)
            set_adapter_enabled(model, True)
            adapter_logits = model(images)
        baseline_correct += (baseline_logits.argmax(1) == labels).sum().item()
        adapter_correct += (adapter_logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return baseline_correct / max(total, 1), adapter_correct / max(total, 1), total


def main():
    args = args_parser()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    specs = args.corruption_spec or [("illumination", 1.), ("defocus", .2), ("sensor_noise", .4)]
    base = ModelNet40MultiView(args.dataset_path, "test", transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), args.num_views)
    model = EarlyFusionMultiViewViT(args.model_name, args.num_views, len(base.classes), pretrained=False); load_baseline(model, args.baseline_checkpoint); model.attach_drift_conditioner(args.condition_dim); attach_adaptformer(model, args.r, condition_dim=args.condition_dim); model.cuda().eval()
    load_adapter_checkpoint(model, args.adapter_checkpoint)
    cases = [("clean", "normal", 0.)] + [(f"{name}_{severity}", name, severity) for name, severity in specs]
    result = {"dataset": "modelnet40", "split": "test", "camera_drift_generator": "modelnet_camera_drift_dataset", "calibrated_corruptions": [f"{name}={severity}" for name, severity in specs], "baseline": {}, "adapter": {}}
    for name, drift, severity in cases:
        # ``normal`` labels the clean reporting row only; the paired dataset
        # still requires a registered camera-drift name.  Zero illumination
        # leaves the image unchanged, and the clean branch below selects it.
        dataset_drift = "illumination" if name == "clean" else drift
        data = PairedModelNetDrift(base, (dataset_drift,), fixed_drift=dataset_drift, fixed_severity=severity, seed=args.seed)
        loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
        baseline_accuracy, adapter_accuracy, samples = paired_accuracy(model, loader, torch.device("cuda"), corrupted=name != "clean")
        result["baseline"][name] = baseline_accuracy
        result["adapter"][name] = adapter_accuracy
    result["samples"] = samples; result["delta_pp"] = {name: (result["adapter"][name] - result["baseline"][name]) * 100 for name in result["baseline"]}
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle: json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
