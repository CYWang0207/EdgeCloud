"""Export Qwen3-VL visual conditions from calibrated corrupted ModelNet40 views."""
import argparse
import json
import os
import random

import torch
import torchvision.transforms as transforms

from dataset import ModelNet40MultiView
from export_boxcars_vlm_hidden_states import extract_sample, load_resume_cache, load_visual_encoder, save_cache
from modelnet_camera_drift_dataset import DRIFTS, PairedModelNetDrift


def types(value):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or set(values) - set(DRIFTS): raise argparse.ArgumentTypeError(f"types must come from {DRIFTS}")
    return values


def fixed(value):
    result = {}
    for item in value.split(","):
        name, separator, raw = item.partition("=")
        if not separator or name not in DRIFTS or not 0 <= float(raw) <= 1: raise argparse.ArgumentTypeError("use TYPE=0..1")
        result[name] = float(raw)
    return result


def args_parser():
    parser = argparse.ArgumentParser(description="Export ModelNet40 degraded-view Qwen conditions")
    parser.add_argument("--dataset-path", required=True); parser.add_argument("--model-path", required=True); parser.add_argument("--output", required=True); parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--num-views", type=int, default=4); parser.add_argument("--drift-types", type=types, default=("illumination", "defocus", "sensor_noise")); parser.add_argument("--drift-weights", type=float, nargs="+", default=(.30, .30, .40)); parser.add_argument("--fixed-severities", type=fixed, default={"illumination": 1., "defocus": .2, "sensor_noise": .4}); parser.add_argument("--clean-probability", type=float, default=.2); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--max-samples", type=int, default=0); parser.add_argument("--log-every", type=int, default=20); parser.add_argument("--save-every", type=int, default=100); parser.add_argument("--resume", action="store_true"); return parser.parse_args()


def main():
    args = args_parser()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    if len(args.drift_types) != len(args.drift_weights): raise ValueError("drift-weights length must match drift-types")
    random.seed(args.seed); torch.manual_seed(args.seed)
    base = ModelNet40MultiView(args.dataset_path, args.split, transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), args.num_views)
    dataset = PairedModelNetDrift(base, args.drift_types, clean_probability=args.clean_probability, seed=args.seed + (0 if args.split == "train" else 1_000_003), drift_weights=args.drift_weights, fixed_severities=args.fixed_severities, normalize=False)
    limit = min(len(dataset), args.max_samples or len(dataset)); metadata = {"model_path": os.path.abspath(args.model_path), "split": args.split, "samples": limit, "num_views": args.num_views, "layers": [8, 16, 24, "final"], "drift_types": list(args.drift_types), "drift_weights": list(args.drift_weights), "fixed_severities": args.fixed_severities, "drift_generator": "modelnet_camera_drift_dataset", "clean_probability": args.clean_probability, "seed": args.seed}
    features, masks = load_resume_cache(args.output, args.split, metadata, limit) if args.resume else ([], [])
    model = load_visual_encoder(args.model_path)
    try:
        from transformers import AutoProcessor
    except ImportError as exc: raise RuntimeError("transformers Qwen3-VL support is required") from exc
    processor = AutoProcessor.from_pretrained(args.model_path)
    for index in range(len(features), limit):
        drifted = dataset[index][1]; mask = torch.ones(args.num_views, dtype=torch.float32)
        features.append(extract_sample(model, processor, drifted, mask)); masks.append(mask.half()); completed = index + 1
        if completed % args.log_every == 0 or completed == limit: print(f"{completed}/{limit} feature_shape={tuple(features[-1].shape)}", flush=True)
        if (args.save_every and completed % args.save_every == 0) or completed == limit: save_cache(args.output, args.split, features, masks, metadata); print(f"checkpointed {completed} samples -> {args.output}", flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
