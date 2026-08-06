"""Export multi-layer Qwen3-VL visual features for BoxCars drift training.

The cache contains one fp16 tensor shaped ``[N, V, L, D]``.  For Qwen3-VL,
``L`` consists of the configured deep-stack layers (normally 8/16/24) plus the
final merged visual output.  No text generation or vehicle-brand prediction is
performed: the VLM is used strictly as a frozen environment representation
teacher.
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

from boxcars_dataset import BoxCarsMultiView, VALID_SPLITS
from boxcars_drift_dataset import PairedBoxCarsDriftDataset, SUPPORTED_TRAIN_DRIFTS


def parse_drift_types(value):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(values) - set(SUPPORTED_TRAIN_DRIFTS))
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            f"drift types must come from {SUPPORTED_TRAIN_DRIFTS}; unknown={unknown}"
        )
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Export Qwen3-VL visual conditions")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=VALID_SPLITS, default="train")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--drift-types", type=parse_drift_types,
                        default=SUPPORTED_TRAIN_DRIFTS)
    parser.add_argument("--severity-min", type=float, default=0.35)
    parser.add_argument("--severity-max", type=float, default=1.0)
    parser.add_argument("--clean-probability", type=float, default=0.15)
    parser.add_argument("--independent-view-drifts", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-every", type=int, default=0,
        help="atomically rewrite the partial cache every N samples; 0 saves only at the end",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="continue from a compatible partial cache already present at --output",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def image_token_counts(grid_thw, spatial_merge_size):
    counts = grid_thw.prod(dim=1) // (spatial_merge_size ** 2)
    return [int(value) for value in counts.tolist()]


def pool_per_image(features, counts):
    # Qwen3VLModel.get_image_features already splits ``pooler_output`` per
    # image, while deep-stack tensors remain concatenated in this transformers
    # version. Support both representations explicitly.
    chunks = tuple(features) if isinstance(features, (tuple, list)) else features.split(counts, dim=0)
    if len(chunks) != len(counts):
        raise RuntimeError("visual feature count does not match image_grid_thw")
    return torch.stack([chunk.float().mean(dim=0) for chunk in chunks])


@torch.inference_mode()
def extract_sample(model, processor, images, view_mask):
    pil_images = [
        to_pil_image(image.clamp(0, 1))
        for image, available in zip(images, view_mask)
        if bool(available)
    ]
    if not pil_images:
        raise RuntimeError("sample has no available views")
    inputs = processor(images=pil_images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(model.device)
    grid_thw = inputs["image_grid_thw"].to(model.device)
    visual = model.get_image_features(pixel_values, grid_thw)
    merge = int(model.config.vision_config.spatial_merge_size)
    counts = image_token_counts(grid_thw, merge)
    layers = [
        pool_per_image(features, counts)
        for features in visual.deepstack_features
    ]
    layers.append(pool_per_image(visual.pooler_output, counts))
    available_features = torch.stack(layers, dim=1).half().cpu()

    # BoxCars pads short tracks by repeating their last observation.  Preserve
    # a fixed V dimension while copying the final real feature into padded slots;
    # the original view mask is saved separately for masked downstream pooling.
    output = available_features.new_empty(
        len(view_mask), available_features.shape[1], available_features.shape[2]
    )
    output[:len(available_features)] = available_features
    output[len(available_features):] = available_features[-1]
    return output


def save_cache(path, split, features, masks, metadata):
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    temporary = path + ".tmp"
    torch.save(
        {
            split: torch.stack(features),
            f"{split}_view_mask": torch.stack(masks),
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, path)


def load_resume_cache(path, split, metadata, limit):
    """Load and validate a cache checkpoint before continuing its sample loop."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--resume requested but cache does not exist: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("resume cache must be a dictionary payload")
    features = payload.get(split)
    masks = payload.get(f"{split}_view_mask")
    if not isinstance(features, torch.Tensor) or not isinstance(masks, torch.Tensor):
        raise ValueError(f"resume cache is missing {split!r} features or view masks")
    if features.ndim != 4 or masks.ndim != 2:
        raise ValueError("resume cache has unexpected feature or view-mask shape")
    if features.shape[0] != masks.shape[0] or features.shape[0] > limit:
        raise ValueError("resume cache length is incompatible with this export")
    cached_metadata = payload.get("metadata")
    if not isinstance(cached_metadata, dict):
        raise ValueError("resume cache is missing metadata")
    for key, expected in metadata.items():
        if cached_metadata.get(key) != expected:
            raise ValueError(
                f"resume cache metadata mismatch for {key}: "
                f"cached={cached_metadata.get(key)!r}, expected={expected!r}"
            )
    return list(features.unbind(0)), list(masks.unbind(0))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL feature export requires CUDA")
    if not 0 <= args.clean_probability < 1:
        raise ValueError("clean_probability must be in [0, 1)")
    if args.save_every < 0:
        raise ValueError("save_every must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Non-interactive SSH sessions can invoke the Conda interpreter by absolute
    # path without adding its sibling executables to PATH.  GPTQModel discovers
    # the installed ``ninja`` binary through PATH when compiling Marlin kernels.
    python_bin = os.path.dirname(os.path.abspath(sys.executable))
    os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")

    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("transformers with Qwen3-VL support is required") from exc

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    base = BoxCarsMultiView(
        args.dataset_path, args.split, "make", args.num_views, transform
    )
    dataset = PairedBoxCarsDriftDataset(
        base,
        drift_types=args.drift_types,
        severity_min=args.severity_min,
        severity_max=args.severity_max,
        clean_probability=args.clean_probability,
        independent_view_drifts=args.independent_view_drifts,
        seed=args.seed + (0 if args.split == "train" else 1_000_003),
        normalize=False,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        dtype="auto",
        trust_remote_code=args.trust_remote_code,
    ).eval()

    limit = min(len(dataset), args.max_samples or len(dataset))
    features, masks = [], []
    layer_indexes = list(model.config.vision_config.deepstack_visual_indexes) + [
        "final"
    ]
    metadata = {
        "model_path": os.path.abspath(args.model_path),
        "split": args.split,
        "samples": limit,
        "num_views": args.num_views,
        "layers": layer_indexes,
        "drift_types": list(args.drift_types),
        "severity_range": [args.severity_min, args.severity_max],
        "clean_probability": args.clean_probability,
        "independent_view_drifts": args.independent_view_drifts,
        "seed": args.seed,
    }
    if args.resume:
        features, masks = load_resume_cache(args.output, args.split, metadata, limit)
        print(f"resuming {len(features)}/{limit} samples from {args.output}", flush=True)
    start_index = len(features)
    if start_index == limit:
        print(f"cache already complete: {args.output}", flush=True)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    for index in range(start_index, limit):
        clean, drifted, view_mask = dataset[index][:3]
        del clean
        features.append(extract_sample(model, processor, drifted, view_mask))
        masks.append(view_mask.half())
        completed = index + 1
        if completed % args.log_every == 0 or completed == limit:
            print(
                f"{completed}/{limit} feature_shape={tuple(features[-1].shape)}",
                flush=True,
            )
        if (args.save_every and completed % args.save_every == 0) or completed == limit:
            save_cache(args.output, args.split, features, masks, metadata)
            print(f"checkpointed {completed} samples -> {args.output}", flush=True)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
