"""Generate RL scheduling trajectories from the BoxCars116k scene-2 model."""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
PERCEPTION_DIR = PROJECT_DIR / "module_edge_perception"
sys.path.insert(0, str(PERCEPTION_DIR))

from boxcars_dataset import BoxCarsMultiView, VALID_TASKS  # noqa: E402
from boxcars_drift_dataset import BoxCarsDriftWrapper  # noqa: E402
from model import EarlyFusionMultiViewViT  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate measured BoxCars116k trajectories for the RL scheduler."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=PROJECT_DIR / "data" / "BoxCars116k_kaggle" / "BoxCars116k",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PERCEPTION_DIR / "checkpoints" / "boxcars_make_baseline" / "best.pth",
    )
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "real_trajectory_data.csv")
    parser.add_argument("--task", choices=VALID_TASKS, default="make")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--model-name", default="vit_small_patch16_224")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--drift-schedule",
        choices=("none", "light", "mixed", "staged", "highfreq"),
        default="none",
    )
    parser.add_argument("--drift-seed", type=int, default=123)
    parser.add_argument("--score-method", choices=("kl", "confidence"), default="kl")
    parser.add_argument("--w-normalization", choices=("sum", "max", "raw"), default="sum")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def normalized_entropy(prob):
    eps = 1e-12
    entropy = -(prob.clamp_min(eps) * prob.clamp_min(eps).log()).sum(dim=1)
    return entropy / math.log(prob.shape[1])


def view_scores(prob_full, masked_probs, method):
    eps = 1e-12
    scores = []
    if method == "kl":
        log_full = prob_full.clamp_min(eps).log()
        for prob_masked in masked_probs:
            log_masked = prob_masked.clamp_min(eps).log()
            scores.append((prob_full * (log_full - log_masked)).sum(dim=1).clamp_min(0.0))
    else:
        pred_full = prob_full.argmax(dim=1)
        conf_full = prob_full.gather(1, pred_full[:, None]).squeeze(1)
        for prob_masked in masked_probs:
            conf_masked = prob_masked.gather(1, pred_full[:, None]).squeeze(1)
            scores.append((conf_full - conf_masked).clamp_min(0.0))
    return torch.stack(scores, dim=1)


def normalize_w(scores, view_mask, mode):
    eps = 1e-8
    scores = scores.clamp_min(0.0) * view_mask
    if mode == "raw":
        return scores.clamp_min(eps) * view_mask
    denom = scores.max(dim=1, keepdim=True).values if mode == "max" else scores.sum(dim=1, keepdim=True)
    normalized = scores / denom.clamp_min(eps)
    valid_count = view_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    fallback = view_mask / valid_count
    normalized = torch.where((denom <= eps).expand_as(normalized), fallback, normalized)
    return normalized.clamp_min(eps) * view_mask


@torch.no_grad()
def measure_batch(model, images, view_mask, args, device):
    images = images.to(device, non_blocking=True)
    view_mask = view_mask.to(device, non_blocking=True)
    prob_full = torch.softmax(model(images, view_mask=view_mask), dim=1)
    drift = normalized_entropy(prob_full)
    pred = prob_full.argmax(dim=1)
    confidence = prob_full.max(dim=1).values

    masked_probs = []
    for view_index in range(args.num_views):
        masked_view_mask = view_mask.clone()
        masked_view_mask[:, view_index] = 0.0
        masked_probs.append(torch.softmax(model(images, view_mask=masked_view_mask), dim=1))
    scores = view_scores(prob_full, masked_probs, args.score_method)
    return drift.cpu(), normalize_w(scores, view_mask, args.w_normalization).cpu(), pred.cpu(), confidence.cpu()


def main():
    args = parse_args()
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    base_dataset = BoxCarsMultiView(
        str(args.dataset_path), args.split, args.task, args.num_views, transform
    )
    dataset = BoxCarsDriftWrapper(
        base_dataset, args.drift_schedule, args.drift_seed, normalize=True
    )
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"), persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = EarlyFusionMultiViewViT(
        args.model_name, args.num_views, len(base_dataset.classes), pretrained=False
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    rows = []
    for sample_offset, batch in enumerate(loader):
        images, view_mask, labels, _metadata, drift_types, severities, struct_drifts = batch
        drift, weights, pred, confidence = measure_batch(model, images, view_mask, args, device)
        for local_index in range(labels.shape[0]):
            row = {
                "sample_id": len(rows), "label": int(labels[local_index]),
                "pred": int(pred[local_index]), "confidence": float(confidence[local_index]),
                "E_drift": float(drift[local_index]), "drift_type": str(drift_types[local_index]),
                "severity": float(severities[local_index]), "struct_drift": float(struct_drifts[local_index]),
            }
            for view_index in range(args.num_views):
                row[f"w_{view_index + 1}"] = float(weights[local_index, view_index])
                row[f"view_mask_{view_index + 1}"] = int(view_mask[local_index, view_index])
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "label", "pred", "confidence", "E_drift", "drift_type", "severity", "struct_drift"]
    fields += [f"w_{index + 1}" for index in range(args.num_views)]
    fields += [f"view_mask_{index + 1}" for index in range(args.num_views)]
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    weights = np.array([[row[f"w_{index + 1}"] for index in range(args.num_views)] for row in rows])
    drift_values = np.array([row["E_drift"] for row in rows])
    print(f"Saved {len(rows)} rows to: {args.output}")
    print(f"drift_schedule={args.drift_schedule}, E_drift mean={drift_values.mean():.4f}, std={drift_values.std():.4f}")
    print(f"w_t mean per view={np.round(weights.mean(axis=0), 4).tolist()}")


if __name__ == "__main__":
    main()
