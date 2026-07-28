import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent


def find_mv_vit_dir():
    search_roots = [SCRIPT_DIR, *SCRIPT_DIR.parents]
    candidates = []
    for root in search_roots:
        candidates.append(root)
        candidates.append(root / "MV-VIT")

    for candidate in candidates:
        if (candidate / "dataset.py").exists() and (candidate / "model.py").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate MV-VIT root. Expected to find dataset.py and model.py "
        "in the script's parent directories or in an MV-VIT subdirectory."
    )


MV_VIT_DIR = find_mv_vit_dir()
sys.path.insert(0, str(MV_VIT_DIR))

from dataset import ModelNet40MultiView  # noqa: E402
from model import EarlyFusionMultiViewViT  # noqa: E402
from drift_dataset import DeterministicDriftWrapper  # noqa: E402


def parse_args():
    default_dataset = MV_VIT_DIR / "data" / "modelnet40v2png_ori4"
    default_checkpoint = MV_VIT_DIR / "checkpoints" / "mv_vit_base_epoch_30.pth"

    parser = argparse.ArgumentParser(
        description="Generate real test-set trajectory data with E_drift and per-view w_t."
    )
    parser.add_argument("--dataset-path", type=Path, default=default_dataset)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--num-classes", type=int, default=40)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--drift-schedule",
        type=str,
        default="none",
        choices=["none", "light", "mixed", "staged", "highfreq"],
        help="Deterministic test-time drift schedule applied before normalization.",
    )
    parser.add_argument("--drift-seed", type=int, default=123)
    parser.add_argument(
        "--score-method",
        type=str,
        default="kl",
        choices=["kl", "confidence"],
        help="Per-view information score: KL(full || masked) or confidence drop.",
    )
    parser.add_argument(
        "--w-normalization",
        type=str,
        default="sum",
        choices=["sum", "max", "raw"],
        help="How to normalize per-view scores into w_t.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    if args.output is None:
        if args.drift_schedule == "none":
            args.output = SCRIPT_DIR / "real_trajectory_data.csv"
        else:
            args.output = SCRIPT_DIR / f"real_trajectory_{args.drift_schedule}.csv"
    return args


def resolve_checkpoint(path):
    if path.exists():
        return path

    candidates = [
        MV_VIT_DIR / "checkpoints" / "mv_vit_base_epoch_30.pth",
        MV_VIT_DIR / "checkpoints" / "mv_vit_epoch_30.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            print(f"Default checkpoint not found, using: {candidate}")
            return candidate

    raise FileNotFoundError(
        "No checkpoint found. Pass --checkpoint with the trained MV-VIT .pth file."
    )


def build_loader(args):
    if args.drift_schedule == "none":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    dataset = ModelNet40MultiView(
        root_dir=str(args.dataset_path),
        split=args.split,
        transform=transform,
        num_views=args.num_views,
    )

    if args.drift_schedule != "none":
        dataset = DeterministicDriftWrapper(
            dataset,
            schedule=args.drift_schedule,
            seed=args.drift_seed,
            normalize=True,
        )

    if args.max_samples is not None:
        max_samples = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, range(max_samples))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and args.device.startswith("cuda"),
    )
    return loader


def load_model(args, checkpoint_path, device):
    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=args.num_views,
        num_classes=args.num_classes,
        pretrained=False,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"权重兼容加载: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


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
            score = (prob_full * (log_full - log_masked)).sum(dim=1)
            scores.append(score.clamp_min(0.0))
    else:
        pred_full = prob_full.argmax(dim=1)
        conf_full = prob_full.gather(1, pred_full[:, None]).squeeze(1)
        for prob_masked in masked_probs:
            conf_masked = prob_masked.gather(1, pred_full[:, None]).squeeze(1)
            scores.append((conf_full - conf_masked).clamp_min(0.0))

    return torch.stack(scores, dim=1)


def normalize_w(scores, mode):
    eps = 1e-8
    if mode == "raw":
        return scores.clamp_min(eps)

    if mode == "max":
        denom = scores.max(dim=1, keepdim=True).values
        normalized = scores / denom.clamp_min(eps)
    else:
        denom = scores.sum(dim=1, keepdim=True)
        normalized = scores / denom.clamp_min(eps)

    fallback = torch.full_like(normalized, 1.0 / normalized.shape[1])
    bad_rows = denom.squeeze(1) <= eps
    normalized[bad_rows] = fallback[bad_rows]
    return normalized.clamp_min(eps)


@torch.no_grad()
def measure_batch(model, images, args, device):
    images = images.to(device, non_blocking=True)

    logits_full = model(images)
    prob_full = torch.softmax(logits_full, dim=1)
    drift = normalized_entropy(prob_full)
    pred = prob_full.argmax(dim=1)
    confidence = prob_full.max(dim=1).values

    masked_probs = []
    for view_idx in range(args.num_views):
        masked_images = images.clone()
        masked_images[:, view_idx] = 0.0
        logits_masked = model(masked_images)
        masked_probs.append(torch.softmax(logits_masked, dim=1))

    scores = view_scores(prob_full, masked_probs, args.score_method)
    w = normalize_w(scores, args.w_normalization)
    return drift.cpu(), w.cpu(), pred.cpu(), confidence.cpu()


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    loader = build_loader(args)
    model = load_model(args, checkpoint_path, device)
    print(f"Drift schedule: {args.drift_schedule}")

    rows = []
    sample_offset = 0
    for batch in tqdm(loader, desc="Measuring real trajectory"):
        if len(batch) == 5:
            images, labels, drift_types, severities, struct_drifts = batch
        elif len(batch) == 4:
            images, labels, drift_types, severities = batch
            struct_drifts = torch.zeros(labels.shape[0])
        else:
            images, labels = batch
            drift_types = ["normal"] * labels.shape[0]
            severities = torch.zeros(labels.shape[0])
            struct_drifts = torch.zeros(labels.shape[0])

        drift, w, pred, confidence = measure_batch(model, images, args, device)
        labels = labels.cpu()
        severities = severities.cpu()
        struct_drifts = struct_drifts.cpu()

        batch_size = labels.shape[0]
        for local_idx in range(batch_size):
            row = {
                "sample_id": sample_offset + local_idx,
                "label": int(labels[local_idx].item()),
                "pred": int(pred[local_idx].item()),
                "confidence": float(confidence[local_idx].item()),
                "E_drift": float(drift[local_idx].item()),
                "drift_type": str(drift_types[local_idx]),
                "severity": float(severities[local_idx].item()),
                "struct_drift": float(struct_drifts[local_idx].item()),
            }
            for view_idx in range(args.num_views):
                row[f"w_{view_idx + 1}"] = float(w[local_idx, view_idx].item())
            rows.append(row)
        sample_offset += batch_size

    fieldnames = [
        "sample_id",
        "label",
        "pred",
        "confidence",
        "E_drift",
        "drift_type",
        "severity",
        "struct_drift",
    ]
    fieldnames.extend([f"w_{i + 1}" for i in range(args.num_views)])

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to: {args.output}")
    if rows:
        w_cols = [f"w_{i + 1}" for i in range(args.num_views)]
        drift_values = np.array([row["E_drift"] for row in rows])
        w_values = np.array([[row[col] for col in w_cols] for row in rows])
        print(f"E_drift mean={drift_values.mean():.4f}, std={drift_values.std():.4f}")
        print(f"w_t mean per view={np.round(w_values.mean(axis=0), 4).tolist()}")


if __name__ == "__main__":
    main()
