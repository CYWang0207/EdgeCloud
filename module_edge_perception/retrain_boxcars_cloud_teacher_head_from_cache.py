"""Cheap task-head ablations on an existing InternViT feature cache.

No cloud-backbone forward is performed.  The cache written by
``train_boxcars_cloud_teacher_head.py`` stores each loader batch as
``clean_batch, corrupt_batch``.  This script reconstructs track-aligned clean
and corrupt rows, deterministically recovers the corruption type, and compares
general, noise-weighted, and noise-specialist heads.
"""
import argparse
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from boxcars_camera_drift_dataset import DRIFTS


def parse_args():
    p = argparse.ArgumentParser(description="Retrain cloud task heads without rerunning InternViT")
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--extraction-batch-size", type=int, default=4)
    p.add_argument("--train-size", type=int, default=0,
                   help="number of original tracks; 0 infers half the cached rows")
    p.add_argument("--cache-seed", type=int, default=42)
    p.add_argument("--cache-drift-types", nargs="+", choices=DRIFTS,
                   default=["illumination", "motion_blur", "sensor_noise"])
    p.add_argument("--noise-repeat", type=int, default=3)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--calibration-modulus", type=int, default=10,
                   help="every Nth track is held out for epoch selection")
    p.add_argument("--min-clean", type=float, default=.928,
                   help="hard gate for automatic head selection")
    p.add_argument("--min-noise", type=float, default=.795,
                   help="hard gate for automatic head selection")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class Head(nn.Module):
    def __init__(self, dimension, classes, hidden=0):
        super().__init__()
        self.network = (nn.Linear(dimension, classes) if not hidden else
                        nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                      nn.Dropout(.1), nn.Linear(hidden, classes)))

    def forward(self, value):
        return self.network(value)


def unpack_batch_interleaved_cache(payload, tracks, batch_size):
    """Undo [clean batch, corrupt batch, ...] concatenation exactly."""
    features, labels = payload["features"], payload["labels"]
    if len(features) != 2 * tracks or len(labels) != 2 * tracks:
        raise ValueError(f"expected {2 * tracks} cached rows, got {len(features)}")
    clean_x, drift_x, clean_y, drift_y = [], [], [], []
    row = track = 0
    while track < tracks:
        count = min(batch_size, tracks - track)
        clean_x.append(features[row:row + count]); clean_y.append(labels[row:row + count])
        row += count
        drift_x.append(features[row:row + count]); drift_y.append(labels[row:row + count])
        row += count; track += count
    result = tuple(torch.cat(parts) for parts in (clean_x, drift_x, clean_y, drift_y))
    if not torch.equal(result[2], result[3]):
        raise RuntimeError("clean/corrupt cache labels are not track-aligned")
    return result


def recover_drift_ids(tracks, seed, drift_types):
    """Mirror PairedBoxCarsCameraDrift's seeded type sampling without image I/O."""
    return recover_drift_metadata(tracks, seed, drift_types, .3, 1.)[0]


def recover_drift_metadata(tracks, seed, drift_types, severity_min, severity_max):
    """Recover type IDs and random severities from the paired-dataset contract."""
    result = torch.empty(tracks, dtype=torch.long)
    severities = torch.empty(tracks, dtype=torch.float32)
    weights = torch.ones(len(drift_types))
    for index in range(tracks):
        generator = torch.Generator().manual_seed(seed + index * 104729)
        torch.rand((), generator=generator)  # clean_probability draw, also consumed when it is zero
        result[index] = torch.multinomial(weights, 1, generator=generator).item()
        severities[index] = severity_min + (severity_max - severity_min) * torch.rand((), generator=generator).item()
    return result, severities


@torch.no_grad()
def accuracy(head, x, y, device):
    correct = 0
    for start in range(0, len(y), 2048):
        prediction = head(x[start:start + 2048].to(device).float()).argmax(1).cpu()
        correct += prediction.eq(y[start:start + 2048]).sum().item()
    return correct / max(1, len(y))


def make_rows(mode, clean_x, drift_x, labels, drift_ids, noise_id, keep, noise_repeat):
    clean_indices = keep.nonzero().flatten()
    all_drift_indices = keep.nonzero().flatten()
    noise_indices = (keep & drift_ids.eq(noise_id)).nonzero().flatten()
    if mode == "general_linear":
        indices = [clean_indices, all_drift_indices]
        return torch.cat((clean_x[indices[0]], drift_x[indices[1]])), torch.cat((labels[indices[0]], labels[indices[1]]))
    if mode in ("noise_weighted_linear", "noise_weighted_mlp"):
        x = [clean_x[clean_indices], drift_x[all_drift_indices]]
        y = [labels[clean_indices], labels[all_drift_indices]]
        for _ in range(max(0, noise_repeat - 1)):
            x.append(drift_x[noise_indices]); y.append(labels[noise_indices])
        return torch.cat(x), torch.cat(y)
    if mode == "noise_specialist_linear":
        return torch.cat((clean_x[clean_indices], drift_x[noise_indices])), torch.cat((labels[clean_indices], labels[noise_indices]))
    raise ValueError(mode)


def main():
    a = parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)
    payload = torch.load(a.train_cache, map_location="cpu", weights_only=True)
    tracks = a.train_size or len(payload["labels"]) // 2
    clean_x, drift_x, clean_y, _ = unpack_batch_interleaved_cache(
        payload, tracks, a.extraction_batch_size)
    drift_ids = recover_drift_ids(tracks, a.cache_seed, a.cache_drift_types)
    if "sensor_noise" not in a.cache_drift_types:
        raise ValueError("cache drift types do not contain sensor_noise")
    noise_id = a.cache_drift_types.index("sensor_noise")
    calibration = torch.arange(tracks).remainder(a.calibration_modulus).eq(0)
    training = ~calibration
    classes = int(clean_y.max().item()) + 1
    val = {}
    for name in ("clean", "illumination_1.0", "motion_blur_0.8", "sensor_noise_0.6"):
        item = torch.load(os.path.join(a.val_cache_dir, f"val_{name}_features.pt"),
                          map_location="cpu", weights_only=True)
        val[name] = item

    modes = {"general_linear": 0, "noise_weighted_linear": 0,
             "noise_specialist_linear": 0, "noise_weighted_mlp": 512}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for mode, hidden in modes.items():
        x, y = make_rows(mode, clean_x, drift_x, clean_y, drift_ids, noise_id,
                         training, a.noise_repeat)
        loader = DataLoader(TensorDataset(x, y), batch_size=a.batch_size, shuffle=True)
        head = Head(clean_x.shape[1], classes, hidden).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        best_score, best_epoch, best_state = -1., 0, None
        cal_noise = calibration & drift_ids.eq(noise_id)
        for epoch in range(1, a.epochs + 1):
            head.train()
            for bx, by in loader:
                bx, by = bx.to(device).float(), by.to(device)
                loss = F.cross_entropy(head(bx), by)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            head.eval()
            clean_score = accuracy(head, clean_x[calibration], clean_y[calibration], device)
            noise_score = accuracy(head, drift_x[cal_noise], clean_y[cal_noise], device)
            selection = (clean_score + noise_score) / 2
            if selection > best_score:
                best_score, best_epoch = selection, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        head.load_state_dict(best_state); head.eval()
        scores = {name: accuracy(head, item["features"], item["labels"], device)
                  for name, item in val.items()}
        checkpoint = {"state_dict": best_state, "feature_dim": clean_x.shape[1],
                      "num_classes": classes, "hidden_dim": hidden,
                      "teacher": "InternViT-6B-224px", "head_mode": mode,
                      "best_epoch": best_epoch, "cache_drift_types": a.cache_drift_types}
        path = os.path.join(a.output_dir, f"{mode}.pt")
        torch.save(checkpoint, path)
        results[mode] = {"best_epoch": best_epoch, "calibration_score": best_score,
                         "accuracy": scores, "checkpoint": path}
        print(mode, json.dumps(scores), flush=True)
    eligible = {name: item for name, item in results.items()
                if item["accuracy"]["clean"] >= a.min_clean
                and item["accuracy"]["sensor_noise_0.6"] >= a.min_noise}
    selected = None
    if eligible:
        selected = max(eligible, key=lambda name: sum(
            eligible[name]["accuracy"][condition]
            for condition in ("illumination_1.0", "motion_blur_0.8", "sensor_noise_0.6")) / 3)
        chosen = torch.load(results[selected]["checkpoint"], map_location="cpu", weights_only=True)
        torch.save(chosen, os.path.join(a.output_dir, "selected_head.pt"))
    summary = {"args": vars(a), "tracks": tracks,
               "noise_tracks": int(drift_ids.eq(noise_id).sum()),
               "selection_gate": {"min_clean": a.min_clean, "min_noise": a.min_noise},
               "selected": selected, "results": results}
    with open(os.path.join(a.output_dir, "metrics.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    if selected is None:
        raise RuntimeError("no cached-feature head passed the clean/noise gate; extract a full noise cache next")
    print(f"selected={selected} -> {os.path.join(a.output_dir, 'selected_head.pt')}", flush=True)


if __name__ == "__main__":
    main()
