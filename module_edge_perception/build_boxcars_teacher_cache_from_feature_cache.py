"""Build formal trained-head supervision from an existing InternViT cache.

This avoids rerunning the 6B backbone and preserves the stronger experimental
design used during feature extraction: train-split corruptions have random
severity in a configured range, while fixed severities are reserved for the
independent validation benchmark.
"""
import argparse
import os

import torch
import torch.nn as nn

from retrain_boxcars_cloud_teacher_head_from_cache import (
    recover_drift_metadata, unpack_batch_interleaved_cache,
)


class Head(nn.Module):
    def __init__(self, dimension, classes, hidden):
        super().__init__()
        self.network = (nn.Linear(dimension, classes) if not hidden else
                        nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                      nn.Dropout(.1), nn.Linear(hidden, classes)))

    def forward(self, value): return self.network(value)


def parse_args():
    p = argparse.ArgumentParser(description="Convert cached InternViT features to formal supervision")
    p.add_argument("--feature-cache", required=True)
    p.add_argument("--head-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--extraction-batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drift-types", nargs="+",
                   default=["illumination", "motion_blur", "sensor_noise"])
    p.add_argument("--severity-min", type=float, default=.3)
    p.add_argument("--severity-max", type=float, default=1.)
    p.add_argument("--teacher-name", default="InternViT-6B-224px")
    return p.parse_args()


def main():
    a = parse_args()
    cached = torch.load(a.feature_cache, map_location="cpu", weights_only=True)
    tracks = len(cached["labels"]) // 2
    _, features, clean_labels, drift_labels = unpack_batch_interleaved_cache(
        cached, tracks, a.extraction_batch_size)
    if not torch.equal(clean_labels, drift_labels):
        raise RuntimeError("feature cache labels are not aligned")
    head_payload = torch.load(a.head_checkpoint, map_location="cpu", weights_only=True)
    required = ("state_dict", "feature_dim", "num_classes", "hidden_dim")
    if any(key not in head_payload for key in required):
        raise ValueError(f"head checkpoint must contain {required}")
    if int(head_payload["feature_dim"]) != features.shape[1]:
        raise ValueError("head/cache feature dimension mismatch")
    head = Head(features.shape[1], int(head_payload["num_classes"]),
                int(head_payload["hidden_dim"]))
    head.load_state_dict(head_payload["state_dict"], strict=True); head.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, tracks, 2048):
            logits.append(head(features[start:start + 2048].float()).half())
    type_ids, severities = recover_drift_metadata(
        tracks, a.seed, a.drift_types, a.severity_min, a.severity_max)
    kinds = [a.drift_types[index] for index in type_ids.tolist()]
    indices = torch.arange(tracks)
    output = {
        "format_version": 2,
        "teacher": a.teacher_name,
        "feature_dim": int(features.shape[1]),
        "drift_types": list(a.drift_types),
        "severity_mode": "random_range",
        "severity_min": float(a.severity_min),
        "severity_max": float(a.severity_max),
        "logit_head": "trained_boxcars_head",
        "head_checkpoint": os.path.abspath(a.head_checkpoint),
        "source_feature_cache": os.path.abspath(a.feature_cache),
        "train": {"features": features, "logits": torch.cat(logits),
                  "labels": clean_labels, "indices": indices,
                  "sample_drift_types": kinds, "sample_severities": severities,
                  "seed": a.seed},
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    torch.save(output, a.output)
    print(f"saved {a.output}: tracks={tracks} dim={features.shape[1]} "
          f"severity=[{a.severity_min},{a.severity_max}]", flush=True)


if __name__ == "__main__": main()
