#!/usr/bin/env python3
"""Reproduction entrypoint for the ModelNet40 cloud-teacher refresh."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=ROOT / "data/modelnet40v2png_ori4")
    parser.add_argument("--baseline-checkpoint", type=Path, default=ROOT / "models/mv_vit_token_epoch_30.pth")
    parser.add_argument("--teacher-model-path", type=Path, default=ROOT / "models/InternViT-6B-224px")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/modelnet40/reproduced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    command = [sys.executable, str(ROOT / "module_edge_perception/modelnet_cloud_teacher_refresh.py"),
               "--dataset-path", str(args.dataset_path), "--baseline-checkpoint", str(args.baseline_checkpoint),
               "--teacher-model-path", str(args.teacher_model_path), "--output-dir", str(args.output_dir),
               "--seed", str(args.seed), *extra]
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
