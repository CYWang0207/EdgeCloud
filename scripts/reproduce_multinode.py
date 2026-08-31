#!/usr/bin/env python3
"""Reproduction entrypoint for BoxCars multi-node arbitration."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=ROOT / "data/BoxCars116k_kaggle/BoxCars116k")
    parser.add_argument("--baseline-checkpoint", type=Path, default=ROOT / "models/boxcars_make_baseline/best.pth")
    parser.add_argument("--adapter-checkpoint", type=Path, default=ROOT / "models/boxcars_cloud_teacher_adapter/cloud_unlabeled/best.pth")
    parser.add_argument("--fusion", choices=("weighted", "bayesian"), default="weighted")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/multi_node/reproduced_weighted.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    command = [sys.executable, str(ROOT / "module_scheduling/multi_node/multi_node_eval.py"),
               "--dataset-path", str(args.dataset_path), "--baseline-checkpoint", str(args.baseline_checkpoint),
               "--adapter-checkpoint", str(args.adapter_checkpoint), "--task", "make", "--split", "test",
               "--fusion", args.fusion, "--output", str(args.output), "--seed", str(args.seed), *extra]
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
