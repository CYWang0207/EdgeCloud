#!/usr/bin/env python3
"""Reproduction entrypoint for the four-mode weak-network simulation."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxcars-input", type=Path, default=ROOT / "artifacts/inputs/trajectory_boxcars.csv")
    parser.add_argument("--modelnet-input", type=Path, default=ROOT / "artifacts/inputs/trajectory_modelnet40.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/network/reproduced")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    command = [sys.executable, str(ROOT / "module_scheduling/EdgeCloud_RL/run_network_resilience_tests.py"),
               "--boxcars-input", str(args.boxcars_input), "--modelnet-input", str(args.modelnet_input),
               "--output-dir", str(args.output_dir), "--seeds", *(str(seed) for seed in args.seeds), *extra]
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
