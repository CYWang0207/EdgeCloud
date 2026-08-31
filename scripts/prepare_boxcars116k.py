#!/usr/bin/env python3
"""Validate and stage an unpacked BoxCars116k archive without absolute paths."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REQUIRED = ("images", "json_data/dataset.json", "json_data/classification_splits.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Unpacked BoxCars116k directory, or a parent containing BoxCars116k/.")
    parser.add_argument("--destination", type=Path,
                        default=Path("data/BoxCars116k_kaggle/BoxCars116k"))
    parser.add_argument("--copy", action="store_true", help="Copy after validation; default is validation only.")
    parser.add_argument("--force", action="store_true", help="Allow copying into a non-empty destination.")
    return parser.parse_args()


def resolve_root(source: Path) -> Path:
    candidate = source / "BoxCars116k" if (source / "BoxCars116k").is_dir() else source
    missing = [item for item in REQUIRED if not (candidate / item).exists()]
    if missing:
        raise ValueError(f"Not an unpacked BoxCars116k root; missing: {', '.join(missing)}")
    return candidate


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Source does not exist: {source}")
    root = resolve_root(source)
    image_count = sum(1 for _ in (root / "images").rglob("*.png"))
    print(f"Validated BoxCars116k root with {image_count:,} PNG files (images and masks may both be present).")
    if not args.copy:
        print("Validation only. Re-run with --copy to stage the dataset.")
        return
    destination = args.destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not args.force:
        raise SystemExit("Destination is not empty; use --force only after checking its contents.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root, destination, dirs_exist_ok=args.force)
    print("BoxCars116k staging complete.")


if __name__ == "__main__":
    main()
