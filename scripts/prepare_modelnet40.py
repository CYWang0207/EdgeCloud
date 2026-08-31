#!/usr/bin/env python3
"""Validate and stage a ModelNet40 four-view export without hard-coded paths."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Unpacked ModelNet40 four-view root (contains class directories).")
    parser.add_argument("--destination", type=Path, default=Path("data/modelnet40v2png_ori4"))
    parser.add_argument("--copy", action="store_true", help="Copy files after validation; default is validation only.")
    parser.add_argument("--force", action="store_true", help="Allow copying into a non-empty destination.")
    return parser.parse_args()


def valid_root(root: Path) -> list[Path]:
    classes = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    if len(classes) != 40:
        raise ValueError(f"Expected 40 ModelNet40 class directories, found {len(classes)}")
    png_count = sum(1 for path in root.rglob("*.png"))
    if png_count == 0:
        raise ValueError(f"{root} has no PNG renders")
    return classes


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Source does not exist: {source}")
    classes = valid_root(source)
    print(f"Validated {len(classes)} class directories and {sum(1 for _ in source.rglob('*.png'))} PNG renders.")
    print(f"Target: {destination}")
    if not args.copy:
        print("Validation only. Re-run with --copy to stage the dataset.")
        return
    if destination.exists() and any(destination.iterdir()) and not args.force:
        raise SystemExit("Destination is not empty; use --force only after checking its contents.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=args.force)
    print("ModelNet40 staging complete.")


if __name__ == "__main__":
    main()
