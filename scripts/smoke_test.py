#!/usr/bin/env python3
"""Run a data/samples four-view image batch through the EdgeCloud model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module_edge_perception.model import EarlyFusionMultiViewViT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path("data/samples/modelnet40"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_track(root: Path) -> list[Path]:
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        images = sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg"))
        if len(images) >= 4:
            return images[:4]
    raise FileNotFoundError(f"No four-view track found under {root}")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is unavailable")
    torch.manual_seed(args.seed)
    paths = choose_track(args.samples)
    transform = v2.Compose([v2.Resize((224, 224)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    views = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths]).unsqueeze(0)
    device = torch.device(args.device)
    model = EarlyFusionMultiViewViT("vit_small_patch16_224", 4, 40, pretrained=False).to(device).eval()
    with torch.no_grad():
        logits = model(views.to(device), view_mask=torch.ones(1, 4, device=device))
    assert logits.shape == (1, 40), f"Unexpected output shape: {tuple(logits.shape)}"
    print(f"SMOKE PASS: {'; '.join(str(path) for path in paths)}")
    print(f"output_shape={tuple(logits.shape)} predicted_class={int(logits.argmax(dim=1).item())}")


if __name__ == "__main__":
    main()
