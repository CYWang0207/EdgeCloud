"""Calibrate ModelNet40 camera-corruption severity before adapter training.

For every corruption family this searches a common [0, 1] severity grid on the
frozen baseline and selects the point closest to a requested accuracy drop.
The resulting JSON is the training/evaluation contract, not an after-the-fact
visualisation.
"""
import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from dataset import ModelNet40MultiView
from model import EarlyFusionMultiViewViT
from train_modelnet_drift_adapter import DRIFTS, NORM, apply_camera_corruption


class CorruptedModelNet(Dataset):
    def __init__(self, base, kind, severity, seed):
        self.base, self.kind, self.severity, self.seed = base, kind, severity, seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        views, label = self.base[index]
        views = apply_camera_corruption(views, self.kind, self.severity, self.seed + index * 1009)
        return torch.stack([NORM(image) for image in views]), label


def parse_args():
    parser = argparse.ArgumentParser(description="ModelNet40 camera-corruption baseline calibration")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--severities", type=float, nargs="+", default=(.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.))
    parser.add_argument("--target-drop-pp", type=float, default=10.)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def load_baseline(model, path):
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def accuracy(model, loader):
    correct = total = 0
    for views, labels in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(views.cuda(non_blocking=True))
        labels = labels.cuda(non_blocking=True)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return correct / total


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.severities or any(not 0 <= value <= 1 for value in args.severities):
        raise ValueError("severities must be non-empty values in [0, 1]")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    base = ModelNet40MultiView(args.dataset_path, "test", transform, 4)
    model = EarlyFusionMultiViewViT("vit_small_patch16_224", 4, len(base.classes), pretrained=False)
    load_baseline(model, args.baseline_checkpoint)
    model.cuda().eval()
    clean = CorruptedModelNet(base, "normal", 0., args.seed)
    clean_accuracy = accuracy(model, DataLoader(clean, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True))
    target = args.target_drop_pp / 100.
    results = {"dataset": "modelnet40", "split": "test", "samples": len(base), "clean_accuracy": clean_accuracy, "target_drop_pp": args.target_drop_pp, "grid": list(args.severities), "corruptions": {}}
    for kind in DRIFTS:
        matrix = []
        for severity in args.severities:
            dataset = CorruptedModelNet(base, kind, severity, args.seed)
            score = accuracy(model, DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True))
            entry = {"severity": severity, "accuracy": score, "drop_pp": 100 * (clean_accuracy - score)}
            matrix.append(entry)
            print(f"{kind} severity={severity:.2f} accuracy={score:.4f} drop_pp={entry['drop_pp']:.2f}", flush=True)
        chosen = min(matrix, key=lambda item: abs(item["drop_pp"] - args.target_drop_pp))
        results["corruptions"][kind] = {"grid_results": matrix, "selected": chosen, "target_met_within_2pp": abs(chosen["drop_pp"] - args.target_drop_pp) <= 2.}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
