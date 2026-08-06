"""Report identical ModelNet40 clean and corruption accuracy for an adapter."""
import argparse
import json
import sys
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from adaptformer import attach_adaptformer, load_adapter_checkpoint, set_adapter_enabled
from dataset import ModelNet40MultiView
from model import EarlyFusionMultiViewViT

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_modelnet_drift_adapter import PairedModelNetDrift, severity_map

def parse_args():
    p = argparse.ArgumentParser(description="ModelNet40 adapter impact evaluation")
    p.add_argument("--dataset-path", required=True); p.add_argument("--baseline-checkpoint", required=True)
    p.add_argument("--adapter-checkpoint", required=True); p.add_argument("--output-json", required=True)
    p.add_argument("--corruption-specs", type=severity_map,
                   default={"illumination": 1.0, "defocus": .2, "sensor_noise": .4},
                   help="only evaluate calibrated NAME=VALUE corruptions")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4); p.add_argument("--r", type=int, default=32); return p.parse_args()

@torch.no_grad()
def accuracy(model, loader, corrupted):
    correct = total = 0
    for clean, corrupt, labels in loader:
        images = corrupt if corrupted else clean
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item(); total += labels.numel()
    return correct / total

def main():
    a = parse_args()
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    base = ModelNet40MultiView(a.dataset_path, "test", transform, 4)
    model = EarlyFusionMultiViewViT("vit_small_patch16_224", 4, len(base.classes), pretrained=False)
    payload = torch.load(a.baseline_checkpoint, map_location="cpu"); model.load_state_dict(payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload, strict=True)
    attach_adaptformer(model, a.r); model.cuda().eval()
    cases = [("clean", "normal", 0.)] + [(f"{kind}_{severity:g}", kind, severity) for kind, severity in a.corruption_specs.items()]
    result = {"dataset": "modelnet40", "split": "test", "calibrated_corruptions": a.corruption_specs, "baseline": {}, "adapter": {}}
    for name, drift, severity in cases:
        data = PairedModelNetDrift(base, (drift,), severity, severity, 0., 123,
                                   fixed_drift=drift, fixed_severity=severity)
        loader = DataLoader(data, batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=True)
        set_adapter_enabled(model, False); result["baseline"][name] = accuracy(model, loader, corrupted=name != "clean")
        load_adapter_checkpoint(model, a.adapter_checkpoint); set_adapter_enabled(model, True); result["adapter"][name] = accuracy(model, loader, corrupted=name != "clean")
    result["delta"] = {key: result["adapter"][key] - result["baseline"][key] for key in result["baseline"]}
    with open(a.output_json, "w") as f: json.dump(result, f, ensure_ascii=False, indent=2); f.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
