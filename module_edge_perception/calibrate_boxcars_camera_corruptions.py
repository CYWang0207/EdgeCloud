"""Scan camera-plausible BoxCars corruptions before training an adapter."""
import argparse, json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from boxcars_dataset import BoxCarsMultiView, VALID_TASKS
from boxcars_camera_drift_dataset import DRIFTS, PairedBoxCarsCameraDrift
from model import EarlyFusionMultiViewViT


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", required=True); p.add_argument("--baseline-checkpoint", required=True)
    p.add_argument("--task", choices=VALID_TASKS, default="make"); p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--severities", type=float, nargs="+", default=(.2, .4, .6, .8, 1.0))
    p.add_argument("--drift-types", nargs="+", choices=DRIFTS, default=DRIFTS)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=123); p.add_argument("--max-batches", type=int, default=0); p.add_argument("--output-json", required=True)
    return p.parse_args()

def load(model, path):
    payload = torch.load(path, map_location="cpu"); model.load_state_dict(payload.get("model", payload.get("state_dict", payload)), strict=True)

@torch.no_grad()
def accuracy(model, loader, device, clean, limit):
    good = total = 0
    for i, batch in enumerate(loader):
        if limit and i >= limit: break
        images, mask, labels = (batch[0] if clean else batch[1]).to(device), batch[2].to(device), batch[3].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(images, view_mask=mask)
        good += (logits.argmax(1) == labels).sum().item(); total += labels.numel()
    return good / max(total, 1), total

def main():
    a = args_parser(); device = torch.device("cuda")
    base = BoxCarsMultiView(a.dataset_path, a.split, a.task, 4, transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]))
    model = EarlyFusionMultiViewViT("vit_small_patch16_224", 4, len(base.classes), pretrained=False); load(model, a.baseline_checkpoint); model.to(device).eval()
    clean_ds = PairedBoxCarsCameraDrift(base, ("illumination",), fixed_drift="normal", fixed_severity=0., seed=a.seed)
    clean_acc, samples = accuracy(model, DataLoader(clean_ds, batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=True), device, True, a.max_batches)
    output = {"dataset":"boxcars116k", "split":a.split, "samples":samples, "clean_accuracy":clean_acc, "target_drop_pp":[8,15], "corruptions":{}}
    for drift in a.drift_types:
        records=[]
        for severity in a.severities:
            ds=PairedBoxCarsCameraDrift(base,(drift,),fixed_drift=drift,fixed_severity=severity,seed=a.seed)
            acc,_=accuracy(model,DataLoader(ds,batch_size=a.batch_size,num_workers=a.num_workers,pin_memory=True),device,False,a.max_batches)
            records.append({"severity":severity,"accuracy":acc,"drop_pp":(clean_acc-acc)*100})
            print(f"{drift} severity={severity:.2f} acc={acc:.4f} drop_pp={(clean_acc-acc)*100:.2f}",flush=True)
        output["corruptions"][drift]=records
    with open(a.output_json,"w",encoding="utf-8") as f: json.dump(output,f,ensure_ascii=False,indent=2)
    print(json.dumps(output,ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
