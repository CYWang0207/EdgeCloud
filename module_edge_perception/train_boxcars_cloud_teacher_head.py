"""Train and gate a BoxCars head on a frozen cloud visual teacher.

The expensive backbone is used once to cache track-level features.  A small
linear/MLP classifier is then trained on clean plus camera-corrupted training
features and evaluated on independent, fixed validation corruptions.
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from boxcars_camera_drift_dataset import PairedBoxCarsCameraDrift
from boxcars_dataset import BoxCarsMultiView

IM = torch.tensor((.485, .456, .406)).view(1, 1, 3, 1, 1)
IS = torch.tensor((.229, .224, .225)).view(1, 1, 3, 1, 1)
CM = torch.tensor((.48145466, .4578275, .40821073)).view(1, 1, 3, 1, 1)
CS = torch.tensor((.26862954, .26130258, .27577711)).view(1, 1, 3, 1, 1)
DRIFTS = ("illumination", "motion_blur", "sensor_noise")


def args():
    p = argparse.ArgumentParser(description="Frozen InternViT BoxCars head training")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=0, help="0 selects a linear head")
    p.add_argument("--train-drifts", nargs="+", choices=DRIFTS, default=list(DRIFTS),
                   help="camera corruptions included in the task-head training cache")
    p.add_argument("--severity-min", type=float, default=.3)
    p.add_argument("--severity-max", type=float, default=1.)
    p.add_argument("--head-batch-size", type=int, default=512)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--resume-cache", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def base(path, split):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return BoxCarsMultiView(path, split, "make", 4, transform)


def subset(dataset, maximum, seed):
    if not maximum or maximum >= len(dataset):
        return dataset
    return Subset(dataset, random.Random(seed).sample(range(len(dataset)), maximum))


@torch.inference_mode()
def feature(model, images, mask, device):
    pixels = ((images * IS + IM).clamp(0, 1) - CM) / CS
    b, v = pixels.shape[:2]
    output = model(pixels.reshape(b * v, *pixels.shape[2:]).to(device, torch.bfloat16))
    hidden = output[0] if isinstance(output, (tuple, list)) else output.last_hidden_state
    hidden = hidden[:, 0] if hidden.ndim == 3 else hidden
    hidden = hidden.reshape(b, v, -1).float()
    weight = mask.to(device, hidden.dtype).unsqueeze(-1)
    return F.normalize((hidden * weight).sum(1) / weight.sum(1).clamp_min(1), dim=-1)


@torch.inference_mode()
def cache_features(model, loader, device, path, include_clean):
    features, labels, indices, roles, kinds, severities = [], [], [], [], [], []
    started = time.time()
    for step, batch in enumerate(loader):
        clean, corrupt, mask, label = batch[:4]
        if include_clean:
            features.append(feature(model, clean, mask, device).cpu().half())
            labels.append(label.clone())
            if len(batch) > 4:
                indices.append(batch[-1].long().cpu()); roles.extend(["clean"] * len(label))
                kinds.extend(["normal"] * len(label)); severities.append(torch.zeros(len(label)))
        features.append(feature(model, corrupt, mask, device).cpu().half())
        labels.append(label.clone())
        if len(batch) > 4:
            indices.append(batch[-1].long().cpu()); roles.extend(["corrupt"] * len(label))
            kinds.extend(list(batch[5])); severities.append(batch[6].float().cpu())
        if step % 25 == 0:
            print(f"cache {os.path.basename(path)}: batches={step + 1}/{len(loader)} "
                  f"elapsed={time.time() - started:.0f}s", flush=True)
    payload = {"features": torch.cat(features), "labels": torch.cat(labels)}
    if indices:
        payload.update({"indices": torch.cat(indices), "roles": roles,
                        "sample_drift_types": kinds, "sample_severities": torch.cat(severities),
                        "layout": "batch_interleaved_clean_then_corrupt" if include_clean else "sequential"})
    torch.save(payload, path)
    print(f"saved {path}: {len(payload['labels'])} x {payload['features'].shape[1]}", flush=True)
    return payload


class Head(nn.Module):
    def __init__(self, dimension, classes, hidden):
        super().__init__()
        self.network = (nn.Linear(dimension, classes) if not hidden else
                        nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                      nn.Dropout(.1), nn.Linear(hidden, classes)))

    def forward(self, value):
        return self.network(value)


@torch.no_grad()
def evaluate(head, payload, device):
    x, y = payload["features"], payload["labels"]
    correct = 0
    for start in range(0, len(y), 2048):
        logits = head(x[start:start + 2048].to(device).float())
        correct += logits.argmax(1).cpu().eq(y[start:start + 2048]).sum().item()
    return correct / max(1, len(y))


def main():
    a = args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)
    device = torch.device("cuda")
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = property(lambda _self: {})
    from transformers import AutoModel
    model = AutoModel.from_pretrained(a.model_path, dtype=torch.bfloat16,
                                      low_cpu_mem_usage=True, trust_remote_code=True).to(device).eval()

    common = dict(batch_size=a.batch_size, num_workers=a.num_workers, shuffle=False,
                  pin_memory=True, persistent_workers=a.num_workers > 0)
    train_base = base(a.dataset_path, "train")
    train = PairedBoxCarsCameraDrift(
        train_base, tuple(a.train_drifts), severity_min=a.severity_min,
        severity_max=a.severity_max, clean_probability=0.,
        seed=a.seed, normalize=True, return_metadata=True)
    train = subset(train, a.max_train_samples, a.seed)
    cache_paths = {"train": os.path.join(a.output_dir, "train_clean_drift_features.pt")}
    specs = {"clean": None, "illumination_1.0": ("illumination", 1.),
             "motion_blur_0.8": ("motion_blur", .8), "sensor_noise_0.6": ("sensor_noise", .6)}
    val_base = base(a.dataset_path, "validation")

    if a.resume_cache and os.path.isfile(cache_paths["train"]):
        train_cache = torch.load(cache_paths["train"], map_location="cpu", weights_only=True)
    else:
        train_cache = cache_features(model, DataLoader(train, **common), device,
                                     cache_paths["train"], include_clean=True)
    val_cache = {}
    for name, spec in specs.items():
        path = os.path.join(a.output_dir, f"val_{name}_features.pt")
        cache_paths[name] = path
        if a.resume_cache and os.path.isfile(path):
            val_cache[name] = torch.load(path, map_location="cpu", weights_only=True)
            continue
        drift = spec[0] if spec else "illumination"
        ds = PairedBoxCarsCameraDrift(val_base, (drift,), clean_probability=0., seed=123,
                                      fixed_drift=drift, fixed_severity=spec[1] if spec else 0.,
                                      normalize=True, return_metadata=True)
        ds = subset(ds, a.max_val_samples, 123)
        val_cache[name] = cache_features(model, DataLoader(ds, **common), device, path,
                                         include_clean=False)
    del model
    torch.cuda.empty_cache()

    x, y = train_cache["features"], train_cache["labels"]
    loader = DataLoader(TensorDataset(x, y), batch_size=a.head_batch_size, shuffle=True)
    head = Head(x.shape[1], len(train_base.classes), a.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    best, best_state, history = -1., None, []
    for epoch in range(1, a.epochs + 1):
        head.train(); loss_sum = count = 0
        for bx, by in loader:
            bx, by = bx.to(device).float(), by.to(device)
            loss = F.cross_entropy(head(bx), by)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(by); count += len(by)
        head.eval()
        scores = {name: evaluate(head, data, device) for name, data in val_cache.items()}
        robust = sum(scores[name] for name in specs if name != "clean") / 3
        history.append({"epoch": epoch, "loss": loss_sum / count, "accuracy": scores,
                        "mean_drift": robust})
        if robust > best:
            best = robust
            best_state = {key: value.detach().cpu() for key, value in head.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch={epoch} loss={loss_sum/count:.4f} mean_drift={robust:.4%} "
                  + " ".join(f"{k}={v:.4%}" for k, v in scores.items()), flush=True)
    head.load_state_dict(best_state)
    final = {name: evaluate(head, data, device) for name, data in val_cache.items()}
    checkpoint = {"state_dict": best_state, "feature_dim": x.shape[1],
                  "num_classes": len(train_base.classes), "hidden_dim": a.hidden_dim,
                  "classes": train_base.classes, "teacher": "InternViT-6B-224px"}
    torch.save(checkpoint, os.path.join(a.output_dir, "boxcars_head.pt"))
    result = {"teacher": checkpoint["teacher"], "frozen_backbone": True,
              "train_samples": len(y), "train_drifts": a.train_drifts,
              "best_mean_drift": best,
              "accuracy": final, "history": history, "cache_paths": cache_paths,
              "args": vars(a)}
    with open(os.path.join(a.output_dir, "metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"best_mean_drift": best, "accuracy": final}, indent=2), flush=True)


if __name__ == "__main__":
    main()
