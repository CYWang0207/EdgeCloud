"""Build cloud-teacher supervision for u=1 adapter refresh.

The exported artifact is the only interface from a frozen large visual model to
the edge training job: per-track drifted features and task logits.  Formal
exports must load a trained BoxCars head; the rejected clean-prototype shortcut
is deliberately unsupported here.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from boxcars_camera_drift_dataset import DRIFTS, PairedBoxCarsCameraDrift
from boxcars_dataset import BoxCarsMultiView, VALID_TASKS

IMAGENET_MEAN = torch.tensor((.485, .456, .406)).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor((.229, .224, .225)).view(1, 1, 3, 1, 1)
CLIP_MEAN = torch.tensor((.48145466, .4578275, .40821073)).view(1, 1, 3, 1, 1)
CLIP_STD = torch.tensor((.26862954, .26130258, .27577711)).view(1, 1, 3, 1, 1)


def parse_types(value):
    result = tuple(x.strip() for x in value.split(",") if x.strip())
    if not result or set(result) - set(DRIFTS): raise argparse.ArgumentTypeError(f"choose {DRIFTS}")
    return result


def parse_fixed(value):
    result = {}
    for item in value.split(","):
        key, sep, raw = item.partition("=")
        if not sep or key not in DRIFTS: raise argparse.ArgumentTypeError("use TYPE=0..1")
        result[key] = float(raw)
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Export frozen cloud visual teacher cache")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--model-path", required=True, help="local InternViT/DINO model directory")
    p.add_argument("--output", required=True)
    p.add_argument("--head-checkpoint", required=True,
                   help="trained linear/MLP BoxCars head checkpoint")
    p.add_argument("--task", choices=VALID_TASKS, default="make")
    p.add_argument("--teacher-name", default="InternViT-6B-224px")
    p.add_argument("--drift-types", type=parse_types, default=("illumination", "motion_blur", "sensor_noise"))
    p.add_argument("--fixed-severities", type=parse_fixed, default={"illumination": 1., "motion_blur": .8, "sensor_noise": .6})
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=0)
    return p.parse_args()


def make_dataset(a, split):
    base = BoxCarsMultiView(a.dataset_path, split, a.task, 4,
        transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    return PairedBoxCarsCameraDrift(base, a.drift_types, clean_probability=0., seed=42 + (0 if split == "train" else 1000003),
        fixed_severities=a.fixed_severities, return_metadata=True)


def output_tensor(output):
    if isinstance(output, torch.Tensor): return output
    if hasattr(output, "last_hidden_state"): return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output: return output[0]
    raise TypeError(f"unsupported teacher output: {type(output)}")


class Head(nn.Module):
    def __init__(self, dimension, classes, hidden):
        super().__init__()
        self.network = (nn.Linear(dimension, classes) if not hidden else
                        nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                      nn.Dropout(.1), nn.Linear(hidden, classes)))

    def forward(self, value): return self.network(value)


def load_head(path, classes, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = ("state_dict", "feature_dim", "num_classes", "hidden_dim")
    if not isinstance(payload, dict) or any(key not in payload for key in required):
        raise ValueError(f"head checkpoint must contain {required}")
    if int(payload["num_classes"]) != classes:
        raise ValueError(f"head has {payload['num_classes']} classes, dataset has {classes}")
    head = Head(int(payload["feature_dim"]), classes, int(payload["hidden_dim"]))
    head.load_state_dict(payload["state_dict"], strict=True)
    return head.to(device).eval(), payload


@torch.no_grad()
def extract(model, loader, device, max_samples, split_name):
    all_features, all_labels, all_indices, all_kinds, all_severities = [], [], [], [], []
    started = time.time()
    for step, batch in enumerate(loader):
        images, mask, labels = batch[1], batch[2], batch[3]
        # Paired dataset is ImageNet-normalized for MV-ViT.  Restore pixels and
        # apply InternViT's official CLIP processor normalization.
        pixels = (images * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
        pixels = ((pixels - CLIP_MEAN) / CLIP_STD).to(device, torch.bfloat16)
        b, v = pixels.shape[:2]
        hidden = output_tensor(model(pixels.reshape(b * v, *pixels.shape[2:])))
        if hidden.ndim == 3: hidden = hidden[:, 0]
        hidden = hidden.reshape(b, v, -1).float()
        weights = mask.to(hidden.device, hidden.dtype).unsqueeze(-1)
        feature = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        all_features.append(F.normalize(feature, dim=-1).cpu().half())
        all_labels.append(labels.cpu())
        all_indices.append(batch[-1].long().cpu())
        all_kinds.extend(list(batch[5]))
        all_severities.append(batch[6].float().cpu())
        if step == 0 or (step + 1) % 50 == 0:
            print(f"export {split_name}: batches={step + 1}/{len(loader)} "
                  f"elapsed={time.time() - started:.0f}s", flush=True)
        if max_samples and sum(x.shape[0] for x in all_labels) >= max_samples: break
    features, labels, indices = torch.cat(all_features), torch.cat(all_labels), torch.cat(all_indices)
    severities = torch.cat(all_severities)
    # Loader is deliberately sequential, but use index sorting as a hard
    # contract with the adapter trainer's cache lookup.
    order = indices.argsort()
    kinds = [all_kinds[index] for index in order.tolist()]
    return features[order], labels[order], indices[order], kinds, severities[order]


def main():
    a = parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("cloud teacher extraction requires CUDA")
    # InternViT's released custom class targets an older Transformers API.
    # The a4 environment is newer and asks custom models for this attribute
    # while finalizing weights.  The model has no tied weights, so the empty
    # mapping is the correct compatibility value (no package downgrade needed).
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = property(lambda _self: {})
    from transformers import AutoModel
    device = torch.device("cuda")
    model = AutoModel.from_pretrained(a.model_path, trust_remote_code=True, low_cpu_mem_usage=True,
                                      torch_dtype=torch.bfloat16).to(device).eval()
    datasets = {split: make_dataset(a, split) for split in ("train", "validation")}
    head, head_payload = load_head(a.head_checkpoint, len(datasets["train"].classes), device)
    loaders = {split: DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                  pin_memory=True, persistent_workers=a.num_workers > 0) for split, ds in datasets.items()}
    output = {"format_version": 2, "teacher": a.teacher_name,
              "feature_dim": int(head_payload["feature_dim"]),
              "classes": datasets["train"].classes,
              "drift_types": list(a.drift_types), "fixed_severities": a.fixed_severities,
              "logit_head": "trained_boxcars_head",
              "head_checkpoint": os.path.abspath(a.head_checkpoint)}
    for split in ("train", "validation"):
        features, labels, indices, kinds, severities = extract(
            model, loaders[split], device, a.max_samples, split)
        if not torch.equal(indices, torch.arange(len(indices))): raise RuntimeError("teacher cache indices must be contiguous")
        if features.shape[1] != int(head_payload["feature_dim"]):
            raise RuntimeError(f"teacher feature dim {features.shape[1]} != head dim {head_payload['feature_dim']}")
        logits = []
        for start in range(0, len(features), 2048):
            logits.append(head(features[start:start + 2048].to(device).float()).cpu().half())
        output[split] = {"features": features, "logits": torch.cat(logits),
                         "labels": labels, "indices": indices,
                         "sample_drift_types": kinds, "sample_severities": severities,
                         "seed": 42 + (0 if split == "train" else 1000003)}
        print(f"{split}: {len(labels)} tracks, feature_dim={features.shape[1]}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    torch.save(output, a.output)
    print(f"saved cloud teacher cache: {a.output}")


if __name__ == "__main__": main()
