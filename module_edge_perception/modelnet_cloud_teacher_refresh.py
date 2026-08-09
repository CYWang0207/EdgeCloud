"""Leakage-controlled InternViT-6B cloud-teacher pipeline for ModelNet40.

This is the formal ModelNet40 implementation of the current method, not the
historical supervised camera Adapter.  It trains a new 40-class task head on
frozen InternViT features, gates the teacher, refreshes an edge AdaptFormer
without uploaded-track labels, and evaluates fixed checkpoints on official
test.

The protocol has three disjoint, deterministic class-stratified portions of
the official training split:

* task-head set: offline labelled data used once to fit the cloud task head;
* refresh set: representative drifted tracks.  Their labels never enter the
  Adapter loss (CE is exactly zero);
* development set: fixed corruptions used to select the checkpoint.

The official test split is touched only after both the head and Adapter have
been selected.  Historical supervised Adapter weights are never loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, WeightedRandomSampler

from adaptformer import (attach_adaptformer, count_adapter_parameters,
                         freeze_backbone, load_adapter_checkpoint,
                         save_adapter_checkpoint, set_adapter_enabled)
from dataset import ModelNet40MultiView
from model import EarlyFusionMultiViewViT
from modelnet_camera_drift_dataset import DRIFTS, PairedModelNetDrift


# Edge inputs are ImageNet normalised; the frozen InternViT expects these
# ImageNet pixels re-normalised with its own processor statistics.
IM = torch.tensor((.485, .456, .406)).view(1, 1, 3, 1, 1)
IS = torch.tensor((.229, .224, .225)).view(1, 1, 3, 1, 1)
CM = torch.tensor((.48145466, .4578275, .40821073)).view(1, 1, 3, 1, 1)
CS = torch.tensor((.26862954, .26130258, .27577711)).view(1, 1, 3, 1, 1)
SPECS = {
    "clean": ("normal", 0.),
    "illumination_1.0": ("illumination", 1.),
    "defocus_0.2": ("defocus", .2),
    "sensor_noise_0.4": ("sensor_noise", .4),
}
TRAIN_DRIFTS = ("illumination", "defocus", "sensor_noise")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ModelNet40 cloud-unlabelled Adapter refresh")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--baseline-checkpoint", required=True)
    p.add_argument("--teacher-model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--head-per-class", type=int, default=0,
                   help="0 uses every train sample left after refresh/dev reservation")
    p.add_argument("--refresh-per-class", type=int, default=12)
    p.add_argument("--dev-per-class", type=int, default=8)
    p.add_argument("--teacher-batch-size", type=int, default=4)
    p.add_argument("--edge-batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--head-epochs", type=int, default=80)
    p.add_argument("--head-hidden-dim", type=int, default=512)
    p.add_argument("--head-lr", type=float, default=3e-3)
    p.add_argument("--adapter-epochs", type=int, default=4)
    p.add_argument("--adapter-lr", type=float, default=2e-4)
    p.add_argument("--r", type=int, default=32)
    p.add_argument("--min-teacher-gain", type=float, default=0.,
                   help="minimum development mean-drift gain over frozen Edge required before distillation")
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--resume-cache", action="store_true")
    p.add_argument("--teacher-only", action="store_true",
                   help="stop after writing the task head and teacher gate")
    p.add_argument("--illumination-tune", action="store_true",
                   help="run dev-selected unlabeled profiles that oversample illumination")
    return p.parse_args()


def make_base(path: str) -> ModelNet40MultiView:
    return ModelNet40MultiView(path, "train", transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
    ]), 4)


def class_split(base: ModelNet40MultiView, sizes: tuple[int, int, int], seed: int):
    """Return disjoint original dataset indices, balanced by class."""
    groups: dict[int, list[int]] = defaultdict(list)
    for index, (_prefix, label) in enumerate(base.samples):
        groups[int(label)].append(index)
    result = [[], [], []]
    for label in range(len(base.classes)):
        candidates = groups[label]
        fixed_needed = sum(size for size in sizes if size > 0)
        if len(candidates) < fixed_needed + (1 if 0 in sizes else 0):
            raise ValueError(f"class {base.classes[label]} has {len(candidates)} train samples, need > {fixed_needed}")
        rng = random.Random(seed + label * 1009)
        rng.shuffle(candidates)
        cursor = 0
        for part_index, (part, size) in enumerate(zip(result, sizes)):
            if size == 0:
                if sum(value == 0 for value in sizes) != 1:
                    raise ValueError("only one split may use all remaining samples")
                size = len(candidates) - sum(value for value in sizes[part_index + 1:] if value > 0) - cursor
            part.extend(candidates[cursor:cursor + size]); cursor += size
    return tuple(sorted(part) for part in result)


def paired(base, indices, *, random_train: bool, seed: int, metadata: bool = True):
    if random_train:
        data = PairedModelNetDrift(base, TRAIN_DRIFTS, severity_min=.3, severity_max=1.,
                                   clean_probability=0., seed=seed, normalize=True,
                                   return_metadata=metadata)
    else:
        # The caller makes a separate dataset for every fixed development case.
        raise ValueError("fixed drift must be constructed with fixed_paired")
    return Subset(data, indices)


def fixed_paired(base, indices, kind: str, severity: float, seed: int):
    source = "illumination" if kind == "normal" else kind
    data = PairedModelNetDrift(base, (source,), clean_probability=0., seed=seed,
                               fixed_drift=source, fixed_severity=severity,
                               normalize=True, return_metadata=True)
    return Subset(data, indices)


@torch.inference_mode()
def teacher_features(teacher, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    pixels = ((images * IS + IM).clamp(0, 1) - CM) / CS
    batch, views = pixels.shape[:2]
    output = teacher(pixels.reshape(batch * views, *pixels.shape[2:]).to(device, torch.bfloat16))
    hidden = output[0] if isinstance(output, (tuple, list)) else output.last_hidden_state
    hidden = hidden[:, 0] if hidden.ndim == 3 else hidden
    # Preserve view-specific evidence for these sparse canonical renderings.
    # The task head averages logits at track level after classifying each view.
    return F.normalize(hidden.reshape(batch, views, -1).float(), dim=-1)


def cache_part(teacher, data, device, batch_size, workers, *, include_clean: bool, name: str):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    result = {"clean_features": [], "corrupt_features": [], "labels": [],
              "indices": [], "drift_types": [], "severities": []}
    started = time.time()
    for step, batch in enumerate(loader):
        clean, corrupt, labels, kinds, severities, indices = batch
        if include_clean:
            result["clean_features"].append(teacher_features(teacher, clean, device).cpu().half())
        result["corrupt_features"].append(teacher_features(teacher, corrupt, device).cpu().half())
        result["labels"].append(labels.cpu())
        result["indices"].append(indices.long().cpu())
        result["drift_types"].extend(list(kinds))
        result["severities"].append(severities.float().cpu())
        if step % 20 == 0:
            print(f"teacher cache {name}: {step + 1}/{len(loader)} batches, {time.time() - started:.0f}s", flush=True)
    payload = {key: (torch.cat(value) if key not in ("drift_types",) else value)
               for key, value in result.items() if value}
    # Fixed-development and refresh caches do not need clean teacher features.
    # Keep the key for a stable, inspectable cache schema without attempting
    # torch.cat([]).
    if "clean_features" not in payload:
        payload["clean_features"] = torch.empty((0, *payload["corrupt_features"].shape[1:]), dtype=torch.float16)
    return payload


class Head(nn.Module):
    def __init__(self, feature_dim: int, classes: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                                     nn.Dropout(.1), nn.Linear(hidden_dim, classes))

    def forward(self, x): return self.network(x)


@torch.no_grad()
def head_accuracy(head, features, labels, device):
    correct = 0
    for start in range(0, len(labels), 2048):
        batch = features[start:start + 2048].to(device).float()
        logits = head(batch.flatten(0, 1)).reshape(len(batch), batch.shape[1], -1).mean(1)
        correct += logits.argmax(1).cpu().eq(labels[start:start + 2048]).sum().item()
    return correct / max(len(labels), 1)


def train_head(train_cache, dev_caches, classes, a, device):
    x = torch.cat((train_cache["clean_features"], train_cache["corrupt_features"])).float()
    y = torch.cat((train_cache["labels"], train_cache["labels"]))
    views = x.shape[1]
    x = x.flatten(0, 1)
    y = y.repeat_interleave(views)
    head = Head(x.shape[-1], classes, a.head_hidden_dim).to(device)
    loader = DataLoader(TensorDataset(x, y), batch_size=1024, shuffle=True)
    opt = torch.optim.AdamW(head.parameters(), lr=a.head_lr, weight_decay=1e-4)
    best_score, best_state, history = -1., None, []
    for epoch in range(1, a.head_epochs + 1):
        head.train()
        for bx, by in loader:
            loss = F.cross_entropy(head(bx.to(device)), by.to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        head.eval()
        scores = {name: head_accuracy(head, cache["corrupt_features"], cache["labels"], device)
                  for name, cache in dev_caches.items()}
        mean_drift = sum(scores[name] for name in SPECS if name != "clean") / 3
        history.append({"epoch": epoch, "accuracy": scores, "mean_drift": mean_drift})
        if mean_drift > best_score:
            best_score = mean_drift
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"head epoch={epoch}/{a.head_epochs} mean_drift={mean_drift:.2%} " +
                  " ".join(f"{k}={v:.2%}" for k, v in scores.items()), flush=True)
    head.load_state_dict(best_state)
    return head, {"best_mean_drift": best_score, "history": history,
                  "accuracy": history[max(range(len(history)), key=lambda i: history[i]["mean_drift"])]["accuracy"]}


def load_baseline(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload, strict=True)


def load_teacher(path, device):
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = property(lambda _self: {})
    from transformers import AutoModel
    return AutoModel.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                     trust_remote_code=True).to(device).eval()


def cache_row_lookup(cache):
    return {int(index): row for row, index in enumerate(cache["indices"].tolist())}


def evaluate_edge(model, data, device, amp, batch_size, workers, use_corrupt: bool):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch[1] if use_corrupt else batch[0]
            labels = batch[2].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp):
                logits = model(images.to(device, non_blocking=True), apply_quality_gate=False)
            correct += logits.argmax(1).eq(labels).sum().item(); total += len(labels)
    return correct / max(total, 1)


def train_adapter(base, refresh_indices, refresh_cache, dev_indices, a, device, mode, profile=None):
    profile = profile or {}
    torch.manual_seed(a.seed + 911)
    data = paired(base, refresh_indices, random_train=True, seed=a.seed)
    illumination_weight = float(profile.get("illumination_weight", 1.))
    sampler = None
    if illumination_weight != 1.:
        sample_weights = [illumination_weight if kind == "illumination" else 1.
                          for kind in refresh_cache["drift_types"]]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True,
                                        generator=torch.Generator().manual_seed(a.seed + 17))
    loader = DataLoader(data, batch_size=a.edge_batch_size, shuffle=sampler is None, sampler=sampler,
                        generator=torch.Generator().manual_seed(a.seed), num_workers=a.num_workers,
                        pin_memory=True, persistent_workers=a.num_workers > 0)
    model = EarlyFusionMultiViewViT("vit_small_patch16_224", 4, len(base.classes), pretrained=False)
    load_baseline(model, a.baseline_checkpoint)
    attach_adaptformer(model, r=a.r, condition_dim=0); freeze_backbone(model)
    for parameter in model.norm.parameters(): parameter.requires_grad = False
    for parameter in model.head.parameters(): parameter.requires_grad = False
    model.to(device)
    projector = nn.Sequential(nn.LayerNorm(model.head.in_features),
                              nn.Linear(model.head.in_features, refresh_cache["features"].shape[1], bias=False)).to(device)
    opt = torch.optim.AdamW(list(p for p in model.parameters() if p.requires_grad) + list(projector.parameters()),
                            lr=a.adapter_lr, weight_decay=.05)
    amp = torch.bfloat16
    rows = cache_row_lookup(refresh_cache)
    dev_sets = {name: fixed_paired(base, dev_indices, *spec, a.seed + 300 + index)
                for index, (name, spec) in enumerate(SPECS.items())}
    best, best_meta = -1., None
    epochs = int(profile.get("epochs", a.adapter_epochs))
    kd_weight = float(profile.get("kd_weight", .7))
    feature_weight = float(profile.get("feature_weight", .3))
    anchor_weight = float(profile.get("anchor_weight", .2))
    for epoch in range(1, epochs + 1):
        model.train(); projector.train(); losses = []
        for clean, corrupt, labels, kinds, severities, indices in loader:
            cache_indices = [rows[int(index)] for index in indices.tolist()]
            expected_kind = [refresh_cache["drift_types"][index] for index in cache_indices]
            expected_severity = refresh_cache["severities"][cache_indices]
            if list(kinds) != expected_kind or not torch.allclose(severities.float(), expected_severity, atol=1e-6, rtol=0):
                raise RuntimeError("teacher cache / refresh batch drift mismatch")
            teacher_logits = refresh_cache["logits"][cache_indices].to(device)
            teacher_feature = refresh_cache["features"][cache_indices].to(device)
            clean, corrupt, labels = clean.to(device), corrupt.to(device), labels.to(device)
            set_adapter_enabled(model, False)
            with torch.no_grad(), torch.autocast("cuda", dtype=amp):
                baseline_clean = model(clean, apply_quality_gate=False)
            set_adapter_enabled(model, True)
            with torch.autocast("cuda", dtype=amp):
                student_logits, student_feature = model(corrupt, return_features=True, apply_quality_gate=False)
                replay = model(clean, apply_quality_gate=False)
                kd = F.kl_div(F.log_softmax(student_logits.float() / 2., 1),
                              F.softmax(teacher_logits.float() / 2., 1), reduction="batchmean") * 4.
                feature = 1 - F.cosine_similarity(F.normalize(projector(student_feature.float()), dim=-1),
                                                   teacher_feature, dim=-1).mean()
                anchor = F.kl_div(F.log_softmax(replay.float() / 2., 1),
                                  F.softmax(baseline_clean.float() / 2., 1), reduction="batchmean") * 4.
                ce = F.cross_entropy(student_logits.float(), labels)
                if mode == "cloud_unlabeled" or mode.startswith("illum_"):
                    # The labels tensor is carried by the dataset for auditing,
                    # but never participates in this method's loss.
                    loss = kd_weight * kd + feature_weight * feature + anchor_weight * anchor
                    ce_weight = 0.
                elif mode == "label_only":
                    loss = ce + anchor_weight * anchor
                    ce_weight = 1.
                elif mode == "hybrid":
                    loss = ce + kd_weight * kd + feature_weight * feature + anchor_weight * anchor
                    ce_weight = 1.
                else:
                    raise ValueError(f"unknown Adapter mode: {mode}")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(loss.item())
        scores = {name: evaluate_edge(model, data, device, amp, a.edge_batch_size, a.num_workers,
                                      use_corrupt=name != "clean") for name, data in dev_sets.items()}
        mean_drift = sum(scores[name] for name in SPECS if name != "clean") / 3
        meta = {"epoch": epoch, "validation_accuracy": scores, "profile": profile,
                "validation_mean_drift_accuracy": mean_drift,
                "loss": sum(losses) / max(len(losses), 1), "ce_weight": ce_weight,
                "adapter_params": count_adapter_parameters(model)}
        print(f"adapter mode={mode} epoch={epoch}/{epochs} loss={meta['loss']:.4f} mean_drift={mean_drift:.2%} " +
              " ".join(f"{k}={v:.2%}" for k, v in scores.items()), flush=True)
        if mean_drift > best:
            best, best_meta = mean_drift, meta
            save_adapter_checkpoint(os.path.join(a.output_dir, f"{mode}_adapter.pth"), model,
                                    include_norm_head=False, **meta)
    return model, best_meta


def evaluate_edge_predictions(model, data, device, amp, batch_size, workers, use_corrupt):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    predictions, labels_out, indices_out = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch[1] if use_corrupt else batch[0]
            labels, indices = batch[2], batch[5]
            with torch.autocast("cuda", dtype=amp):
                logits = model(images.to(device, non_blocking=True), apply_quality_gate=False)
            predictions.extend(logits.argmax(1).cpu().tolist())
            labels_out.extend(labels.tolist()); indices_out.extend(indices.tolist())
    accuracy = sum(int(p == y) for p, y in zip(predictions, labels_out)) / max(len(labels_out), 1)
    return {"accuracy": accuracy, "predictions": predictions, "labels": labels_out, "indices": indices_out}


def final_test(base_path, models, a, device):
    test_base = ModelNet40MultiView(base_path, "test", transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
    ]), 4)
    indices = list(range(len(test_base)))
    test_sets = {name: fixed_paired(test_base, indices, *spec, a.seed + 700 + i)
                 for i, (name, spec) in enumerate(SPECS.items())}
    amp = torch.bfloat16
    results = {}
    for method, model in models.items():
        results[method] = {name: evaluate_edge_predictions(model, data, device, amp,
                                                            a.edge_batch_size, a.num_workers,
                                                            use_corrupt=name != "clean")
                           for name, data in test_sets.items()}
    return results


def main():
    a = parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("cloud-teacher refresh requires CUDA")
    if a.head_per_class < 0 or min(a.refresh_per_class, a.dev_per_class) < 1: raise ValueError("invalid per-class budgets")
    random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    base = make_base(a.dataset_path)
    head_indices, refresh_indices, dev_indices = class_split(base, (a.head_per_class, a.refresh_per_class, a.dev_per_class), a.seed)
    split_manifest = {"seed": a.seed, "classes": base.classes, "head_indices": head_indices,
                      "refresh_indices": refresh_indices, "dev_indices": dev_indices}
    with open(out / "split_manifest.json", "w") as f: json.dump(split_manifest, f, indent=2)
    print(f"split: head={len(head_indices)}, refresh={len(refresh_indices)}, dev={len(dev_indices)}", flush=True)

    cache_file = out / "teacher_feature_cache.pt"
    if a.resume_cache and cache_file.exists():
        caches = torch.load(cache_file, map_location="cpu", weights_only=False)
    else:
        teacher = load_teacher(a.teacher_model_path, device)
        head_train = paired(base, head_indices, random_train=True, seed=a.seed)
        caches = {"head_train": cache_part(teacher, head_train, device, a.teacher_batch_size, a.num_workers,
                                            include_clean=True, name="head_train")}
        for i, (name, spec) in enumerate(SPECS.items()):
            caches[name] = cache_part(teacher, fixed_paired(base, dev_indices, *spec, a.seed + 300 + i),
                                      device, a.teacher_batch_size, a.num_workers, include_clean=False, name=name)
        refresh_data = paired(base, refresh_indices, random_train=True, seed=a.seed)
        caches["refresh"] = cache_part(teacher, refresh_data, device, a.teacher_batch_size, a.num_workers,
                                       include_clean=False, name="refresh")
        torch.save(caches, cache_file)
        del teacher; torch.cuda.empty_cache()

    head, head_metrics = train_head(caches["head_train"], {name: caches[name] for name in SPECS}, len(base.classes), a, device)
    head_payload = {"state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                    "feature_dim": int(caches["refresh"]["corrupt_features"].shape[-1]), "classes": base.classes,
                    "hidden_dim": a.head_hidden_dim, "metrics": head_metrics}
    torch.save(head_payload, out / "teacher_head.pth")
    with open(out / "teacher_head_metrics.json", "w") as f: json.dump(head_metrics, f, indent=2)
    with torch.no_grad():
        feature_views = caches["refresh"]["corrupt_features"].to(device).float()
        refresh_logits = head(feature_views.flatten(0, 1)).reshape(len(feature_views), feature_views.shape[1], -1).mean(1).cpu().half()
    refresh_cache = {"features": F.normalize(caches["refresh"]["corrupt_features"].float().mean(1), dim=-1),
                     "logits": refresh_logits, "indices": caches["refresh"]["indices"],
                     "drift_types": caches["refresh"]["drift_types"], "severities": caches["refresh"]["severities"]}
    torch.save(refresh_cache, out / "refresh_teacher_cache.pt")

    # A teacher that loses to the frozen edge model on the same development
    # tracks is not admissible for the u=1 claim.  Stop here rather than
    # allowing an attractive-but-uninterpretable Adapter number to consume
    # compute or appear in a report.
    baseline = EarlyFusionMultiViewViT("vit_small_patch16_224", 4, len(base.classes), pretrained=False)
    load_baseline(baseline, a.baseline_checkpoint); baseline.to(device).eval()
    dev_sets = {name: fixed_paired(base, dev_indices, *spec, a.seed + 300 + i)
                for i, (name, spec) in enumerate(SPECS.items())}
    baseline_dev = {name: evaluate_edge(baseline, data, device, torch.bfloat16,
                                        a.edge_batch_size, a.num_workers,
                                        use_corrupt=name != "clean")
                    for name, data in dev_sets.items()}
    teacher_dev = head_metrics["accuracy"]
    baseline_mean = sum(baseline_dev[name] for name in SPECS if name != "clean") / 3
    teacher_mean = sum(teacher_dev[name] for name in SPECS if name != "clean") / 3
    gate = {"baseline_development": baseline_dev, "teacher_development": teacher_dev,
            "baseline_mean_drift": baseline_mean, "teacher_mean_drift": teacher_mean,
            "teacher_gain_over_edge": teacher_mean - baseline_mean,
            "minimum_required_gain": a.min_teacher_gain,
            "passed": teacher_mean - baseline_mean >= a.min_teacher_gain}
    with open(out / "teacher_gate.json", "w") as f: json.dump(gate, f, indent=2)
    if not gate["passed"]:
        print(json.dumps({"teacher_gate": "FAIL", **gate}, indent=2), flush=True)
        return
    if a.teacher_only:
        print(json.dumps({"teacher_gate": "PASS", **gate}, indent=2), flush=True)
        return

    if a.illumination_tune:
        # Profiles are selected using development data only.  The official
        # test is evaluated once, after the winning profile is fixed.
        profiles = {
            "illum_balanced": {"illumination_weight": 2., "kd_weight": 1.,
                               "feature_weight": .2, "anchor_weight": .15, "epochs": 8},
            "illum_focus": {"illumination_weight": 4., "kd_weight": 1.,
                            "feature_weight": .1, "anchor_weight": .15, "epochs": 8},
            "illum_strong": {"illumination_weight": 6., "kd_weight": 1.2,
                             "feature_weight": .05, "anchor_weight": .1, "epochs": 10},
        }
        candidate_meta = {}
        for name, profile in profiles.items():
            _model, candidate_meta[name] = train_adapter(base, refresh_indices, refresh_cache,
                                                          dev_indices, a, device, name, profile)
        eligible = []
        for name, meta in candidate_meta.items():
            scores = meta["validation_accuracy"]
            if (scores["clean"] >= baseline_dev["clean"] - .01 and
                    scores["defocus_0.2"] >= baseline_dev["defocus_0.2"] and
                    scores["sensor_noise_0.4"] >= baseline_dev["sensor_noise_0.4"]):
                eligible.append(name)
        pool = eligible or list(profiles)
        selected_name = max(pool, key=lambda name: (
            candidate_meta[name]["validation_accuracy"]["illumination_1.0"],
            candidate_meta[name]["validation_mean_drift_accuracy"]))
        selected_path = out / f"{selected_name}_adapter.pth"
        final_path = out / "cloud_unlabeled_illumination_tuned_adapter.pth"
        shutil.copy2(selected_path, final_path)
        selected = EarlyFusionMultiViewViT("vit_small_patch16_224", 4, len(base.classes), pretrained=False)
        load_baseline(selected, a.baseline_checkpoint); attach_adaptformer(selected, r=a.r, condition_dim=0)
        load_adapter_checkpoint(selected, final_path); selected.to(device).eval(); set_adapter_enabled(selected, True)
        baseline_model = EarlyFusionMultiViewViT("vit_small_patch16_224", 4, len(base.classes), pretrained=False)
        load_baseline(baseline_model, a.baseline_checkpoint); baseline_model.to(device).eval()
        models = {"edge_baseline": baseline_model, "cloud_unlabeled_illumination_tuned": selected}
        raw_test = final_test(a.dataset_path, models, a, device)
        test_scores = {method: {condition: payload["accuracy"] for condition, payload in conditions.items()}
                       for method, conditions in raw_test.items()}
        with open(out / "illumination_tuned_test_predictions.jsonl", "w") as f:
            for condition in SPECS:
                reference = raw_test["edge_baseline"][condition]
                for row, (index, label) in enumerate(zip(reference["indices"], reference["labels"])):
                    f.write(json.dumps({"condition": condition, "index": index, "label": label,
                                        "predictions": {method: raw_test[method][condition]["predictions"][row]
                                                        for method in models}}) + "\n")
        tuned_summary = {
            "selection_boundary": "profiles selected on train-internal development only; official test evaluated after selection",
            "profiles": profiles, "candidate_development": candidate_meta,
            "eligible_profiles": eligible, "selected_profile": selected_name,
            "baseline_development": baseline_dev, "teacher_development": teacher_dev,
            "official_test": test_scores,
            "delta_over_edge": {name: test_scores["cloud_unlabeled_illumination_tuned"][name] - test_scores["edge_baseline"][name]
                                for name in SPECS}}
        with open(out / "illumination_tuned_summary.json", "w") as f:
            json.dump(tuned_summary, f, indent=2)
        print(json.dumps(tuned_summary, indent=2), flush=True)
        return

    adapter_dev, models = {}, {}
    for mode in ("cloud_unlabeled", "label_only", "hybrid"):
        _model, adapter_dev[mode] = train_adapter(base, refresh_indices, refresh_cache,
                                                   dev_indices, a, device, mode)
        selected = EarlyFusionMultiViewViT("vit_small_patch16_224", 4, len(base.classes), pretrained=False)
        load_baseline(selected, a.baseline_checkpoint); attach_adaptformer(selected, r=a.r, condition_dim=0)
        load_adapter_checkpoint(selected, out / f"{mode}_adapter.pth")
        selected.to(device).eval(); set_adapter_enabled(selected, True)
        models[mode] = selected
    baseline_model = EarlyFusionMultiViewViT("vit_small_patch16_224", 4, len(base.classes), pretrained=False)
    load_baseline(baseline_model, a.baseline_checkpoint); baseline_model.to(device).eval()
    models = {"edge_baseline": baseline_model, **models}
    raw_test = final_test(a.dataset_path, models, a, device)
    test_scores = {method: {condition: payload["accuracy"] for condition, payload in conditions.items()}
                   for method, conditions in raw_test.items()}
    with open(out / "official_test_predictions.jsonl", "w") as f:
        for condition in SPECS:
            reference = raw_test["edge_baseline"][condition]
            for row, (index, label) in enumerate(zip(reference["indices"], reference["labels"])):
                record = {"condition": condition, "index": index, "label": label,
                          "predictions": {method: raw_test[method][condition]["predictions"][row]
                                          for method in models}}
                f.write(json.dumps(record) + "\n")
    result = {"protocol": "train-stratified head/refresh/dev; final official test fixed before test evaluation",
              "budgets_per_class": {"head": a.head_per_class, "refresh": a.refresh_per_class, "dev": a.dev_per_class},
              "actual_track_budgets": {"head": len(head_indices), "refresh": len(refresh_indices), "dev": len(dev_indices)},
              "teacher_dev": head_metrics["accuracy"],
              "adapter_dev": {mode: meta["validation_accuracy"] for mode, meta in adapter_dev.items()},
              "official_test": test_scores,
              "cloud_delta_over_edge": {name: test_scores["cloud_unlabeled"][name] - test_scores["edge_baseline"][name]
                                         for name in SPECS}}
    with open(out / "summary.json", "w") as f: json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
