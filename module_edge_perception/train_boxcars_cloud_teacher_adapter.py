"""Cloud-teacher-guided BoxCars AdaptFormer refresh (the u=1 path).

The cloud vision foundation model is deliberately *not* part of the edge
forward graph.  ``--teacher-cache`` is produced on the cloud from drifted
representative tracks and contains teacher logits/features.  This program
only optimizes the small edge AdaptFormer; its output checkpoint remains an
adapter-only u=1 payload.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from adaptformer import attach_adaptformer, count_adapter_parameters, freeze_backbone, save_adapter_checkpoint, set_adapter_enabled
from boxcars_camera_drift_dataset import DRIFTS, PairedBoxCarsCameraDrift
from boxcars_dataset import BoxCarsMultiView, VALID_TASKS
from model import EarlyFusionMultiViewViT


def drift_types(value):
    result = tuple(x.strip() for x in value.split(",") if x.strip())
    if not result or set(result) - set(DRIFTS):
        raise argparse.ArgumentTypeError(f"choose comma-separated values from {DRIFTS}")
    return result


def fixed_severities(value):
    result = {}
    for item in value.split(","):
        name, sep, number = item.partition("=")
        if not sep or name not in DRIFTS:
            raise argparse.ArgumentTypeError("use TYPE=0..1")
        result[name] = float(number)
    return result


def args_parser():
    p = argparse.ArgumentParser(description="Cloud visual-teacher guided adapter refresh")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--baseline-checkpoint", required=True)
    p.add_argument("--teacher-cache", required=True,
                   help="cloud cache: train/validation each contain logits [N,C] and features [N,D]")
    p.add_argument("--task", choices=VALID_TASKS, default="make")
    p.add_argument("--expert-name", default="cloud_teacher")
    p.add_argument("--drift-types", type=drift_types,
                   default=("illumination", "motion_blur", "sensor_noise"))
    p.add_argument("--fixed-severities", type=fixed_severities,
                   default={"illumination": 1., "motion_blur": .8, "sensor_noise": .6})
    p.add_argument("--random-train-severity", action="store_true",
                   help="train on a severity range; fixed severities remain validation-only")
    p.add_argument("--severity-min", type=float, default=.3)
    p.add_argument("--severity-max", type=float, default=1.)
    p.add_argument("--clean-probability", type=float, default=0.,
                   help="deprecated alignment guard; formal teacher-cache training requires 0")
    p.add_argument("--ce-weight", type=float, default=1.,
                   help="set 0 for deployment-realistic unlabeled refresh")
    p.add_argument("--kd-weight", type=float, default=.35)
    p.add_argument("--teacher-feature-weight", type=float, default=.25)
    p.add_argument("--anchor-weight", type=float, default=.20)
    p.add_argument("--train-norm-head", action="store_true",
                   help="ablation only; formal u=1 refresh trains AdaptFormer, not norm/head")
    p.add_argument("--temperature", type=float, default=2.)
    p.add_argument("--model-name", default="vit_small_patch16_224")
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--r", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accumulation-steps", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=.05)
    p.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--save-dir", default="./checkpoints/boxcars_drift_adapters")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-clean-accuracy", type=float, default=.928,
                   help="do not select a drift checkpoint that damages clean capability")
    return p.parse_args()


def load_baseline(model, path):
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload, strict=True)


def make_set(a, split):
    # No random augmentation: cache row i must always be exactly dataset item i.
    base = BoxCarsMultiView(a.dataset_path, split, a.task, a.num_views,
                            transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    return PairedBoxCarsCameraDrift(
        base, a.drift_types, clean_probability=0.,
        severity_min=a.severity_min, severity_max=a.severity_max,
        fixed_severities=None if a.random_train_severity else a.fixed_severities,
        seed=42 + (0 if split == "train" else 1000003), return_metadata=True)


def make_fixed_validation_set(a, drift=None, severity=0.):
    base = BoxCarsMultiView(a.dataset_path, "validation", a.task, a.num_views,
                            transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    selected = drift or "illumination"
    return PairedBoxCarsCameraDrift(base, (selected,), clean_probability=0.,
        fixed_drift=selected, fixed_severity=severity, seed=123, return_metadata=True)


def load_cache(path, split, length, classes):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    part = payload.get(split) if isinstance(payload, dict) else None
    if not isinstance(part, dict):
        raise ValueError(f"teacher cache missing '{split}' section")
    logits, features = part.get("logits"), part.get("features")
    if not isinstance(logits, torch.Tensor) or not isinstance(features, torch.Tensor):
        raise ValueError("teacher cache must contain tensor logits and features")
    if logits.shape != (length, classes) or features.ndim != 2 or features.shape[0] != length:
        raise ValueError(f"bad {split} cache: logits={tuple(logits.shape)}, features={tuple(features.shape)}")
    indices = part.get("indices")
    kinds, severities = part.get("sample_drift_types"), part.get("sample_severities")
    if not isinstance(indices, torch.Tensor) or not torch.equal(indices.long(), torch.arange(length)):
        raise ValueError(f"{split} cache must contain contiguous sample indices")
    if not isinstance(kinds, list) or len(kinds) != length:
        raise ValueError(f"{split} cache must contain one drift type per sample")
    if not isinstance(severities, torch.Tensor) or severities.shape != (length,):
        raise ValueError(f"{split} cache must contain one severity per sample")
    return {"logits": logits.float(), "features": F.normalize(features.float(), dim=-1),
            "indices": indices.long(), "drift_types": kinds, "severities": severities.float(),
            "payload": payload}


def verify_batch_alignment(cache, indices, kinds, severities):
    rows = indices.long().tolist()
    expected_kinds = [cache["drift_types"][index] for index in rows]
    if list(kinds) != expected_kinds:
        raise RuntimeError(f"teacher/student drift mismatch: {list(kinds)[:4]} != {expected_kinds[:4]}")
    expected_severity = cache["severities"][indices.long()].float()
    if not torch.allclose(severities.float().cpu(), expected_severity, atol=1e-6, rtol=0):
        raise RuntimeError("teacher/student severity mismatch")


def kd_loss(student, teacher, temperature):
    t = float(temperature)
    return F.kl_div(F.log_softmax(student.float() / t, dim=1),
                    F.softmax(teacher.float() / t, dim=1), reduction="batchmean") * t * t


def main():
    a = args_parser()
    torch.manual_seed(a.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("cloud-teacher adapter refresh requires CUDA")
    if min(a.ce_weight, a.kd_weight, a.teacher_feature_weight, a.anchor_weight) < 0 or a.temperature <= 0:
        raise ValueError("loss weights must be non-negative and temperature positive")
    if a.clean_probability != 0:
        raise ValueError("--clean-probability must be 0: teacher cache and student drift input must align")
    device = torch.device("cuda")
    train_set = make_set(a, "train")
    common = dict(batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=True,
                  persistent_workers=a.num_workers > 0)
    train_generator = torch.Generator().manual_seed(a.seed)
    train_loader = DataLoader(train_set, shuffle=True, generator=train_generator, **common)
    validation_specs = {"clean": (None, 0.), "illumination_1.0": ("illumination", 1.),
                        "motion_blur_0.8": ("motion_blur", .8),
                        "sensor_noise_0.6": ("sensor_noise", .6)}
    val_loaders = {name: DataLoader(make_fixed_validation_set(a, *spec), shuffle=False, **common)
                   for name, spec in validation_specs.items()}
    train_cache = load_cache(a.teacher_cache, "train", len(train_set), len(train_set.classes))
    train_logits, train_features = train_cache["logits"], train_cache["features"]
    cache_payload = train_cache["payload"]
    if cache_payload.get("logit_head") != "trained_boxcars_head":
        raise ValueError("formal Adapter training requires logits from a trained BoxCars head")
    if tuple(cache_payload.get("drift_types", ())) != tuple(a.drift_types):
        raise ValueError("teacher cache drift_types do not match Adapter dataset")
    if a.random_train_severity:
        if cache_payload.get("severity_mode") != "random_range":
            raise ValueError("Adapter requests random severity but teacher cache does not")
        if abs(float(cache_payload.get("severity_min", -1)) - a.severity_min) > 1e-6 \
                or abs(float(cache_payload.get("severity_max", -1)) - a.severity_max) > 1e-6:
            raise ValueError("teacher cache severity range does not match Adapter dataset")
    else:
        cached_fixed = cache_payload.get("fixed_severities", {})
        if any(abs(float(cached_fixed.get(name, -1)) - float(value)) > 1e-6
               for name, value in a.fixed_severities.items()):
            raise ValueError("teacher cache fixed severities do not match Adapter dataset")

    model = EarlyFusionMultiViewViT(a.model_name, a.num_views, len(train_set.classes), pretrained=False)
    load_baseline(model, a.baseline_checkpoint)
    attach_adaptformer(model, r=a.r)
    freeze_backbone(model)
    if not a.train_norm_head:
        for parameter in model.norm.parameters(): parameter.requires_grad = False
        for parameter in model.head.parameters(): parameter.requires_grad = False
    # Cloud-only projector is required only while training; it is excluded from
    # save_adapter_checkpoint, so it is never sent to the edge.
    projector = torch.nn.Sequential(torch.nn.LayerNorm(model.head.in_features),
                                    torch.nn.Linear(model.head.in_features, train_features.shape[1], bias=False)).to(device)
    model.to(device)
    optimizer = torch.optim.AdamW(list(p for p in model.parameters() if p.requires_grad) + list(projector.parameters()), lr=a.lr, weight_decay=a.weight_decay)
    amp = torch.bfloat16 if a.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp_dtype == "fp16")
    best = -1.
    out_dir = os.path.join(a.save_dir, a.expert_name); os.makedirs(out_dir, exist_ok=True)
    print(f"cloud_teacher_cache={a.teacher_cache} teacher_dim={train_features.shape[1]} adapter_params={count_adapter_parameters(model):,}")
    for epoch in range(a.epochs):
        model.train(); projector.train(); optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(6, device=device)
        limit = a.max_train_batches or len(train_loader)
        for step, batch in enumerate(train_loader):
            if step >= limit: break
            clean, corrupt, mask, labels = (x.to(device, non_blocking=True) for x in batch[:4])
            indices = batch[-1].long()
            verify_batch_alignment(train_cache, indices, batch[5], batch[6])
            tz, th = train_logits[indices].to(device), train_features[indices].to(device)
            set_adapter_enabled(model, False)
            with torch.no_grad(), torch.autocast("cuda", dtype=amp):
                clean_z, _ = model(clean, view_mask=mask, return_features=True)
            set_adapter_enabled(model, True)
            with torch.autocast("cuda", dtype=amp):
                student_z, student_h = model(corrupt, view_mask=mask, return_features=True)
                replay_z = model(clean, view_mask=mask)
                ce = F.cross_entropy(student_z, labels)
                kd = kd_loss(student_z, tz, a.temperature)
                feature = 1 - F.cosine_similarity(F.normalize(projector(student_h.float()), dim=-1), th, dim=-1).mean()
                anchor = kd_loss(replay_z, clean_z, a.temperature)
                loss = a.ce_weight * ce + a.kd_weight * kd + a.teacher_feature_weight * feature + a.anchor_weight * anchor
            scaler.scale(loss / a.accumulation_steps).backward()
            if (step + 1) % a.accumulation_steps == 0 or step + 1 == limit:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            batch_count = labels.numel()
            totals += torch.tensor([loss.item() * batch_count, ce.item() * batch_count,
                                    kd.item() * batch_count, feature.item() * batch_count,
                                    anchor.item() * batch_count, batch_count], device=device)
        model.eval(); validation = {}
        with torch.no_grad():
            for name, loader in val_loaders.items():
                correct = count = 0
                for step, batch in enumerate(loader):
                    if a.max_val_batches and step >= a.max_val_batches: break
                    source = batch[0] if name == "clean" else batch[1]
                    images, mask, labels = source.to(device), batch[2].to(device), batch[3].to(device)
                    with torch.autocast("cuda", dtype=amp):
                        z = model(images, view_mask=mask)
                    correct += (z.argmax(1) == labels).sum().item(); count += labels.numel()
                validation[name] = correct / max(count, 1)
        accuracy = sum(validation[name] for name in validation_specs if name != "clean") / 3
        n = max(totals[-1].item(), 1)
        meta = {"epoch": epoch + 1, "teacher_cache": os.path.abspath(a.teacher_cache),
                "teacher_feature_dim": train_features.shape[1],
                "teacher_type": "cloud_visual_foundation_model", "conditioned_adapter": False,
                "validation_accuracy": validation, "validation_mean_drift_accuracy": accuracy,
                "args": vars(a)}
        save_adapter_checkpoint(os.path.join(out_dir, "latest.pth"), model,
                                include_norm_head=a.train_norm_head, **meta)
        if accuracy > best and validation["clean"] >= a.min_clean_accuracy:
            best = accuracy
            save_adapter_checkpoint(os.path.join(out_dir, "best.pth"), model,
                                    include_norm_head=a.train_norm_head, **meta)
            with open(os.path.join(out_dir, "manifest.json"), "w") as f: json.dump(meta, f, indent=2)
        scores = " ".join(f"{name}={value:.4f}" for name, value in validation.items())
        print(f"epoch={epoch+1}/{a.epochs} loss={totals[0]/n:.4f} ce={totals[1]/n:.4f} "
              f"kd={totals[2]/n:.4f} feature={totals[3]/n:.4f} anchor={totals[4]/n:.4f} "
              f"mean_drift={accuracy:.4f} {scores}")
    if best < 0:
        raise RuntimeError(f"no Adapter checkpoint preserved clean accuracy >= {a.min_clean_accuracy:.4f}")


if __name__ == "__main__":
    main()
