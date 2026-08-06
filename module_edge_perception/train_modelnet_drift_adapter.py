"""Train an adapter-only ModelNet40 drift-correction expert.

The frozen MV-ViT baseline is used as the clean-domain reference.  The
adapter is optimized on paired clean/corrupted renderings, then saved in the
portable adapter-only format used by the scheduler's u=1 update path.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from adaptformer import (attach_adaptformer, count_adapter_parameters,
                         freeze_backbone, save_adapter_checkpoint,
                         set_adapter_enabled)
from dataset import ModelNet40MultiView
from model import EarlyFusionMultiViewViT

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "common"))
DRIFTS = ("illumination", "defocus", "sensor_noise", "compression", "partial_occlusion")
# A deployment mixture, not equal-probability synthetic augmentations.  Noise
# gets more mass because the measured ModelNet failure is large, while every
# other family remains represented in the adapter's domain.
DEFAULT_DRIFT_WEIGHTS = (0.25, 0.15, 0.30, 0.15, 0.15)
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def drift_list(value):
    result = tuple(x.strip() for x in value.split(",") if x.strip())
    if not result or set(result) - set(DRIFTS):
        raise argparse.ArgumentTypeError("drift-types must be drawn from " + ",".join(DRIFTS))
    return result


class PairedModelNetDrift(Dataset):
    """Aligned clean/corrupted pairs; train assignments are deterministic."""
    def __init__(self, base, drift_types, severity_min, severity_max, clean_probability,
                 seed, drift_weights=None, fixed_drift=None, fixed_severity=None):
        self.base, self.drift_types = base, drift_types
        self.severity_min, self.severity_max = severity_min, severity_max
        self.clean_probability, self.seed = clean_probability, seed
        self.drift_weights = torch.tensor(
            drift_weights if drift_weights is not None else [1.] * len(drift_types),
            dtype=torch.float,
        )
        if len(self.drift_weights) != len(drift_types) or (self.drift_weights < 0).any() or self.drift_weights.sum() == 0:
            raise ValueError("drift_weights must be non-negative and match drift_types")
        self.fixed_drift, self.fixed_severity = fixed_drift, fixed_severity

    def __len__(self):
        return len(self.base)

    @property
    def classes(self):
        return self.base.classes

    def __getitem__(self, index):
        clean, label = self.base[index]
        if self.fixed_drift is not None:
            kind, severity = self.fixed_drift, self.fixed_severity
        else:
            generator = torch.Generator().manual_seed(self.seed + index * 104729)
            if torch.rand((), generator=generator).item() < self.clean_probability:
                kind, severity = "normal", 0.0
            else:
                kind = self.drift_types[torch.multinomial(self.drift_weights, 1, generator=generator).item()]
                severity = self.severity_min + (self.severity_max - self.severity_min) * torch.rand((), generator=generator).item()
        corrupted = apply_camera_corruption(clean, kind, severity, self.seed + index * 1009)
        return torch.stack([NORM(image) for image in clean]), torch.stack([NORM(image) for image in corrupted]), label


def apply_camera_corruption(views, kind, severity, seed):
    """ModelNet camera/render degradation policy; input and output are [V,C,H,W] in [0,1].

    These are deliberately not global brightness multipliers.  The effects
    model capture/exposure and rendering defects while retaining an object
    visible enough for a meaningful robustness experiment.
    """
    generator = torch.Generator().manual_seed(seed)
    severity = float(max(0., min(1., severity)))
    if kind == "normal" or severity == 0:
        return views
    result = []
    for view_index, image in enumerate(views):
        local_seed = seed + view_index * 9176
        if kind == "illumination":
            # Exposure response (gamma), mild colour-temperature shift, and a
            # smooth local shadow/vignette rather than a uniform dark image.
            gamma = 1.0 + (0.75 * severity if view_index % 2 else -0.45 * severity)
            output = image.clamp(1e-4, 1).pow(gamma)
            cast = torch.tensor([1 + .12 * severity, 1., 1 - .10 * severity], dtype=image.dtype).view(3, 1, 1)
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, image.shape[1]), torch.linspace(-1, 1, image.shape[2]), indexing="ij")
            shadow = 1 - (.28 * severity) * ((xx - .25) ** 2 + (yy + .15) ** 2).clamp(max=1)
            output = output * cast * shadow.to(image.dtype)
        elif kind == "defocus":
            sigma = .25 + 1.45 * severity
            output = TF.gaussian_blur(image, kernel_size=5, sigma=[sigma, sigma])
        elif kind == "sensor_noise":
            # Poisson-Gaussian sensor noise is signal-dependent and differs by
            # camera/view, unlike adding one identical Gaussian field to all views.
            gain = 18.0 / (1 + 12 * severity)
            shot = torch.poisson((image * gain).clamp_min(0), generator=torch.Generator().manual_seed(local_seed)) / gain
            read = torch.randn(image.shape, generator=torch.Generator().manual_seed(local_seed + 1), dtype=image.dtype) * (.008 + .045 * severity)
            output = shot + read
        elif kind == "compression":
            # Render-resize-quantize approximates codec/rate degradation without
            # depending on a PIL/JPEG build on the training host.
            scale = 1 - .45 * severity
            small = F.interpolate(image.unsqueeze(0), scale_factor=scale, mode="bilinear", align_corners=False)
            output = F.interpolate(small, size=image.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
            levels = int(256 - 208 * severity)
            output = torch.round(output * (levels - 1)) / (levels - 1)
        elif kind == "partial_occlusion":
            output = image.clone()
            # Only one camera normally loses a local region; preserve the
            # remaining observations as the multi-view system's redundancy.
            if view_index == int(torch.randint(len(views), (), generator=generator)):
                h, w = image.shape[-2:]; side = int(min(h, w) * (.12 + .28 * severity))
                y = int(torch.randint(max(1, h - side), (), generator=generator)); x = int(torch.randint(max(1, w - side), (), generator=generator))
                output[:, y:y + side, x:x + side] = output.mean(dim=(-2, -1), keepdim=True)
        else:
            raise ValueError(f"unknown camera corruption: {kind}")
        result.append(output.clamp(0, 1))
    return torch.stack(result)


def args_parser():
    parser = argparse.ArgumentParser(description="ModelNet40 paired-drift AdaptFormer training")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--drift-types", type=drift_list, default=DRIFTS)
    parser.add_argument("--drift-weights", type=float, nargs="+", default=DEFAULT_DRIFT_WEIGHTS,
                        help="mixture weights aligned with drift-types")
    parser.add_argument("--severity-min", type=float, default=.4)
    parser.add_argument("--severity-max", type=float, default=.8)
    parser.add_argument("--clean-probability", type=float, default=.2)
    parser.add_argument("--feature-weight", type=float, default=.25)
    parser.add_argument("--consistency-weight", type=float, default=.20)
    parser.add_argument("--model-name", default="vit_small_patch16_224")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def load_baseline(model, path):
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def evaluate(model, loader, device, dtype, limit):
    model.eval(); totals = [0., 0., 0.]
    for step, (clean, corrupt, labels) in enumerate(loader):
        if limit and step >= limit: break
        clean, corrupt, labels = clean.to(device), corrupt.to(device), labels.to(device)
        with torch.autocast("cuda", dtype=dtype):
            clean_logits = model(clean); corrupt_logits = model(corrupt)
        totals[0] += (clean_logits.argmax(1) == labels).sum().item()
        totals[1] += (corrupt_logits.argmax(1) == labels).sum().item()
        totals[2] += labels.numel()
    return totals[0] / max(totals[2], 1), totals[1] / max(totals[2], 1)


def main():
    args = args_parser()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train = PairedModelNetDrift(ModelNet40MultiView(args.dataset_path, "train", transform, args.num_views), args.drift_types, args.severity_min, args.severity_max, args.clean_probability, args.seed, drift_weights=args.drift_weights)
    if len(args.drift_weights) != len(args.drift_types):
        raise ValueError("drift-weights length must match drift-types")
    validation = PairedModelNetDrift(ModelNet40MultiView(args.dataset_path, "test", transform, args.num_views), args.drift_types, .8, .8, 0., args.seed + 1, drift_weights=args.drift_weights, fixed_drift="sensor_noise", fixed_severity=.8)
    loaders = [DataLoader(data, batch_size=args.batch_size, shuffle=(i == 0), num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0) for i, data in enumerate((train, validation))]
    model = EarlyFusionMultiViewViT(args.model_name, args.num_views, len(train.classes), pretrained=False)
    load_baseline(model, args.baseline_checkpoint); attach_adaptformer(model, args.r); freeze_backbone(model); model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")
    os.makedirs(args.save_dir, exist_ok=True); best = -1.
    print(f"ModelNet40 camera-mixture adapter: train={len(train)}, test={len(validation)}, adapter_params={count_adapter_parameters(model)}, drifts={args.drift_types}, weights={args.drift_weights}")
    for epoch in range(args.epochs):
        model.train(); loss_sum = correct = count = 0
        for step, (clean, corrupt, labels) in enumerate(loaders[0]):
            if args.max_train_batches and step >= args.max_train_batches: break
            clean, corrupt, labels = clean.to(device), corrupt.to(device), labels.to(device)
            set_adapter_enabled(model, False)
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                clean_logits, clean_features = model(clean, return_features=True)
            set_adapter_enabled(model, True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                logits, features = model(corrupt, return_features=True)
                classification = F.cross_entropy(logits, labels)
                consistency = F.kl_div(F.log_softmax(logits.float(), 1), F.softmax(clean_logits.float(), 1), reduction="batchmean")
                alignment = 1 - F.cosine_similarity(features.float(), clean_features.float(), dim=1).mean()
                loss = classification + args.consistency_weight * consistency + args.feature_weight * alignment
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += loss.item() * labels.numel(); correct += (logits.argmax(1) == labels).sum().item(); count += labels.numel()
        scheduler.step(); clean_acc, corrupt_acc = evaluate(model, loaders[1], device, dtype, args.max_val_batches)
        print(f"epoch={epoch + 1}/{args.epochs} loss={loss_sum/max(count,1):.4f} train_acc={correct/max(count,1):.4f} test_clean={clean_acc:.4f} test_noise_0.8={corrupt_acc:.4f}")
        save_adapter_checkpoint(os.path.join(args.save_dir, "latest.pth"), model, dataset="modelnet40", baseline_checkpoint=args.baseline_checkpoint, r=args.r, drift_types=args.drift_types, epoch=epoch + 1)
        if corrupt_acc > best:
            best = corrupt_acc
            save_adapter_checkpoint(os.path.join(args.save_dir, "best.pth"), model, dataset="modelnet40", baseline_checkpoint=args.baseline_checkpoint, r=args.r, drift_types=args.drift_types, best_test_noise_0_8=best, epoch=epoch + 1)


if __name__ == "__main__": main()
