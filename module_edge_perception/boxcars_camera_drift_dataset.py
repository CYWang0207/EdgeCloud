"""Calibrated camera-chain corruptions for paired BoxCars adapter experiments."""

import math

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset


DRIFTS = (
    "illumination", "motion_blur", "sensor_noise", "compression",
    "partial_occlusion", "haze", "glare", "lens_obstruction",
)
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _motion_kernel(size, angle, dtype):
    axis = torch.linspace(-(size - 1) / 2, (size - 1) / 2, size, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    radians = math.radians(angle)
    perpendicular = (-math.sin(radians) * xx + math.cos(radians) * yy).abs()
    longitudinal = math.cos(radians) * xx + math.sin(radians) * yy
    kernel = ((perpendicular < .55) & (longitudinal.abs() <= (size - 1) / 2)).to(dtype)
    return kernel / kernel.sum().clamp_min(1)


def apply_camera_corruption(views, view_mask, kind, severity, seed):
    """Apply one deterministic, track-consistent camera corruption to [V,C,H,W]."""
    severity = float(max(0., min(1., severity)))
    if kind == "normal" or severity == 0:
        return views
    if kind not in DRIFTS:
        raise ValueError(f"unknown BoxCars camera corruption: {kind}")
    generator = torch.Generator().manual_seed(int(seed))
    underexposed = bool(torch.randint(2, (), generator=generator))
    gamma = 1. + (2.25 * severity if underexposed else -.62 * severity)
    cast = torch.tensor([1 + .24 * severity, 1., 1 - .20 * severity], dtype=views.dtype).view(1, 3, 1, 1)
    angle = float(torch.rand((), generator=generator).item() * 130 - 65)
    obstruction_u, obstruction_v = torch.rand((), generator=generator).item(), torch.rand((), generator=generator).item()
    result = []
    for view_index, image in enumerate(views):
        local = torch.Generator().manual_seed(int(seed) + 9176 * view_index)
        if kind == "illumination":
            output = image.clamp(1e-4, 1).pow(gamma) * cast[0]
            h, w = image.shape[-2:]
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
            cx = -.35 + .7 * torch.rand((), generator=generator).item()
            cy = -.25 + .5 * torch.rand((), generator=generator).item()
            shade = 1 - .78 * severity * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / .30)
            output = output * shade.to(image.dtype)
        elif kind == "motion_blur":
            size = int(3 + 10 * severity) | 1
            kernel = _motion_kernel(size, angle, image.dtype).to(image.device)
            output = F.conv2d(image.unsqueeze(0), kernel.view(1, 1, size, size).expand(3, 1, size, size), padding=size // 2, groups=3).squeeze(0)
        elif kind == "sensor_noise":
            gain = 520. / (1 + 105 * severity * severity)
            shot = torch.poisson((image * gain).clamp_min(0), generator=local) / gain
            read = torch.randn(image.shape, generator=torch.Generator().manual_seed(int(seed) + 9176 * view_index + 1), dtype=image.dtype) * (.002 + .032 * severity)
            output = shot + read
        elif kind == "compression":
            scale = 1 - .52 * severity
            small = F.interpolate(image.unsqueeze(0), scale_factor=scale, mode="bilinear", align_corners=False)
            output = F.interpolate(small, size=image.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
            levels = max(12, int(256 - 220 * severity))
            output = torch.round(output * (levels - 1)) / (levels - 1)
        elif kind == "haze":
            air = torch.tensor([.78, .82, .85], dtype=image.dtype).view(3, 1, 1)
            transmission = 1 - .72 * severity
            output = image * transmission + air * (1 - transmission)
        elif kind == "glare":
            h, w = image.shape[-2:]
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
            cx = -.65 + 1.3 * torch.rand((), generator=generator).item()
            cy = -.55 + .7 * torch.rand((), generator=generator).item()
            bloom = torch.exp(-((xx - cx) ** 2 + 1.4 * (yy - cy) ** 2) / (.08 + .22 * (1 - severity)))
            output = image + (.48 + .42 * severity) * bloom.to(image.dtype).unsqueeze(0)
        else:
            output = image.clone()
            affected = view_index == int(torch.randint(len(views), (), generator=generator))
            if kind == "lens_obstruction":
                affected = bool(view_mask[view_index] > 0)
            if view_mask[view_index] > 0 and affected:
                h, w = image.shape[-2:]
                side_h = int(h * (.12 + (.42 if kind == "lens_obstruction" else .30) * severity))
                side_w = int(w * (.12 + (.38 if kind == "lens_obstruction" else .26) * severity))
                if kind == "lens_obstruction":
                    y, x = int(obstruction_v * max(1, h - side_h)), int(obstruction_u * max(1, w - side_w))
                else:
                    y, x = int(torch.randint(max(1, h - side_h), (), generator=local)), int(torch.randint(max(1, w - side_w), (), generator=local))
                fill = output.mean(dim=(-2, -1), keepdim=True)
                output[:, y:y + side_h, x:x + side_w] = (.35 * output[:, y:y + side_h, x:x + side_w] + .65 * fill if kind == "lens_obstruction" else fill)
        result.append(output.clamp(0, 1))
    return torch.stack(result)


class PairedBoxCarsCameraDrift(Dataset):
    """Aligned clean/corrupted tracks; sampling is deterministic by seed/index."""
    def __init__(self, base, drift_types, severity_min=.3, severity_max=.8,
                 clean_probability=.2, seed=123, drift_weights=None,
                 fixed_severities=None, fixed_drift=None, fixed_severity=None,
                 normalize=True, return_metadata=False):
        self.base, self.drift_types = base, tuple(drift_types)
        if not self.drift_types or set(self.drift_types) - set(DRIFTS):
            raise ValueError(f"drift_types must be selected from {DRIFTS}")
        self.severity_min, self.severity_max = float(severity_min), float(severity_max)
        self.clean_probability, self.seed = float(clean_probability), int(seed)
        self.fixed_severities = fixed_severities or {}
        self.fixed_drift, self.fixed_severity = fixed_drift, fixed_severity
        self.normalize, self.return_metadata = normalize, return_metadata
        self.drift_weights = torch.tensor(drift_weights or [1.] * len(self.drift_types), dtype=torch.float)
        if len(self.drift_weights) != len(self.drift_types) or (self.drift_weights < 0).any() or self.drift_weights.sum() <= 0:
            raise ValueError("drift_weights must be non-negative, match drift_types, and sum to > 0")

    def __len__(self): return len(self.base)
    @property
    def classes(self): return self.base.classes

    def __getitem__(self, index):
        clean, view_mask, label, metadata = self.base[index]
        if self.fixed_drift is not None:
            kind, severity = self.fixed_drift, self.fixed_severity
        else:
            generator = torch.Generator().manual_seed(self.seed + index * 104729)
            if torch.rand((), generator=generator).item() < self.clean_probability:
                kind, severity = "normal", 0.
            else:
                kind = self.drift_types[torch.multinomial(self.drift_weights, 1, generator=generator).item()]
                severity = self.fixed_severities.get(kind, self.severity_min + (self.severity_max - self.severity_min) * torch.rand((), generator=generator).item())
        corrupt = apply_camera_corruption(clean, view_mask, kind, severity, self.seed + index * 1009)
        output_clean, output_corrupt = (NORM(clean), NORM(corrupt)) if self.normalize else (clean, corrupt)
        if not self.return_metadata:
            return output_clean, output_corrupt, view_mask, label
        quality = torch.full_like(view_mask, 1. - float(severity), dtype=torch.float32) * view_mask
        return output_clean, output_corrupt, view_mask, label, metadata, kind, float(severity), quality, index
