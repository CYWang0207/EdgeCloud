"""Calibrated camera/render corruptions shared by ModelNet40 Adapter ablations."""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

DRIFTS = ("illumination", "defocus", "sensor_noise", "compression", "partial_occlusion")
NORM = transforms.Normalize([.485, .456, .406], [.229, .224, .225])


def apply_camera_corruption(views, kind, severity, seed):
    severity = float(max(0., min(1., severity)))
    if kind == "normal" or severity == 0:
        return views
    if kind not in DRIFTS:
        raise ValueError(f"unknown ModelNet camera corruption: {kind}")
    generator = torch.Generator().manual_seed(int(seed))
    result = []
    for view_index, image in enumerate(views):
        local_seed = int(seed) + view_index * 9176
        if kind == "illumination":
            underexposed = bool(torch.randint(2, (), generator=generator))
            gamma = 1. + (1.65 * severity if underexposed else -.65 * severity)
            output = image.clamp(1e-4, 1).pow(gamma)
            cast = torch.tensor([1 + .22 * severity, 1., 1 - .18 * severity], dtype=image.dtype).view(3, 1, 1)
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, image.shape[1]), torch.linspace(-1, 1, image.shape[2]), indexing="ij")
            center_x = -.35 + .7 * torch.rand((), generator=generator).item()
            center_y = -.35 + .7 * torch.rand((), generator=generator).item()
            shadow = 1 - .75 * severity * torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / .18)
            output = output * cast * shadow.to(image.dtype)
        elif kind == "defocus":
            sigma = .25 + 1.45 * severity
            output = TF.gaussian_blur(image, kernel_size=5, sigma=[sigma, sigma])
        elif kind == "sensor_noise":
            gain = 500. / (1 + 24 * severity * severity)
            shot = torch.poisson((image * gain).clamp_min(0), generator=torch.Generator().manual_seed(local_seed)) / gain
            read = torch.randn(image.shape, generator=torch.Generator().manual_seed(local_seed + 1), dtype=image.dtype) * (.002 + .020 * severity)
            output = shot + read
        elif kind == "compression":
            small = F.interpolate(image.unsqueeze(0), scale_factor=1 - .45 * severity, mode="bilinear", align_corners=False)
            output = F.interpolate(small, size=image.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
            levels = int(256 - 208 * severity)
            output = torch.round(output * (levels - 1)) / (levels - 1)
        else:
            output = image.clone()
            if view_index == int(torch.randint(len(views), (), generator=generator)):
                h, w = image.shape[-2:]; side = int(min(h, w) * (.12 + .28 * severity))
                y = int(torch.randint(max(1, h - side), (), generator=generator)); x = int(torch.randint(max(1, w - side), (), generator=generator))
                output[:, y:y + side, x:x + side] = output.mean(dim=(-2, -1), keepdim=True)
        result.append(output.clamp(0, 1))
    return torch.stack(result)


class PairedModelNetDrift(Dataset):
    """Deterministic aligned ModelNet40 clean/camera-corrupted pairs."""
    def __init__(self, base, drift_types, severity_min=.4, severity_max=.8,
                 clean_probability=.2, seed=42, drift_weights=None,
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
        clean, label = self.base[index]
        if self.fixed_drift is not None:
            kind, severity = self.fixed_drift, self.fixed_severity
        else:
            generator = torch.Generator().manual_seed(self.seed + index * 104729)
            if torch.rand((), generator=generator).item() < self.clean_probability:
                kind, severity = "normal", 0.
            else:
                kind = self.drift_types[torch.multinomial(self.drift_weights, 1, generator=generator).item()]
                severity = self.fixed_severities.get(kind, self.severity_min + (self.severity_max - self.severity_min) * torch.rand((), generator=generator).item())
        corrupt = apply_camera_corruption(clean, kind, severity, self.seed + index * 1009)
        output_clean, output_corrupt = (torch.stack([NORM(image) for image in clean]), torch.stack([NORM(image) for image in corrupt])) if self.normalize else (clean, corrupt)
        if not self.return_metadata:
            return output_clean, output_corrupt, label
        return output_clean, output_corrupt, label, kind, float(severity), index
