"""Camera-plausible corruptions for BoxCars adapter calibration and training.

This module intentionally does not replace ``boxcars_drift_dataset.py``: the
older synthetic-drift experiments remain reproducible and their checkpoints
remain valid.  Here a sample is one traffic-camera track, so illumination and
codec settings are shared across its real views while sensor noise is sampled
per frame.
"""

import math

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset


DRIFTS = ("illumination", "motion_blur", "sensor_noise", "compression", "partial_occlusion", "haze", "glare", "lens_obstruction")
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
    """Apply one camera-chain corruption to a [V,C,H,W] BoxCars track."""
    severity = float(max(0., min(1., severity)))
    if kind == "normal" or severity == 0:
        return views
    if kind not in DRIFTS:
        raise ValueError(f"unknown BoxCars camera corruption: {kind}")
    generator = torch.Generator().manual_seed(int(seed))
    # Track-level capture settings: all frames came from one physical camera.
    underexposed = bool(torch.randint(2, (), generator=generator))
    # The high end represents a night / backlight failure, not a mild daytime
    # brightness tweak.  This needs a wider exposure response than ModelNet's
    # clean renderer before it has a measurable effect on surveillance crops.
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
            # A broad cast shadow / backlight falloff, not global brightness.
            cx = -.35 + .7 * torch.rand((), generator=generator).item()
            cy = -.25 + .5 * torch.rand((), generator=generator).item()
            shade = 1 - .78 * severity * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / .30)
            output = output * shade.to(image.dtype)
        elif kind == "motion_blur":
            size = int(3 + 10 * severity) | 1
            kernel = _motion_kernel(size, angle, image.dtype).to(image.device)
            output = F.conv2d(image.unsqueeze(0), kernel.view(1, 1, size, size).expand(3, 1, size, size), padding=size // 2, groups=3).squeeze(0)
        elif kind == "sensor_noise":
            # Low-light sensor chain: same gain setting, independent frame noise.
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
            # Atmospheric scattering: contrast/saturation loss towards a cool
            # air-light colour.  It models fog/smog, not a uniform gray veil.
            air = torch.tensor([.78, .82, .85], dtype=image.dtype).view(3, 1, 1)
            transmission = 1 - .72 * severity
            output = image * transmission + air * (1 - transmission)
        elif kind == "glare":
            # Low-angle sun / windshield reflection: a broad, off-centre bloom
            # that clips local details without obscuring the entire frame.
            h, w = image.shape[-2:]
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
            cx = -.65 + 1.3 * torch.rand((), generator=generator).item()
            cy = -.55 + .7 * torch.rand((), generator=generator).item()
            bloom = torch.exp(-((xx - cx) ** 2 + 1.4 * (yy - cy) ** 2) / (.08 + .22 * (1 - severity)))
            output = image + (.48 + .42 * severity) * bloom.to(image.dtype).unsqueeze(0)
        else:  # partial_occlusion/lens_obstruction: foreground or dirty optics.
            output = image.clone()
            affected = (view_index == int(torch.randint(len(views), (), generator=generator)))
            if kind == "lens_obstruction":
                # A wet/dirty lens persists through a track but moves slightly
                # with frame crop; it is translucent and soft-edged.
                affected = bool(view_mask[view_index] > 0)
            if view_mask[view_index] > 0 and affected:
                h, w = image.shape[-2:]
                side_h, side_w = int(h * (.12 + (.42 if kind == "lens_obstruction" else .30) * severity)), int(w * (.12 + (.38 if kind == "lens_obstruction" else .26) * severity))
                if kind == "lens_obstruction":
                    y, x = int(obstruction_v * max(1, h - side_h)), int(obstruction_u * max(1, w - side_w))
                else:
                    y = int(torch.randint(max(1, h - side_h), (), generator=local))
                    x = int(torch.randint(max(1, w - side_w), (), generator=local))
                fill = output.mean(dim=(-2, -1), keepdim=True)
                if kind == "lens_obstruction":
                    output[:, y:y + side_h, x:x + side_w] = .35 * output[:, y:y + side_h, x:x + side_w] + .65 * fill
                else:
                    output[:, y:y + side_h, x:x + side_w] = fill
        result.append(output.clamp(0, 1))
    return torch.stack(result)


class PairedBoxCarsCameraDrift(Dataset):
    """Aligned clean/corrupted traffic-camera tracks with deterministic sampling."""
    def __init__(self, base, drift_types, severity_min=.3, severity_max=.8,
                 clean_probability=.2, seed=123, drift_weights=None,
                 fixed_severities=None, fixed_drift=None, fixed_severity=None):
        self.base, self.drift_types = base, tuple(drift_types)
        if not self.drift_types or set(self.drift_types) - set(DRIFTS):
            raise ValueError(f"drift_types must be selected from {DRIFTS}")
        self.severity_min, self.severity_max = severity_min, severity_max
        self.clean_probability, self.seed = clean_probability, int(seed)
        self.fixed_severities = fixed_severities or {}
        self.fixed_drift, self.fixed_severity = fixed_drift, fixed_severity
        self.drift_weights = torch.tensor(drift_weights or [1.] * len(self.drift_types), dtype=torch.float)
        if len(self.drift_weights) != len(self.drift_types) or self.drift_weights.sum() <= 0:
            raise ValueError("drift_weights must match drift_types and sum to > 0")

    def __len__(self): return len(self.base)
    @property
    def classes(self): return self.base.classes

    def __getitem__(self, index):
        clean, view_mask, label, _metadata = self.base[index]
        if self.fixed_drift is not None:
            kind, severity = self.fixed_drift, self.fixed_severity
        else:
            g = torch.Generator().manual_seed(self.seed + index * 104729)
            if torch.rand((), generator=g).item() < self.clean_probability:
                kind, severity = "normal", 0.
            else:
                kind = self.drift_types[torch.multinomial(self.drift_weights, 1, generator=g).item()]
                severity = self.fixed_severities.get(kind, self.severity_min + (self.severity_max - self.severity_min) * torch.rand((), generator=g).item())
        corrupt = apply_camera_corruption(clean, view_mask, kind, severity, self.seed + index * 1009)
        return NORM(clean), NORM(corrupt), view_mask, label
