"""Drift adapter for BoxCars116k without changing the shared scene-1 wrapper."""

import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


COMMON_DIR = Path(__file__).resolve().parent.parent / "common"
sys.path.insert(0, str(COMMON_DIR))

from drift_dataset import (  # noqa: E402
    apply_drift_to_image,
    drift_metadata_for_index,
    structural_drift_score,
)


class BoxCarsDriftWrapper(Dataset):
    """Apply deterministic visual drift while retaining BoxCars view masks."""

    def __init__(self, base_dataset, schedule="mixed", seed=123, normalize=True):
        self.base_dataset = base_dataset
        self.schedule = schedule
        self.seed = seed
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ) if normalize else None

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        images, view_mask, label, metadata = self.base_dataset[index]
        drift_type, severity, struct_override = drift_metadata_for_index(
            index, len(self.base_dataset), self.schedule
        )
        drifted = []
        for view_index, image in enumerate(images):
            image = apply_drift_to_image(
                image,
                drift_type,
                severity,
                self.seed + index * 1009 + view_index * 9176,
            )
            if self.normalizer is not None:
                image = self.normalizer(image)
            drifted.append(image)
        struct_drift = structural_drift_score(
            drift_type, severity, struct_override
        )
        return (
            torch.stack(drifted),
            view_mask,
            label,
            metadata,
            drift_type,
            float(severity),
            struct_drift,
        )


SUPPORTED_TRAIN_DRIFTS = ("bright", "dark", "blur", "noise", "occlusion")


class PairedBoxCarsDriftDataset(Dataset):
    """Return an aligned clean/drifted pair for adapter training.

    Unlike :class:`BoxCarsDriftWrapper`, drift is sampled per sample rather than
    assigned by a time schedule.  A time schedule is appropriate for online
    evaluation, but it confounds drift type with dataset order during training.

    The wrapped ``base_dataset`` must return unnormalised tensors in ``[0, 1]``.
    Sampling is deterministic for a given ``seed`` and index, which keeps DDP
    workers reproducible and makes clean/drift feature alignment exact.
    """

    def __init__(
        self,
        base_dataset,
        drift_types=SUPPORTED_TRAIN_DRIFTS,
        severity_min=0.35,
        severity_max=1.0,
        clean_probability=0.15,
        independent_view_drifts=False,
        seed=123,
        normalize=True,
    ):
        self.base_dataset = base_dataset
        self.drift_types = tuple(drift_types)
        unknown = sorted(set(self.drift_types) - set(SUPPORTED_TRAIN_DRIFTS))
        if unknown:
            raise ValueError(f"unsupported training drift types: {unknown}")
        if not self.drift_types:
            raise ValueError("drift_types must not be empty")
        if not 0.0 <= severity_min <= severity_max <= 1.0:
            raise ValueError("severity range must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= clean_probability < 1.0:
            raise ValueError("clean_probability must be in [0, 1)")
        self.severity_min = float(severity_min)
        self.severity_max = float(severity_max)
        self.clean_probability = float(clean_probability)
        self.independent_view_drifts = bool(independent_view_drifts)
        self.seed = int(seed)
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ) if normalize else None

    def __len__(self):
        return len(self.base_dataset)

    @property
    def classes(self):
        return self.base_dataset.classes

    def _sample_spec(self, index):
        generator = torch.Generator().manual_seed(self.seed + index * 104729)
        if torch.rand((), generator=generator).item() < self.clean_probability:
            return "normal", 0.0
        type_index = int(torch.randint(
            len(self.drift_types), (1,), generator=generator
        ).item())
        unit = torch.rand((), generator=generator).item()
        severity = self.severity_min + unit * (
            self.severity_max - self.severity_min
        )
        return self.drift_types[type_index], float(severity)

    def _sample_view_spec(self, index, view_index):
        if not self.independent_view_drifts:
            return self._sample_spec(index)
        return self._sample_spec(index * 31 + view_index * 1049)

    def __getitem__(self, index):
        images, view_mask, label, metadata = self.base_dataset[index]
        clean, drifted, quality_targets = [], [], []
        drift_types, severities = [], []
        for view_index, image in enumerate(images):
            drift_type, severity = self._sample_view_spec(index, view_index)
            drift_types.append(drift_type)
            severities.append(severity)
            clean_image = image
            drifted_image = apply_drift_to_image(
                image,
                drift_type,
                severity,
                self.seed + index * 1009 + view_index * 9176,
            )
            if self.normalizer is not None:
                clean_image = self.normalizer(clean_image)
                drifted_image = self.normalizer(drifted_image)
            clean.append(clean_image)
            drifted.append(drifted_image)
            quality_targets.append(1.0 - severity)
        drift_description = ",".join(drift_types)
        mean_severity = sum(severities) / len(severities)
        return (
            torch.stack(clean),
            torch.stack(drifted),
            view_mask,
            label,
            metadata,
            drift_description,
            mean_severity,
            torch.tensor(quality_targets, dtype=torch.float32) * view_mask,
            index,
        )
