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
