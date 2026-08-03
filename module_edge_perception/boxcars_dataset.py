"""BoxCars116k multi-view classification dataset for application scene 2."""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


VALID_SPLITS = ("train", "validation", "test")
VALID_TASKS = ("make", "body", "medium", "hard")


class BoxCarsMultiView(Dataset):
    """Load up to four observations from one BoxCars116k vehicle track.

    BoxCars116k does not provide cross-camera identity.  A sample therefore
    consists of temporally separated crops from one vehicle track recorded by
    one physical traffic camera, not four cameras observing the same vehicle.
    Tracks shorter than ``num_views`` are padded by repeating the last image;
    ``view_mask`` marks those padded slots as unavailable.

    Each item is ``(views, view_mask, label, metadata)``:

    - ``views``: ``[num_views, C, H, W]``
    - ``view_mask``: ``[num_views]``, 1 for a real observation and 0 for padding
    - ``label``: contiguous class index from the official classification split
    - ``metadata``: vehicle, camera, annotation and selected image paths
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        task: str = "make",
        num_views: int = 4,
        transform: Optional[Callable] = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got {task!r}")
        if num_views < 1:
            raise ValueError("num_views must be at least 1")

        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.split = split
        self.task = task
        self.num_views = num_views
        self.transform = transform

        dataset_path = os.path.join(self.root_dir, "json_data", "dataset.json")
        splits_path = os.path.join(
            self.root_dir, "json_data", "classification_splits.json"
        )
        if not os.path.isfile(dataset_path) or not os.path.isfile(splits_path):
            raise FileNotFoundError(
                "BoxCars116k root must contain json_data/dataset.json and "
                f"json_data/classification_splits.json: {self.root_dir}"
            )

        with open(dataset_path, "r", encoding="utf-8") as handle:
            dataset = json.load(handle)
        with open(splits_path, "r", encoding="utf-8") as handle:
            classification_splits = json.load(handle)

        self.vehicles: Sequence[dict] = dataset["samples"]
        task_split = classification_splits[task]
        self.samples: List[Tuple[int, int]] = [
            (int(vehicle_id), int(label)) for vehicle_id, label in task_split[split]
        ]
        mapping: Dict[str, int] = {
            name: int(index) for name, index in task_split["types_mapping"].items()
        }
        self.class_to_idx = mapping
        self.classes = [""] * len(mapping)
        for name, index in mapping.items():
            self.classes[index] = name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        vehicle_id, label = self.samples[index]
        vehicle = self.vehicles[vehicle_id]
        instances = vehicle["instances"]
        if not instances:
            raise RuntimeError(f"vehicle {vehicle_id} contains no images")

        selected_indices = _evenly_spaced_indices(len(instances), self.num_views)
        real_view_count = min(len(instances), self.num_views)
        view_mask = torch.zeros(self.num_views, dtype=torch.float32)
        view_mask[:real_view_count] = 1.0

        views = []
        relative_paths = []
        for instance_index in selected_indices:
            relative_path = instances[instance_index]["path"]
            image_path = os.path.join(self.root_dir, "images", relative_path)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"missing BoxCars116k image: {image_path}")
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                view = self.transform(image) if self.transform else _pil_to_tensor(image)
            if not isinstance(view, torch.Tensor):
                raise TypeError("transform must return a torch.Tensor")
            views.append(view)
            relative_paths.append(relative_path)

        metadata = {
            "vehicle_id": vehicle_id,
            "camera": str(vehicle["camera"]),
            "annotation": vehicle["annotation"],
            "to_camera": bool(vehicle.get("to_camera", False)),
            "image_paths": relative_paths,
        }
        return torch.stack(views), view_mask, label, metadata


def _evenly_spaced_indices(length: int, count: int) -> List[int]:
    """Select the widest temporal baseline, padding short tracks at the end."""
    if length >= count:
        if count == 1:
            return [length // 2]
        return [round(i * (length - 1) / (count - 1)) for i in range(count)]
    return list(range(length)) + [length - 1] * (count - length)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(
        image.size[1], image.size[0], len(image.getbands())
    )
    return tensor.permute(2, 0, 1).float().div(255.0)
