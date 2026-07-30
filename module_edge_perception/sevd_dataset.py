"""SEVD four-camera RGB dataset for the second application scene."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


SEVD_CLASSES = ("car", "truck", "van", "pedestrian", "motorcycle", "bicycle")
DEFAULT_SCENES = ("scene_002", "scene_003", "scene_030", "scene_032")
DEFAULT_CAMERAS = ("camera_01", "camera_02", "camera_03", "camera_04")
SPLIT_RANGES = {
    "train": (0.00, 0.70),
    "val": (0.70, 0.85),
    "test": (0.85, 1.00),
    "all": (0.00, 1.00),
}


class SEVDMultiView(Dataset):
    """Load one synchronized timestamp from several fixed SEVD cameras.

    Each item is ``(views, targets, metadata)`` where ``views`` has shape
    ``[num_views, C, H, W]``. ``targets`` contains one COCO-style target dict
    per view. Splits are chronological within every scene so synchronized
    views from one timestamp can never leak across splits.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        scenes: Sequence[str] = DEFAULT_SCENES,
        cameras: Sequence[str] = DEFAULT_CAMERAS,
        strict: bool = True,
    ) -> None:
        if split not in SPLIT_RANGES:
            raise ValueError(f"split must be one of {tuple(SPLIT_RANGES)}, got {split!r}")
        if not cameras:
            raise ValueError("at least one camera is required")

        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.split = split
        self.transform = transform
        self.scenes = tuple(scenes)
        self.cameras = tuple(cameras)
        self.num_views = len(self.cameras)
        self.strict = strict
        self.samples: List[Tuple[str, str]] = []
        self._annotations: Dict[Tuple[str, str], Dict[str, List[dict]]] = {}
        self._scene_metadata: Dict[str, dict] = {}

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"SEVD root directory does not exist: {self.root_dir}")

        for scene in self.scenes:
            self._index_scene(scene)

        if not self.samples:
            raise RuntimeError(
                f"no synchronized SEVD samples found under {self.root_dir} "
                f"for scenes={self.scenes}, split={split}"
            )

    def _organized_dir(self, scene: str) -> str:
        return os.path.join(self.root_dir, scene, "organized")

    def _index_scene(self, scene: str) -> None:
        organized_dir = self._organized_dir(scene)
        if not os.path.isdir(organized_dir):
            if self.strict:
                raise FileNotFoundError(f"missing organized scene directory: {organized_dir}")
            return

        metadata_path = os.path.join(organized_dir, "metadata.json")
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                self._scene_metadata[scene] = json.load(handle)
        else:
            self._scene_metadata[scene] = {}

        frame_sets = []
        for camera in self.cameras:
            image_dir = os.path.join(organized_dir, camera, "images")
            coco_path = os.path.join(organized_dir, camera, "annotations", "coco.json")
            if not os.path.isdir(image_dir) or not os.path.isfile(coco_path):
                raise FileNotFoundError(f"missing images or COCO annotation for {scene}/{camera}")

            with open(coco_path, "r", encoding="utf-8") as handle:
                coco = json.load(handle)
            image_id_to_name = {item["id"]: item["file_name"] for item in coco["images"]}
            by_file: Dict[str, List[dict]] = defaultdict(list)
            for annotation in coco.get("annotations", []):
                file_name = image_id_to_name.get(annotation["image_id"])
                if file_name is not None:
                    by_file[file_name].append(annotation)
            self._annotations[(scene, camera)] = by_file
            frame_sets.append(set(image_id_to_name.values()))

        common_frames = set.intersection(*frame_sets)
        if self.strict and any(frames != common_frames for frames in frame_sets):
            raise RuntimeError(f"camera frame sets are not synchronized in {scene}")

        ordered_frames = sorted(common_frames, key=_natural_frame_key)
        start_ratio, end_ratio = SPLIT_RANGES[self.split]
        start = int(len(ordered_frames) * start_ratio)
        end = int(len(ordered_frames) * end_ratio)
        self.samples.extend((scene, frame_name) for frame_name in ordered_frames[start:end])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        scene, frame_name = self.samples[index]
        views = []
        targets = []

        for camera in self.cameras:
            image_path = os.path.join(
                self._organized_dir(scene), camera, "images", frame_name
            )
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            view = self.transform(image) if self.transform is not None else _pil_to_tensor(image)
            if not isinstance(view, torch.Tensor):
                raise TypeError("transform must return a torch.Tensor")
            views.append(view)

            boxes = []
            labels = []
            areas = []
            crowd = []
            for annotation in self._annotations[(scene, camera)].get(frame_name, []):
                x, y, box_width, box_height = annotation["bbox"]
                boxes.append([x, y, x + box_width, y + box_height])
                # Keep the official COCO IDs (1-6); 0 remains available for background.
                labels.append(int(annotation["category_id"]))
                areas.append(float(annotation.get("area", box_width * box_height)))
                crowd.append(int(annotation.get("iscrowd", 0)))

            targets.append(
                {
                    "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                    "labels": torch.tensor(labels, dtype=torch.int64),
                    "area": torch.tensor(areas, dtype=torch.float32),
                    "iscrowd": torch.tensor(crowd, dtype=torch.int64),
                    "image_size": torch.tensor([height, width], dtype=torch.int64),
                    "camera": camera,
                }
            )

        metadata = dict(self._scene_metadata.get(scene, {}))
        metadata.update({"scene": scene, "frame_id": os.path.splitext(frame_name)[0]})
        return torch.stack(views, dim=0), targets, metadata

    @staticmethod
    def collate_fn(batch):
        """Collate images while preserving variable-length detection targets."""
        views, targets, metadata = zip(*batch)
        return torch.stack(views, dim=0), list(targets), list(metadata)


def _natural_frame_key(file_name: str):
    stem = os.path.splitext(os.path.basename(file_name))[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(
        image.size[1], image.size[0], len(image.getbands())
    )
    return tensor.permute(2, 0, 1).float().div(255.0)
