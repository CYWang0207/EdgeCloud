import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


DRIFT_SCHEDULES = {
    "none": [
        (0.00, 1.00, "normal", 0.0),
    ],
    "light": [
        (0.00, 0.20, "normal", 0.0),
        (0.20, 0.50, "bright", 1.0),
        (0.50, 0.80, "dark", 1.0),
        (0.80, 1.00, "normal", 0.0),
    ],
    "mixed": [
        (0.00, 0.20, "normal", 0.0),
        (0.20, 0.40, "bright", 1.0),
        (0.40, 0.60, "dark", 1.0),
        (0.60, 0.75, "blur", 1.0),
        (0.75, 0.90, "noise", 1.0),
        (0.90, 1.00, "occlusion", 1.0),
    ],
    "staged": [
        (0.00, 0.20, "normal", 0.0, 0.0),
        (0.20, 0.32, "bright", 0.40, 0.30),
        (0.32, 0.48, "normal", 0.0, 0.0),
        (0.48, 0.62, "blur", 0.55, 0.50),
        (0.62, 0.74, "normal", 0.0, 0.0),
        (0.74, 0.88, "noise", 0.70, 0.70),
        (0.88, 1.00, "normal", 0.0, 0.0),
    ],
    "highfreq": [
        (0.00, 0.05, "normal", 0.0, 0.0),
        (0.05, 0.10, "bright", 0.55, 0.40),
        (0.10, 0.15, "normal", 0.0, 0.0),
        (0.15, 0.20, "dark", 0.55, 0.40),
        (0.20, 0.25, "normal", 0.0, 0.0),
        (0.25, 0.30, "blur", 0.70, 0.65),
        (0.30, 0.35, "normal", 0.0, 0.0),
        (0.35, 0.40, "noise", 0.75, 0.75),
        (0.40, 0.45, "normal", 0.0, 0.0),
        (0.45, 0.50, "occlusion", 0.70, 0.80),
        (0.50, 0.55, "normal", 0.0, 0.0),
        (0.55, 0.60, "bright", 0.65, 0.45),
        (0.60, 0.65, "normal", 0.0, 0.0),
        (0.65, 0.70, "dark", 0.65, 0.45),
        (0.70, 0.75, "normal", 0.0, 0.0),
        (0.75, 0.80, "blur", 0.80, 0.75),
        (0.80, 0.85, "normal", 0.0, 0.0),
        (0.85, 0.90, "noise", 0.85, 0.85),
        (0.90, 0.95, "normal", 0.0, 0.0),
        (0.95, 1.00, "occlusion", 0.80, 0.90),
    ],
}

STRUCT_DRIFT_BY_TYPE = {
    "normal": 0.0,
    "bright": 0.1,
    "dark": 0.1,
    "blur": 0.5,
    "noise": 0.6,
    "occlusion": 0.8,
    "multi_view_failure": 1.0,
}

DRIFT_TYPE_TO_CONDITION = {
    "normal": 0,
    "bright": 1,
    "dark": 2,
    "blur": 3,
    "noise": 4,
    "occlusion": 5,
    "multi_view_failure": 6,
}


def condition_id_for_drift(drift_type):
    return DRIFT_TYPE_TO_CONDITION.get(str(drift_type), DRIFT_TYPE_TO_CONDITION["normal"])


def normalize_schedule_entry(entry):
    if len(entry) == 4:
        start, end, drift_type, severity = entry
        struct_override = None
    elif len(entry) == 5:
        start, end, drift_type, severity, struct_override = entry
    else:
        raise ValueError(f"Invalid drift schedule entry: {entry}")
    return start, end, drift_type, severity, struct_override


def drift_metadata_for_index(index, total, schedule_name):
    if schedule_name not in DRIFT_SCHEDULES:
        raise ValueError(f"Unknown drift schedule: {schedule_name}")

    if total <= 1:
        position = 0.0
    else:
        position = index / float(total)

    for entry in DRIFT_SCHEDULES[schedule_name]:
        start, end, drift_type, severity, struct_override = normalize_schedule_entry(entry)
        if start <= position < end:
            return drift_type, severity, struct_override

    _, _, drift_type, severity, struct_override = normalize_schedule_entry(
        DRIFT_SCHEDULES[schedule_name][-1]
    )
    return drift_type, severity, struct_override


def drift_spec_for_index(index, total, schedule_name):
    drift_type, severity, _struct_override = drift_metadata_for_index(index, total, schedule_name)
    return drift_type, severity


def structural_drift_score(drift_type, severity, override=None):
    if override is not None:
        return float(max(0.0, min(1.0, override)))
    base_score = STRUCT_DRIFT_BY_TYPE.get(drift_type, 0.0)
    return float(max(0.0, min(1.0, base_score * severity)))


def add_deterministic_noise(image, severity, seed):
    generator = torch.Generator(device=image.device)
    generator.manual_seed(seed)
    noise = torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype)
    return (image + noise * (0.12 * severity)).clamp(0.0, 1.0)


def add_deterministic_occlusion(image, severity, seed):
    generator = torch.Generator(device=image.device)
    generator.manual_seed(seed)

    _, height, width = image.shape
    box_ratio = 0.20 + 0.15 * severity
    box_h = max(1, int(height * box_ratio))
    box_w = max(1, int(width * box_ratio))

    max_y = max(1, height - box_h + 1)
    max_x = max(1, width - box_w + 1)
    y0 = int(torch.randint(0, max_y, (1,), generator=generator, device=image.device).item())
    x0 = int(torch.randint(0, max_x, (1,), generator=generator, device=image.device).item())

    occluded = image.clone()
    occluded[:, y0:y0 + box_h, x0:x0 + box_w] = 0.0
    return occluded


def apply_drift_to_image(image, drift_type, severity, seed):
    if drift_type == "normal" or severity <= 0:
        return image
    if drift_type == "bright":
        return TF.adjust_brightness(image, 1.0 + 0.5 * severity)
    if drift_type == "dark":
        return TF.adjust_brightness(image, max(0.1, 1.0 - 0.5 * severity))
    if drift_type == "blur":
        sigma = 0.1 + 2.0 * severity
        return TF.gaussian_blur(image, kernel_size=5, sigma=[sigma, sigma])
    if drift_type == "noise":
        return add_deterministic_noise(image, severity, seed)
    if drift_type == "occlusion":
        return add_deterministic_occlusion(image, severity, seed)

    raise ValueError(f"Unknown drift type: {drift_type}")


class DeterministicDriftWrapper(Dataset):
    def __init__(self, base_dataset, schedule="none", seed=123, normalize=True):
        self.base_dataset = base_dataset
        self.schedule = schedule
        self.seed = seed
        self.normalize = normalize
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        images, label = self.base_dataset[index]
        drift_type, severity, struct_override = drift_metadata_for_index(
            index,
            len(self.base_dataset),
            self.schedule,
        )

        drifted_images = []
        for view_idx, image in enumerate(images):
            view_seed = self.seed + index * 1009 + view_idx * 9176
            image = apply_drift_to_image(image, drift_type, severity, view_seed)
            if self.normalize:
                image = self.normalizer(image)
            drifted_images.append(image)

        images = torch.stack(drifted_images, dim=0)
        struct_drift = structural_drift_score(drift_type, severity, struct_override)
        return images, label, drift_type, float(severity), struct_drift
