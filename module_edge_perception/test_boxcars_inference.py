"""Smoke-test BoxCars116k loading and the existing multi-view ViT data path."""

import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from boxcars_dataset import BoxCarsMultiView, VALID_TASKS
from model import EarlyFusionMultiViewViT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default="/root/autodl-tmp/EdgeCloud/data/BoxCars116k_kaggle/BoxCars116k",
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    parser.add_argument("--task", choices=VALID_TASKS, default="make")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model-name", default="vit_small_patch16_224")
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    dataset = BoxCarsMultiView(
        root_dir=args.dataset_path,
        split=args.split,
        task=args.task,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=dataset.num_views,
        num_classes=len(dataset.classes),
        pretrained=False,
    ).to(device)
    model.eval()

    views, view_mask, labels, metadata = next(iter(loader))
    with torch.no_grad():
        logits = model(views.to(device), view_mask=view_mask.to(device))

    print(f"dataset_samples={len(dataset)}")
    print(f"classes={len(dataset.classes)}")
    print(f"views_shape={tuple(views.shape)}")
    print(f"view_mask={view_mask.tolist()}")
    print(f"labels_shape={tuple(labels.shape)}")
    print(f"logits_shape={tuple(logits.shape)}")
    first_sample = dataset[0][3]
    print(f"first_sample={first_sample}")
    print("BoxCars116k DataLoader and MV-ViT forward data path: OK")


if __name__ == "__main__":
    main()
