"""Smoke-test the SEVD DataLoader and the existing multi-view ViT data path."""

import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import EarlyFusionMultiViewViT
from sevd_dataset import SEVD_CLASSES, SEVDMultiView


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="/root/autodl-tmp/SEVD")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--scene", action="append", dest="scenes")
    parser.add_argument("--batch-size", type=int, default=1)
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
    dataset_kwargs = {}
    if args.scenes:
        dataset_kwargs["scenes"] = tuple(args.scenes)
    dataset = SEVDMultiView(
        root_dir=args.dataset_path,
        split=args.split,
        transform=transform,
        **dataset_kwargs,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=dataset.num_views,
        num_classes=len(SEVD_CLASSES),
        pretrained=False,
    ).to(device)
    model.eval()

    views, targets, metadata = next(iter(loader))
    with torch.no_grad():
        logits = model(views.to(device))

    target_counts = [[len(target["labels"]) for target in sample] for sample in targets]
    print(f"dataset_samples={len(dataset)}")
    print(f"views_shape={tuple(views.shape)}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"objects_per_view={target_counts}")
    print(f"first_sample={metadata[0]}")
    print("SEVD DataLoader and MV-ViT forward data path: OK")


if __name__ == "__main__":
    main()
