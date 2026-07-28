import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from dataset import ModelNet40MultiView
from model import EarlyFusionMultiViewViT
from prompt_tuning.prompt_model import PromptGenerator


ROOT_DIR = Path(__file__).resolve().parent
EDGE_RL_DIR = ROOT_DIR / "EdgeCloud_RL"
sys.path.insert(0, str(EDGE_RL_DIR))

from drift_dataset import (  # noqa: E402
    DeterministicDriftWrapper,
    condition_id_for_drift,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints directly on the training split.")
    parser.add_argument("--dataset-path", type=str, default="./data/modelnet40v2png_ori4")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/mv_vit_token_epoch_19.pth")
    parser.add_argument("--prompt-checkpoint", type=str, default="")
    parser.add_argument("--retrain-checkpoint", type=str, default="")
    parser.add_argument("--model-name", type=str, default="vit_small_patch16_224")
    parser.add_argument("--num-classes", type=int, default=40)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-prompt-tokens", type=int, default=4)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--drift-schedule", type=str, default="mixed", choices=["none", "light", "mixed", "staged", "highfreq"])
    parser.add_argument("--drift-seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--token-keep-ratio",
        type=float,
        default=1.0,
        help="Use < 1.0 to evaluate deterministic token pruning on the train split.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def build_loader(args):
    if args.drift_schedule == "none":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    dataset = ModelNet40MultiView(
        root_dir=args.dataset_path,
        split=args.split,
        transform=transform,
        num_views=args.num_views,
    )

    if args.drift_schedule != "none":
        dataset = DeterministicDriftWrapper(
            dataset,
            schedule=args.drift_schedule,
            seed=args.drift_seed,
            normalize=True,
        )

    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_model(args, checkpoint_path, device):
    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=args.num_views,
        num_classes=args.num_classes,
        pretrained=False,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"权重兼容加载: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


def load_prompt_bundle(args, model, prompt_path, device):
    checkpoint = torch.load(prompt_path, map_location=device)
    embed_dim = model.cls_token.shape[-1]
    num_conditions = int(checkpoint.get("num_conditions", 7)) if isinstance(checkpoint, dict) else 7
    num_prompt_tokens = int(checkpoint.get("num_prompt_tokens", args.num_prompt_tokens)) if isinstance(checkpoint, dict) else args.num_prompt_tokens

    prompt_gen = PromptGenerator(
        vit_embed_dim=embed_dim,
        num_prompt_tokens=num_prompt_tokens,
        num_conditions=num_conditions,
    ).to(device)

    if isinstance(checkpoint, dict) and "prompt_gen" in checkpoint:
        prompt_gen.load_state_dict(checkpoint["prompt_gen"])
        if "vit_norm" in checkpoint:
            model.norm.load_state_dict(checkpoint["vit_norm"], strict=False)
        if "vit_head" in checkpoint:
            model.head.load_state_dict(checkpoint["vit_head"], strict=False)
    else:
        prompt_gen.load_state_dict(checkpoint)

    prompt_gen.eval()
    model.eval()
    return prompt_gen


def drift_types_to_condition_ids(drift_types, device):
    ids = [condition_id_for_drift(drift_type) for drift_type in drift_types]
    return torch.tensor(ids, dtype=torch.long, device=device)


@torch.no_grad()
def evaluate(model, loader, device, name, args, prompt_gen=None):
    correct = 0
    total = 0
    by_drift = {}

    for batch in tqdm(loader, desc=f"Evaluating {name}"):
        if len(batch) == 5:
            images, labels, drift_types, _severities, _struct_drifts = batch
        elif len(batch) == 4:
            images, labels, drift_types, _severities = batch
        else:
            images, labels = batch
            drift_types = ["normal"] * labels.shape[0]

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        keep_ratios = None
        if args.token_keep_ratio < 1.0:
            keep_ratios = torch.full(
                (labels.shape[0], args.num_views),
                args.token_keep_ratio,
                dtype=torch.float32,
                device=device,
            )

        prompt_tokens = None
        if prompt_gen is not None:
            condition_ids = drift_types_to_condition_ids(drift_types, device)
            prompt_tokens = prompt_gen(condition_ids)

        outputs = model(
            images,
            keep_ratios=keep_ratios,
            token_score_mode="importance",
            prompt_tokens=prompt_tokens,
        )
        predicted = outputs.argmax(dim=1)
        matches = predicted == labels

        correct += matches.sum().item()
        total += labels.size(0)

        for idx, drift_type in enumerate(drift_types):
            drift_type = str(drift_type)
            if drift_type not in by_drift:
                by_drift[drift_type] = [0, 0]
            by_drift[drift_type][0] += int(matches[idx].item())
            by_drift[drift_type][1] += 1

    acc = correct / max(total, 1)
    print("-" * 72)
    print(f"{name}: {acc:.4%} ({correct}/{total})")
    for drift_type in sorted(by_drift.keys()):
        drift_correct, drift_total = by_drift[drift_type]
        print(f"  {drift_type}: {drift_correct / max(drift_total, 1):.4%} ({drift_correct}/{drift_total})")
    return acc


def main():
    args = parse_args()
    device = torch.device(args.device)
    loader = build_loader(args)

    base_model = load_model(args, args.checkpoint, device)
    evaluate(base_model, loader, device, "Base-NoPrompt", args)

    if args.prompt_checkpoint:
        prompt_model = load_model(args, args.checkpoint, device)
        prompt_gen = load_prompt_bundle(args, prompt_model, args.prompt_checkpoint, device)
        evaluate(prompt_model, loader, device, "Base-WithPrompt", args, prompt_gen=prompt_gen)

    if args.retrain_checkpoint:
        retrain_model = load_model(args, args.retrain_checkpoint, device)
        evaluate(retrain_model, loader, device, "Retrain-NoPrompt", args)

        if args.prompt_checkpoint:
            retrain_prompt_model = load_model(args, args.retrain_checkpoint, device)
            retrain_prompt_gen = load_prompt_bundle(args, retrain_prompt_model, args.prompt_checkpoint, device)
            evaluate(retrain_prompt_model, loader, device, "Retrain-WithPrompt", args, prompt_gen=retrain_prompt_gen)


if __name__ == "__main__":
    main()
