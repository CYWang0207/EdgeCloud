import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent


def find_mv_vit_dir():
    search_roots = [SCRIPT_DIR, *SCRIPT_DIR.parents]
    candidates = []
    for root in search_roots:
        candidates.append(root)
        candidates.append(root / "MV-VIT")

    for candidate in candidates:
        if (candidate / "dataset.py").exists() and (candidate / "model.py").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate MV-VIT root. Expected dataset.py and model.py in a parent directory."
    )


MV_VIT_DIR = find_mv_vit_dir()
sys.path.insert(0, str(MV_VIT_DIR))

from dataset import ModelNet40MultiView  # noqa: E402
from model import EarlyFusionMultiViewViT  # noqa: E402
from prompt_tuning.prompt_model import PromptGenerator  # noqa: E402
from drift_dataset import (  # noqa: E402
    DeterministicDriftWrapper,
    condition_id_for_drift,
)


def parse_args():
    default_dataset = MV_VIT_DIR / "data" / "modelnet40v2png_ori4"
    default_checkpoint = MV_VIT_DIR / "checkpoints" / "mv_vit_base_epoch_30.pth"
    default_policy = SCRIPT_DIR / "rl_decision_log.csv"
    default_output = SCRIPT_DIR / "mvvit_rl_policy_eval.csv"

    parser = argparse.ArgumentParser(
        description="Evaluate RL view/token decisions on real MV-VIT inference."
    )
    parser.add_argument("--dataset-path", type=Path, default=default_dataset)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--prompt-checkpoint", type=Path, default=None)
    parser.add_argument("--retrain-checkpoint", type=Path, default=None)
    parser.add_argument("--policy-log", type=Path, default=default_policy)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--num-classes", type=int, default=40)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-prompt-tokens", type=int, default=4)
    parser.add_argument("--tokens-per-view", type=int, default=196)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--drift-schedule",
        type=str,
        default="none",
        choices=["none", "light", "mixed", "staged", "highfreq"],
        help="Use the same deterministic drift schedule as trajectory generation.",
    )
    parser.add_argument("--drift-seed", type=int, default=123)
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="token_prune",
        choices=["token_prune", "view_mask"],
        help="token_prune keeps all views but replaces dropped patch tokens; view_mask blanks inactive views.",
    )
    parser.add_argument(
        "--prompt-for-u",
        type=str,
        default="u1",
        choices=["none", "u1", "u1_u2"],
        help="When to inject prompt tokens if --prompt-checkpoint is provided.",
    )
    parser.add_argument(
        "--inactive-view-keep",
        type=float,
        default=0.1,
        help="Effective keep ratio for v_i=0 when eval-mode=token_prune.",
    )
    parser.add_argument(
        "--min-keep",
        type=float,
        default=0.0,
        help="Optional lower bound for all token keep ratios.",
    )
    parser.add_argument(
        "--drop-token-mode",
        type=str,
        default="learned",
        choices=["learned", "mean", "zero"],
        help="Replacement for pruned patch tokens in token_prune mode.",
    )
    parser.add_argument(
        "--blank-mode",
        type=str,
        default="zero",
        choices=["zero", "black"],
        help="Used only by view_mask mode. zero means normalized mean-color view.",
    )
    parser.add_argument(
        "--missing-policy",
        type=str,
        default="error",
        choices=["error", "all_views"],
        help="What to do when a sample_id is missing in the RL decision log.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def resolve_checkpoint(path):
    if path.exists():
        return path

    candidates = [
        MV_VIT_DIR / "checkpoints" / "mv_vit_base_epoch_30.pth",
        MV_VIT_DIR / "checkpoints" / "mv_vit_epoch_30.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            print(f"Default checkpoint not found, using: {candidate}")
            return candidate

    raise FileNotFoundError(
        "No checkpoint found. Pass --checkpoint with the trained MV-VIT .pth file."
    )


def load_policy(path, num_views):
    if not path.exists():
        raise FileNotFoundError(f"Policy log not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Policy log is empty: {path}")

    required_v = [f"v_{i + 1}" for i in range(num_views)]
    missing = [col for col in required_v if col not in rows[0]]
    if missing:
        raise ValueError(f"Policy log is missing required view columns: {missing}")

    has_k = all(f"k_{i + 1}" in rows[0] for i in range(num_views))
    has_u = "u" in rows[0]
    policy = {}
    for row_idx, row in enumerate(rows):
        sample_id = row.get("sample_id", "")
        key = int(float(sample_id)) if sample_id != "" else row_idx
        view_mask = np.array(
            [int(float(row[f"v_{i + 1}"])) for i in range(num_views)],
            dtype=np.int64,
        )
        if has_k:
            keep_ratios = np.array(
                [float(row[f"k_{i + 1}"]) for i in range(num_views)],
                dtype=np.float32,
            )
        else:
            keep_ratios = view_mask.astype(np.float32)

        policy[key] = {
            "view_mask": view_mask,
            "keep_ratios": keep_ratios,
            "u": int(float(row["u"])) if has_u and row["u"] != "" else 0,
        }

    return policy


def effective_keep_ratios(view_mask, keep_ratios, args):
    keep = keep_ratios.astype(np.float32).copy()
    keep = np.where(view_mask == 1, keep, args.inactive_view_keep)
    keep = np.maximum(keep, args.min_keep)
    return np.clip(keep, 0.0, 1.0)


def build_loader(args):
    if args.drift_schedule == "none":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    dataset = ModelNet40MultiView(
        root_dir=str(args.dataset_path),
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
        max_samples = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, range(max_samples))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and args.device.startswith("cuda"),
    )
    return loader


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


def load_prompt_bundle(args, model, device):
    if args.prompt_checkpoint is None:
        return None
    if not args.prompt_checkpoint.exists():
        raise FileNotFoundError(f"Prompt checkpoint not found: {args.prompt_checkpoint}")

    embed_dim = model.cls_token.shape[-1]
    checkpoint = torch.load(args.prompt_checkpoint, map_location=device)
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


def load_retrain_model(args, device):
    if args.retrain_checkpoint is None:
        return None
    retrain_model = load_model(args, args.retrain_checkpoint, device)
    retrain_model.eval()
    return retrain_model


def should_use_prompt(u_values, args):
    if args.prompt_for_u == "none" or args.prompt_checkpoint is None:
        return torch.zeros_like(u_values, dtype=torch.bool)
    if args.prompt_for_u == "u1":
        return u_values == 1
    return (u_values == 1) | (u_values == 2)


def drift_types_to_condition_ids(drift_types, device):
    ids = [condition_id_for_drift(drift_type) for drift_type in drift_types]
    return torch.tensor(ids, dtype=torch.long, device=device)


def make_blank_view(images, mode):
    if mode == "zero":
        return torch.zeros_like(images[:, 0])

    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (torch.zeros_like(images[:, 0]) - mean) / std


def apply_view_policy(images, view_masks, blank_mode):
    masked_images = images.clone()
    blank_view = make_blank_view(images, blank_mode)

    for view_idx in range(view_masks.shape[1]):
        inactive = view_masks[:, view_idx] == 0
        if torch.any(inactive):
            masked_images[inactive, view_idx] = blank_view[inactive]

    return masked_images


def build_token_keep_mask(token_features, keep_ratios):
    batch_size, num_views, num_patches, _ = token_features.shape
    importance = token_features.norm(dim=-1)
    keep_counts = torch.ceil(keep_ratios * num_patches).long().clamp(1, num_patches)
    keep_mask = torch.zeros(
        batch_size,
        num_views,
        num_patches,
        dtype=torch.bool,
        device=token_features.device,
    )

    for batch_idx in range(batch_size):
        for view_idx in range(num_views):
            keep_count = int(keep_counts[batch_idx, view_idx].item())
            top_idx = torch.topk(importance[batch_idx, view_idx], keep_count).indices
            keep_mask[batch_idx, view_idx, top_idx] = True

    return keep_mask


def replace_pruned_tokens(token_features, keep_mask, mode):
    if mode == "zero":
        replacement = torch.zeros_like(token_features)
    else:
        mask_float = keep_mask.unsqueeze(-1).float()
        denom = mask_float.sum(dim=2, keepdim=True).clamp_min(1.0)
        replacement = (token_features * mask_float).sum(dim=2, keepdim=True) / denom
        replacement = replacement.expand_as(token_features)

    return torch.where(keep_mask.unsqueeze(-1), token_features, replacement)


def forward_with_token_pruning(model, images, keep_ratios, args, prompt_tokens=None):
    if args.drop_token_mode == "learned":
        return model(
            images,
            keep_ratios=keep_ratios,
            token_score_mode="importance",
            prompt_tokens=prompt_tokens,
        )

    batch_size, num_views, channels, height, width = images.shape
    x = images.view(batch_size * num_views, channels, height, width)
    x = model.patch_embed(x)
    _, num_patches, embed_dim = x.shape
    x = x.view(batch_size, num_views, num_patches, embed_dim)

    keep_mask = build_token_keep_mask(x, keep_ratios)
    x = replace_pruned_tokens(x, keep_mask, args.drop_token_mode)

    x = x + model.spatial_pos_embed + model.view_pos_embed
    x = x.reshape(batch_size, num_views * num_patches, embed_dim)

    cls_tokens = model.cls_token.expand(batch_size, -1, -1) + model.cls_pos_embed
    if prompt_tokens is not None:
        prompt_tokens = prompt_tokens.to(device=x.device, dtype=x.dtype)
        x = torch.cat((cls_tokens, prompt_tokens, x), dim=1)
    else:
        x = torch.cat((cls_tokens, x), dim=1)

    for block in model.blocks:
        x = block(x)

    x = model.norm(x)
    return model.head(x[:, 0])


def get_batch_policy(sample_ids, policy, args):
    view_masks = []
    raw_keep_ratios = []
    effective_ratios = []
    u_values = []

    for sample_id in sample_ids:
        if sample_id in policy:
            item = policy[sample_id]
            view_mask = item["view_mask"]
            keep_ratios = item["keep_ratios"]
        elif args.missing_policy == "all_views":
            view_mask = np.ones(args.num_views, dtype=np.int64)
            keep_ratios = np.ones(args.num_views, dtype=np.float32)
        else:
            raise KeyError(f"sample_id={sample_id} not found in policy log: {args.policy_log}")

        view_masks.append(view_mask)
        raw_keep_ratios.append(keep_ratios)
        effective_ratios.append(effective_keep_ratios(view_mask, keep_ratios, args))
        u_values.append(int(item.get("u", 0)) if sample_id in policy else 0)

    return (
        torch.tensor(np.stack(view_masks, axis=0), dtype=torch.long),
        torch.tensor(np.stack(raw_keep_ratios, axis=0), dtype=torch.float32),
        torch.tensor(np.stack(effective_ratios, axis=0), dtype=torch.float32),
        torch.tensor(u_values, dtype=torch.long),
    )


def write_rows(path, rows, num_views):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "label",
        "baseline_pred",
        "baseline_correct",
        "rl_policy_pred",
        "rl_policy_correct",
        "same_prediction",
        "drift_type",
        "severity",
        "struct_drift",
        "u",
        "used_prompt",
        "used_retrain",
        "active_views",
        "effective_token_ratio_sum",
        *[f"v_{i + 1}" for i in range(num_views)],
        *[f"k_raw_{i + 1}" for i in range(num_views)],
        *[f"k_effective_{i + 1}" for i in range(num_views)],
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_policy_forward(model, images, view_masks, effective_ratios, args, prompt_tokens=None):
    if args.eval_mode == "view_mask":
        policy_images = apply_view_policy(images, view_masks, args.blank_mode)
        return model(policy_images, prompt_tokens=prompt_tokens)
    return forward_with_token_pruning(
        model,
        images,
        effective_ratios,
        args,
        prompt_tokens=prompt_tokens,
    )


@torch.no_grad()
def evaluate(model, retrain_model, prompt_gen, loader, policy, args, device):
    total = 0
    baseline_correct = 0
    policy_correct = 0
    same_predictions = 0
    active_view_sum = 0
    effective_token_ratio_sum = 0.0
    output_rows = []
    sample_offset = 0

    for batch in tqdm(loader, desc="Evaluating RL policy on MV-VIT"):
        if len(batch) == 5:
            images, labels, drift_types, severities, struct_drifts = batch
        elif len(batch) == 4:
            images, labels, drift_types, severities = batch
            struct_drifts = torch.zeros(labels.shape[0])
        else:
            images, labels = batch
            drift_types = ["normal"] * labels.shape[0]
            severities = torch.zeros(labels.shape[0])
            struct_drifts = torch.zeros(labels.shape[0])

        batch_size = labels.shape[0]
        sample_ids = list(range(sample_offset, sample_offset + batch_size))
        sample_offset += batch_size

        view_masks, raw_keep_ratios, effective_ratios, u_values = get_batch_policy(sample_ids, policy, args)
        view_masks = view_masks.to(device)
        raw_keep_ratios = raw_keep_ratios.to(device)
        effective_ratios = effective_ratios.to(device)
        u_values = u_values.to(device)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        severities = severities.cpu()
        struct_drifts = struct_drifts.cpu()

        baseline_logits = model(images)
        baseline_pred = baseline_logits.argmax(dim=1)

        use_prompt = should_use_prompt(u_values, args)
        use_retrain = (u_values == 2) & (retrain_model is not None)
        policy_logits = torch.empty_like(baseline_logits)

        condition_ids = None
        if prompt_gen is not None and torch.any(use_prompt):
            condition_ids = drift_types_to_condition_ids(drift_types, device)

        for model_selector, active_model in [("base", model), ("retrain", retrain_model)]:
            if active_model is None:
                continue
            if model_selector == "retrain":
                model_mask = use_retrain
            else:
                model_mask = ~use_retrain
            if not torch.any(model_mask):
                continue

            for prompt_state in [False, True]:
                sub_mask = model_mask & (use_prompt == prompt_state)
                if not torch.any(sub_mask):
                    continue

                local_indices = torch.where(sub_mask)[0]
                prompt_tokens = None
                if prompt_state and prompt_gen is not None:
                    prompt_tokens = prompt_gen(condition_ids[sub_mask])

                local_logits = run_policy_forward(
                    active_model,
                    images[sub_mask],
                    view_masks[sub_mask],
                    effective_ratios[sub_mask],
                    args,
                    prompt_tokens=prompt_tokens,
                )
                policy_logits[local_indices] = local_logits

        policy_pred = policy_logits.argmax(dim=1)
        baseline_match = baseline_pred == labels
        policy_match = policy_pred == labels
        same_match = baseline_pred == policy_pred

        total += batch_size
        baseline_correct += baseline_match.sum().item()
        policy_correct += policy_match.sum().item()
        same_predictions += same_match.sum().item()
        active_view_sum += view_masks.sum().item()
        effective_token_ratio_sum += effective_ratios.sum().item()

        view_masks_cpu = view_masks.cpu().numpy()
        raw_keep_cpu = raw_keep_ratios.cpu().numpy()
        effective_cpu = effective_ratios.cpu().numpy()
        u_cpu = u_values.cpu().numpy()
        use_prompt_cpu = use_prompt.cpu().numpy()
        use_retrain_cpu = use_retrain.cpu().numpy()
        for local_idx, sample_id in enumerate(sample_ids):
            row = {
                "sample_id": sample_id,
                "label": int(labels[local_idx].item()),
                "baseline_pred": int(baseline_pred[local_idx].item()),
                "baseline_correct": int(baseline_match[local_idx].item()),
                "rl_policy_pred": int(policy_pred[local_idx].item()),
                "rl_policy_correct": int(policy_match[local_idx].item()),
                "same_prediction": int(same_match[local_idx].item()),
                "drift_type": str(drift_types[local_idx]),
                "severity": float(severities[local_idx].item()),
                "struct_drift": float(struct_drifts[local_idx].item()),
                "u": int(u_cpu[local_idx]),
                "used_prompt": int(use_prompt_cpu[local_idx]),
                "used_retrain": int(use_retrain_cpu[local_idx]),
                "active_views": int(view_masks_cpu[local_idx].sum()),
                "effective_token_ratio_sum": float(effective_cpu[local_idx].sum()),
            }
            for view_idx in range(args.num_views):
                row[f"v_{view_idx + 1}"] = int(view_masks_cpu[local_idx, view_idx])
                row[f"k_raw_{view_idx + 1}"] = float(raw_keep_cpu[local_idx, view_idx])
                row[f"k_effective_{view_idx + 1}"] = float(effective_cpu[local_idx, view_idx])
            output_rows.append(row)

    return {
        "total": total,
        "baseline_acc": baseline_correct / total,
        "policy_acc": policy_correct / total,
        "same_pred_rate": same_predictions / total,
        "avg_active_views": active_view_sum / total,
        "avg_effective_token_ratio_sum": effective_token_ratio_sum / total,
        "rows": output_rows,
    }


def print_summary(metrics, args):
    full_tokens = 1 + args.num_views * args.tokens_per_view
    if args.eval_mode == "view_mask":
        avg_policy_tokens = 1 + metrics["avg_active_views"] * args.tokens_per_view
    else:
        avg_policy_tokens = 1 + metrics["avg_effective_token_ratio_sum"] * args.tokens_per_view

    estimated_attention_saving = 1.0 - (avg_policy_tokens ** 2) / (full_tokens ** 2)
    estimated_view_saving = 1.0 - metrics["avg_active_views"] / args.num_views
    acc_drop = metrics["baseline_acc"] - metrics["policy_acc"]

    print("-" * 72)
    print("MV-VIT RL 策略评估完成")
    print(f"评估模式: {args.eval_mode}")
    print(f"漂移日程: {args.drift_schedule}")
    print(f"Prompt 接入: {args.prompt_checkpoint is not None}, prompt_for_u={args.prompt_for_u}")
    print(f"Retrain 接入: {args.retrain_checkpoint is not None}")
    print(f"样本数: {metrics['total']}")
    print(f"完整 4 视角准确率: {metrics['baseline_acc']:.4%}")
    print(f"RL 策略准确率: {metrics['policy_acc']:.4%}")
    print(f"准确率下降: {acc_drop:.4%}")
    print(f"与完整视角预测一致率: {metrics['same_pred_rate']:.4%}")
    print(f"平均激活视角数: {metrics['avg_active_views']:.4f}/{args.num_views}")
    print(f"平均有效 Token 保留率总和: {metrics['avg_effective_token_ratio_sum']:.4f}")
    print(f"理论视角数节省: {estimated_view_saving:.4%}")
    print(f"理论 attention token 二次复杂度节省: {estimated_attention_saving:.4%}")
    if args.eval_mode == "token_prune":
        print("注意: 当前 token_prune 用 token 替换保持序列长度，节省值是物理删除 token 后的理论估计。")
    else:
        print("注意: view_mask 会整视角置空，早期融合模型通常对此非常敏感。")


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")

    policy = load_policy(args.policy_log, args.num_views)
    loader = build_loader(args)
    model = load_model(args, checkpoint_path, device)
    prompt_gen = load_prompt_bundle(args, model, device)
    retrain_model = load_retrain_model(args, device)

    metrics = evaluate(model, retrain_model, prompt_gen, loader, policy, args, device)
    write_rows(args.output, metrics["rows"], args.num_views)
    print_summary(metrics, args)
    print(f"逐样本评估日志已保存: {args.output}")


if __name__ == "__main__":
    main()
