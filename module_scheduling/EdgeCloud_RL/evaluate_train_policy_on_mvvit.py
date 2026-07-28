import argparse
from pathlib import Path

import torch

import evaluate_rl_policy_on_mvvit as rl_eval


SCRIPT_DIR = Path(__file__).resolve().parent
MV_VIT_DIR = rl_eval.MV_VIT_DIR


def parse_args():
    default_dataset = MV_VIT_DIR / "data" / "modelnet40v2png_ori4"
    default_checkpoint = MV_VIT_DIR / "checkpoints" / "mv_vit_token_epoch_19.pth"
    default_prompt = MV_VIT_DIR / "checkpoints" / "prompt_token_aware_best.pth"
    default_retrain = MV_VIT_DIR / "checkpoints" / "mv_vit_retrain_mixed_epoch_8.pth"
    default_policy = SCRIPT_DIR / "rl_train_mixed.csv"
    default_output = SCRIPT_DIR / "eval_train_rl_vku_assisted.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate train split with RL view selection, token pruning, "
            "prompt assistance, and retrained model switching."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=default_dataset)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--prompt-checkpoint", type=Path, default=default_prompt)
    parser.add_argument("--retrain-checkpoint", type=Path, default=default_retrain)
    parser.add_argument("--policy-log", type=Path, default=default_policy)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--model-name", type=str, default="vit_small_patch16_224")
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
        default="mixed",
        choices=["none", "light", "mixed", "staged", "highfreq"],
    )
    parser.add_argument("--drift-seed", type=int, default=123)
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="token_prune",
        choices=["token_prune", "view_mask"],
    )
    parser.add_argument(
        "--prompt-for-u",
        type=str,
        default="u1",
        choices=["none", "u1", "u1_u2"],
        help="u1 means prompt is used for u=1, while u=2 uses the retrained model.",
    )
    parser.add_argument("--inactive-view-keep", type=float, default=0.1)
    parser.add_argument("--min-keep", type=float, default=0.0)
    parser.add_argument(
        "--drop-token-mode",
        type=str,
        default="learned",
        choices=["learned", "mean", "zero"],
    )
    parser.add_argument(
        "--blank-mode",
        type=str,
        default="zero",
        choices=["zero", "black"],
    )
    parser.add_argument(
        "--missing-policy",
        type=str,
        default="error",
        choices=["error", "all_views"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = rl_eval.resolve_checkpoint(args.checkpoint)

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")

    print("-" * 72)
    print("Train split RL-v/k/u MV-VIT evaluation")
    print(f"dataset split: {args.split}")
    print(f"policy log: {args.policy_log}")
    print(f"prompt checkpoint: {args.prompt_checkpoint}")
    print(f"retrain checkpoint: {args.retrain_checkpoint}")
    print("-" * 72)

    policy = rl_eval.load_policy(args.policy_log, args.num_views)
    loader = rl_eval.build_loader(args)
    model = rl_eval.load_model(args, checkpoint_path, device)
    prompt_gen = rl_eval.load_prompt_bundle(args, model, device)
    retrain_model = rl_eval.load_retrain_model(args, device)

    metrics = rl_eval.evaluate(model, retrain_model, prompt_gen, loader, policy, args, device)
    rl_eval.write_rows(args.output, metrics["rows"], args.num_views)
    rl_eval.print_summary(metrics, args)
    print(f"Train-set evaluation log saved: {args.output}")


if __name__ == "__main__":
    main()
