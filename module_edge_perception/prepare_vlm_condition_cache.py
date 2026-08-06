"""Convert raw VLM visual hidden states into a compact condition cache.

Input is a torch file whose split values are tensors shaped ``[N, ..., D]``.
All axes between sample and feature are mean-pooled.  PCA is fitted on the
training split, reused for validation/test, and the resulting conditions are
L2-normalized.  The output is consumed by ``--vlm-condition-cache``.
"""

import argparse
import os

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare compact VLM conditions")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition-dim", type=int, default=128)
    parser.add_argument("--train-key", default="train")
    return parser.parse_args()


def pool_features(tensor, view_mask=None):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
        raise ValueError("each split must be a tensor shaped [N, ..., D]")
    tensor = tensor.float()
    if view_mask is not None:
        if tensor.ndim < 3 or view_mask.shape != tensor.shape[:2]:
            raise ValueError("view mask must match the first two feature axes")
        weights = view_mask.float()
        while weights.ndim < tensor.ndim:
            weights = weights.unsqueeze(-1)
        tensor = (tensor * weights).sum(dim=1)
        denominator = view_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        while denominator.ndim < tensor.ndim:
            denominator = denominator.unsqueeze(-1)
        tensor = tensor / denominator
    if tensor.ndim > 2:
        tensor = tensor.mean(dim=tuple(range(1, tensor.ndim - 1)))
    return tensor


def main():
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu")
    if not isinstance(payload, dict) or args.train_key not in payload:
        raise ValueError(f"input must be a split dictionary containing {args.train_key!r}")
    pooled = {}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor) or key.endswith("_view_mask"):
            continue
        mask = payload.get(f"{key}_view_mask")
        pooled[key] = pool_features(value, mask)
    train = pooled[args.train_key]
    if not 0 < args.condition_dim <= min(train.shape):
        raise ValueError("condition_dim must not exceed train samples or feature width")
    mean = train.mean(dim=0, keepdim=True)
    _, _, basis = torch.pca_lowrank(
        train - mean, q=args.condition_dim, center=False
    )
    conditions = {
        key: F.normalize((value - mean) @ basis, dim=-1).half()
        for key, value in pooled.items()
    }
    conditions["projection_mean"] = mean
    conditions["projection_basis"] = basis
    conditions["condition_dim"] = args.condition_dim
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(conditions, args.output)
    print(
        f"saved {args.output}: "
        + ", ".join(
            f"{key}={tuple(value.shape)}"
            for key, value in conditions.items()
            if isinstance(value, torch.Tensor) and key in pooled
        )
    )


if __name__ == "__main__":
    main()
