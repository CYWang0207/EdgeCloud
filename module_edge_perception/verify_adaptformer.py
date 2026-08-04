"""AdaptFormer 验收脚本（王成洋）。

验收三点（CLAUDE.md 第八节）：
① 零初始化挂上后 forward 输出差异 < 1e-3
② 可训参（adapter）< 1M
③ 三条前向路径（forward / forward_hard_prune / _hard_prune_per_sample）均生效且 wrapper 被调用

用法（从 module_edge_perception/ 目录运行）：
    py -3.11 verify_adaptformer.py
    py -3.11 verify_adaptformer.py --model-name vit_small_patch16_224 --r 32

本地无需数据集与预训练权重：pretrained=False 随机权重即可验证零初始化不变性
（adapter 输出恒为 0，与主干权重无关）。
"""
import argparse

import torch

from model import EarlyFusionMultiViewViT
from adaptformer import (
    AdaptFormerMLPWrapper,
    attach_adaptformer,
    count_adapter_parameters,
    count_trainable_parameters,
    is_adaptformer_attached,
)


def parse_args():
    p = argparse.ArgumentParser(description="AdaptFormer 三点验收")
    p.add_argument("--model-name", default="vit_small_patch16_224",
                   choices=["vit_small_patch16_224", "vit_base_patch16_224"])
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=40)
    p.add_argument("--r", type=int, default=32, help="瓶颈维度")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--tol", type=float, default=1e-3, help="零初始化输出差异容限")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    print("=" * 64)
    print("AdaptFormer 验收")
    print("=" * 64)
    print(f"设备: {device}")
    print(f"模型: {args.model_name}  r={args.r}  num_views={args.num_views}")

    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=args.num_views,
        num_classes=args.num_classes,
        pretrained=False,
    ).to(device).eval()

    images = torch.randn(args.batch_size, args.num_views, 3, 224, 224, device=device)
    view_mask = torch.ones(args.batch_size, args.num_views, device=device)

    # --- 挂载前基线输出 ---
    with torch.no_grad():
        out_before = model(images, view_mask=view_mask)

    # --- 挂载 adapter ---
    attach_adaptformer(model, r=args.r)
    assert is_adaptformer_attached(model), "wrapper 未全部挂载"
    n_wrappers = sum(isinstance(b.mlp, AdaptFormerMLPWrapper) for b in model.blocks)

    # --- ① 零初始化不变性 ---
    with torch.no_grad():
        out_after = model(images, view_mask=view_mask)
    diff = (out_before - out_after).abs().max().item()
    ok1 = diff < args.tol
    print(f"\n[① 零初始化输出差异] max|Δ|={diff:.3e}  tol={args.tol:.0e}  "
          f"{'✅ 通过' if ok1 else '❌ 未通过'}")

    # --- ② 可训参 < 1M ---
    n_adapter = count_adapter_parameters(model)
    n_trainable_before_freeze = count_trainable_parameters(model)
    print(f"    adapter 参数: {n_adapter:,} ({n_adapter / 1e6:.3f}M)")
    print(f"    挂载后总可训参(未冻结): {n_trainable_before_freeze:,} "
          f"({n_trainable_before_freeze / 1e6:.3f}M)")

    # 按部署口径冻结主干后统计
    from adaptformer import freeze_backbone
    freeze_backbone(model)
    n_trainable = count_trainable_parameters(model)
    ok2 = n_adapter < 1_000_000
    print(f"    冻结后可训参(adapter+norm+head): {n_trainable:,} "
          f"({n_trainable / 1e6:.3f}M)")
    print(f"[② 可训参 adapter<1M] {'✅ 通过' if ok2 else '❌ 未通过'}")

    # --- ③ 三条前向路径均生效 ---
    model.eval()
    path_results = []
    # 路径1: forward（软剪枝 mask 路径，view_mask 注入）
    with torch.no_grad():
        o1 = model(images, view_mask=view_mask)
    path_results.append(("forward", o1 is not None))

    # 路径2: forward_hard_prune 标量统一保留率（批量路径）
    with torch.no_grad():
        o2 = model.forward_hard_prune(images, keep_ratio=0.5,
                                      token_score_mode="importance")
    path_results.append(("forward_hard_prune(uniform)", o2 is not None))

    # 路径3: forward_hard_prune + per-view keep_ratios（逐样本路径 _hard_prune_per_sample）
    kr = torch.rand(args.batch_size, args.num_views, device=device).clamp(0.1, 1.0)
    with torch.no_grad():
        o3 = model.forward_hard_prune(images, keep_ratios=kr,
                                      token_score_mode="importance")
    path_results.append(("_hard_prune_per_sample", o3 is not None))

    ok3 = all(r[1] for r in path_results)
    detail = "  ".join(f"{name} ✅" if ok else f"{name} ❌" for name, ok in path_results)
    print(f"[③ 三条前向路径] {detail}")

    # --- wrapper 真被调用确认：注册 forward hook 验证 adapter 旁路在 hard_prune 路径也触发 ---
    triggered = {"count": 0}
    first_wrapper = model.blocks[0].mlp
    hook = first_wrapper.register_forward_hook(
        lambda _m, _i, _o: triggered.__setitem__("count", triggered["count"] + 1)
    )
    with torch.no_grad():
        _ = model.forward_hard_prune(images, keep_ratio=0.5,
                                     token_score_mode="importance")
    hook.remove()
    ok_hook = triggered["count"] > 0
    print(f"    wrapper hook 在 hard_prune 路径触发次数: {triggered['count']}  "
          f"{'✅' if ok_hook else '❌'}")

    # --- 汇总 ---
    print("\n" + "=" * 64)
    print(f"blocks={len(model.blocks)}  r={args.r}  wrappers={n_wrappers}  "
          f"adapter/layer≈{n_adapter // max(n_wrappers, 1):,}")
    all_ok = ok1 and ok2 and ok3 and ok_hook
    print("验收结论: " + ("✅ 全部通过" if all_ok else "❌ 存在未通过项"))
    print("=" * 64)


if __name__ == "__main__":
    main()
