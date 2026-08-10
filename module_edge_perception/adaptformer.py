"""AdaptFormer PEFT 模块（王成洋实现）。

挂载点：每个 timm Transformer block 的 mlp(FFN) 旁路并行。
结构：adapter = down(D->r) -> GELU -> up(r->D)，W_up 零初始化使启动时 adapter 输出为 0，
      不破坏预训练主干。
前向：wrapper.forward(x) = mlp(x) + scale * adapter(x)。替换 block.mlp 后，
      model.py 的三条前向路径（forward / forward_hard_prune / _hard_prune_per_sample）
      共用同一组 self.blocks，因此 adapter 自动在三条路径全部生效，无需改前向逻辑。
冻结策略：主干全部 requires_grad=False，只训 adapter + norm + head（见 freeze_backbone）。

参考论文：AdaptFormer (ICCV 2022)；相关对比：LoRA (ICLR 2022)、Houlsby Adapter (ICML 2019)。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AdaptFormerMLP(nn.Module):
    """单层 AdaptFormer adapter：down(D->r) -> GELU -> up(r->D)。

    W_up（含权重与偏置）零初始化，故挂载瞬间 adapter 输出恒为 0 向量，
    串联进 FFN 旁路后不改变预训练主干输出（验收①的来源）。
    """

    def __init__(self, dim: int, r: int = 32):
        super().__init__()
        self.dim = dim
        self.r = r
        self.down = nn.Linear(dim, r)
        self.act = nn.GELU()
        self.up = nn.Linear(r, dim)
        # 零初始化 W_up，保证启动时 adapter 输出为 0
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


class AdaptFormerMLPWrapper(nn.Module):
    """包裹 timm 原生 block.mlp，旁路并联 adapter。

    forward(x) = mlp(x) + scale * adapter(x)
    其中 scale 为可学习标量（每层一个参数），init=1.0；因 adapter 输出零初始化，
    scale 任意初值乘 0 仍为 0，挂载即不改输出。训练阶段 scale 学到合适增益。
    """

    def __init__(self, mlp: nn.Module, dim: int, r: int = 32, scale: float = 1.0):
        super().__init__()
        self.mlp = mlp  # 保留原 FFN（预训练权重随 mlp 一并保留）
        self.adapter = AdaptFormerMLP(dim, r)
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        # 运行时开关（默认 True，向后兼容）：False 时旁路完全关闭，输出 = 原 FFN。
        # 供 B 的 benchmark 做"带/不带 adapter"干净对比。
        self.enabled = True

    def forward(self, x):
        if not self.enabled:
            return self.mlp(x)
        return self.mlp(x) + self.scale * self.adapter(x)


def _block_embed_dim(block) -> int:
    """从 timm block 推断 embed_dim。"""
    mlp = block.mlp
    if hasattr(mlp, "fc1") and hasattr(mlp.fc1, "in_features"):
        return int(mlp.fc1.in_features)
    if hasattr(block, "norm2") and hasattr(block.norm2, "normalized_shape"):
        return int(block.norm2.normalized_shape[0])
    raise ValueError("无法从 block 推断 embed_dim，请检查 timm 版本")


def attach_adaptformer(model, r: int = 32, scale: float = 1.0):
    """遍历 model.blocks，把每个 block.mlp 替换为 AdaptFormerMLPWrapper（就地替换）。

    返回 model 本身（已在原对象上完成替换，无需重新赋值）。
    适配 EarlyFusionMultiViewViT：model.blocks 即 timm VisionTransformer 的 blocks。
    """
    if not hasattr(model, "blocks") or len(model.blocks) == 0:
        raise ValueError("model 缺少 blocks 属性或为空，无法挂载 AdaptFormer")
    dim = _block_embed_dim(model.blocks[0])
    for block in model.blocks:
        block.mlp = AdaptFormerMLPWrapper(block.mlp, dim, r=r, scale=scale)
    return model


def freeze_backbone(model):
    """主干全部冻结，只训 adapter + norm + head。

    按 CLAUDE.md 冻结策略：missing_view_token / token_mask_token /
    view_pos_embed / patch_embed / pos_embed / cls_token 等全部 requires_grad=False，
    只放开 adapter（含 scale）、norm、head。
    """
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks:
        mlp = block.mlp
        if isinstance(mlp, AdaptFormerMLPWrapper):
            for p in mlp.adapter.parameters():
                p.requires_grad = True
            mlp.scale.requires_grad = True
    for p in model.norm.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    return model


def adapter_parameters(model):
    """迭代返回所有 adapter 参数（含 scale），供优化器单独分组用。"""
    for block in model.blocks:
        mlp = block.mlp
        if isinstance(mlp, AdaptFormerMLPWrapper):
            yield from mlp.adapter.parameters()
            yield mlp.scale


def count_adapter_parameters(model) -> int:
    """统计 adapter（含 scale）参数总量。"""
    return sum(p.numel() for p in adapter_parameters(model))


def count_trainable_parameters(model) -> int:
    """统计 requires_grad=True 的参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_adaptformer_attached(model) -> bool:
    """是否所有 block.mlp 均已替换为 wrapper。"""
    if not hasattr(model, "blocks") or len(model.blocks) == 0:
        return False
    return all(isinstance(b.mlp, AdaptFormerMLPWrapper) for b in model.blocks)


def set_adapter_enabled(model, enabled: bool):
    """批量开关所有 wrapper 的 adapter 旁路（benchmark 对比用）。

    enabled=False 时所有 wrapper 输出 = 原 FFN，等价于"不带 adapter"的模型。
    """
    for block in model.blocks:
        mlp = block.mlp
        if isinstance(mlp, AdaptFormerMLPWrapper):
            mlp.enabled = bool(enabled)
    return model


def collect_adapter_state(model) -> dict:
    """收集仅 adapter（含 scale）+ norm + head 的参数，供云端下发（几百 KB~MB）。

    主干（patch_embed / pos_embed / blocks 原 FFN 等）一律不收集。
    返回 {name: cpu tensor}，name 与 model.state_dict() 的键一致。
    """
    state = {}
    for name, p in model.named_parameters():
        if (
            "mlp.adapter." in name
            or name.endswith("mlp.scale")
            or name.startswith(("norm.", "head."))
        ):
            state[name] = p.detach().cpu()
    return state


def save_adapter_checkpoint(path, model, **meta):
    """以 adapter-only 格式保存蒸馏产物：{"adapter": {...}, **meta}。

    adapter 子字典只含 blocks.*.mlp.adapter.* / blocks.*.mlp.scale / norm / head，
    体积 ≈ 0.3M 参 × 4B ≈ 1.2MB，对应接口契约 u=1 的 S_adapter 通信口径。
    """
    payload = {"adapter": collect_adapter_state(model)}
    payload.update(meta)
    torch.save(payload, path)


def load_adapter_checkpoint(model, path, device="cpu"):
    """把 adapter 权重加载进已 attach 的 model（不覆盖冻结主干）。

    兼容两种格式：
    - {"adapter": {...}, "norm": {...}, "head": {...}}（adapter-only 训练产出）
    - 整模型 state_dict（含 blocks.*.mlp.adapter.*，如 baseline 风格 checkpoint）
    返回 (missing, unexpected)。
    """
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "adapter" in payload:
        missing, unexpected = model.load_state_dict(payload["adapter"], strict=False)
        for prefix in ("norm", "head"):
            if prefix in payload and isinstance(payload[prefix], dict):
                model.load_state_dict(payload[prefix], strict=False)
        return missing, unexpected
    return model.load_state_dict(payload, strict=False)
