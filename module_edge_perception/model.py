import math

import torch
import torch.nn as nn
import timm


class MultiViewDriftConditioner(nn.Module):
    """Predict a continuous environment vector and per-view reliability.

    It consumes frozen MV-ViT patch features, so the trainable part stays small.
    Reliability is explicit and can be weakly supervised by synthetic drift.
    """

    def __init__(self, embed_dim, condition_dim=128):
        super().__init__()
        hidden_dim = max(condition_dim, 128)
        self.view_projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )
        self.condition_head = nn.Linear(hidden_dim, condition_dim)
        self.quality_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.quality_head.weight)
        nn.init.constant_(self.quality_head.bias, 4.0)

    def forward(self, tokens, view_mask=None):
        view_features = tokens.mean(dim=2)
        hidden = self.view_projector(view_features)
        if view_mask is None:
            weights = torch.ones(
                hidden.shape[:2], device=hidden.device, dtype=hidden.dtype
            )
        else:
            weights = view_mask.to(device=hidden.device, dtype=hidden.dtype)
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        condition = self.condition_head(pooled)
        condition = torch.nn.functional.normalize(condition.float(), dim=-1).to(hidden.dtype)
        quality = torch.sigmoid(self.quality_head(hidden).squeeze(-1)) * weights
        return condition, quality


class EarlyFusionMultiViewViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_views=4, num_classes=40, pretrained=True):
        """
        保留了早期跨视角融合的 ViT，同时加载了 ImageNet 预训练权重
        """
        super().__init__()
        self.num_views = num_views

        # 1. 获取预训练的 2D ViT
        print(f"正在加载 {model_name} 预训练权重用于早期融合...")
        vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = vit.num_features

        # ================== 偷取并复用预训练组件 ==================

        # 复用 Patch Embedding 层
        self.patch_embed = vit.patch_embed
        num_patches = self.patch_embed.num_patches  # 通常是 14x14 = 196

        # 预训练的位置编码 (包含了 1 个 CLS 和 196 个 Patch) -> 形状: [1, 197, 768]
        pretrained_pos_embed = vit.pos_embed

        # 把 CLS 的位置编码和 Patch 的空间位置编码拆开，继承给我们的模型
        self.cls_token = vit.cls_token  # [1, 1, 768]
        self.cls_pos_embed = nn.Parameter(pretrained_pos_embed[:, 0:1, :])  # 取出第0个作为 CLS 位置编码

        # 空间位置编码 (196 个 Patch 的相对位置)
        # 形状变为 [1, 1, 196, 768]，方便后面广播相加
        self.spatial_pos_embed = nn.Parameter(pretrained_pos_embed[:, 1:, :].unsqueeze(1))

        # ================== 我们自己定义的多视角组件 ==================

        # 视角位置编码: 标识该 Patch 属于哪个摄像头视角 (预训练模型没有这个，需要随机初始化)
        # 形状: [1, num_views, 1, embed_dim]
        self.view_pos_embed = nn.Parameter(torch.randn(1, num_views, 1, embed_dim) * 0.02)
        self.missing_view_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.token_mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.normal_(self.missing_view_token, std=0.02)
        nn.init.normal_(self.token_mask_token, std=0.02)

        # 复用预训练的 Transformer 核心层 (这里是深层跨视角融合的发生地)
        self.blocks = vit.blocks
        self.norm = vit.norm

        # 最终的分类头
        self.head = nn.Linear(embed_dim, num_classes)
        self.drift_conditioner = None
        self.bad_view_token = None

    def attach_drift_conditioner(self, condition_dim=128):
        """Attach the edge-side condition encoder without changing old models."""
        embed_dim = self.head.in_features
        self.drift_conditioner = MultiViewDriftConditioner(
            embed_dim, condition_dim=condition_dim
        )
        self.bad_view_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.normal_(self.bad_view_token, std=0.02)
        return self

    def _build_token_keep_mask(self, tokens, keep_ratios, token_score_mode):
        """
        tokens: [B, V, N, D]
        keep_ratios: [B, V], values in [0, 1]
        """
        B, V, N, _ = tokens.shape
        keep_ratios = keep_ratios.to(device=tokens.device, dtype=tokens.dtype).clamp(0.0, 1.0)
        keep_counts = torch.ceil(keep_ratios * N).long().clamp(1, N)

        if token_score_mode == "random":
            scores = torch.rand(B, V, N, device=tokens.device)
        elif token_score_mode == "importance":
            scores = tokens.detach().norm(dim=-1)
        else:
            raise ValueError(f"不支持的 token_score_mode: {token_score_mode}")

        keep_mask = torch.zeros(B, V, N, dtype=torch.bool, device=tokens.device)
        for b in range(B):
            for v in range(V):
                keep_count = int(keep_counts[b, v].item())
                top_idx = torch.topk(scores[b, v], keep_count).indices
                keep_mask[b, v, top_idx] = True

        return keep_mask

    def _apply_token_pruning(self, tokens, keep_ratios, token_score_mode):
        keep_mask = self._build_token_keep_mask(tokens, keep_ratios, token_score_mode)
        mask_token = self.token_mask_token.expand_as(tokens)
        return torch.where(keep_mask.unsqueeze(-1), tokens, mask_token)

    def _apply_view_mask(self, tokens, view_mask):
        view_mask = view_mask.to(device=tokens.device)
        active = view_mask.bool().view(tokens.shape[0], tokens.shape[1], 1, 1)
        missing = self.missing_view_token.expand_as(tokens)
        return torch.where(active, tokens, missing)

    def forward(self, x, view_mask=None, keep_ratios=None, token_score_mode=None,
                prompt_tokens=None, return_features=False, condition_vector=None,
                return_aux=False, apply_quality_gate=True):
        """
        x: [Batch_size, Views, Channels, Height, Width]
        view_mask: optional [Batch_size, Views], 1 means available, 0 means missing.
        keep_ratios: optional [Batch_size, Views], token keep ratio for each view.
        token_score_mode: "random" for pruning-aware training, "importance" for deterministic evaluation.
        prompt_tokens: optional [Batch_size, PromptTokens, EmbedDim].
        """
        B, V, C, H, W = x.shape
        assert V == self.num_views, f"期待输入 {self.num_views} 个视角，但得到了 {V} 个"

        # --- 1. Patch 提取 ---
        # 展平以便送入 2D Patch Embed
        x = x.view(B * V, C, H, W)
        x = self.patch_embed(x)  # 形状: [B*V, num_patches, embed_dim]

        _, N, D = x.shape

        # --- 2. 双重位置编码注入 ---
        # 恢复出 B 和 V 维度 -> [B, V, num_patches, embed_dim]
        x = x.view(B, V, N, D)

        edge_condition = quality = None
        if self.drift_conditioner is not None:
            edge_condition, quality = self.drift_conditioner(x, view_mask)
            if apply_quality_gate:
                bad = self.bad_view_token.expand_as(x)
                gate = quality.to(dtype=x.dtype).view(B, V, 1, 1)
                x = gate * x + (1.0 - gate) * bad

        active_condition = condition_vector
        if active_condition is None:
            active_condition = edge_condition
        # Runtime conditioning keeps timm block signatures and pruning paths
        # backward compatible. Import locally to avoid a model/adaptformer cycle.
        try:
            from adaptformer import set_adapter_condition
        except ImportError:
            from .adaptformer import set_adapter_condition
        set_adapter_condition(self, active_condition)

        if keep_ratios is not None:
            if token_score_mode is None:
                token_score_mode = "random" if self.training else "importance"
            x = self._apply_token_pruning(x, keep_ratios, token_score_mode)

        if view_mask is not None:
            x = self._apply_view_mask(x, view_mask)

        # 广播机制：注入继承自 ImageNet 的空间位置编码，以及我们自己的视角位置编码
        x = x + self.spatial_pos_embed + self.view_pos_embed

        # --- 3. 序列大拼接 (早期融合的核心) ---
        # 拉平成一条包含所有视角 Patch 的超长序列 -> [B, V * N, D]
        x = x.reshape(B, V * N, D)

        # 加上 CLS Token 和可选 Prompt Tokens
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos_embed
        if prompt_tokens is not None:
            prompt_tokens = prompt_tokens.to(device=x.device, dtype=x.dtype)
            x = torch.cat((cls_tokens, prompt_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)  # 最终序列长度为 1 + 4*196 = 785

        # --- 4. 联合自注意力计算 ---
        # 所有 785 个 Token 在同一群 Transformer Block 中进行密集交互！
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # --- 5. 提取输出 ---
        cls_out = x[:, 0]
        out = self.head(cls_out)

        if return_aux:
            return out, {
                "features": cls_out,
                "edge_condition": edge_condition,
                "view_quality": quality,
                "active_condition": active_condition,
            }
        if return_features:
            return out, cls_out
        return out

    def forward_hard_prune(self, x, keep_ratio=1.0, keep_ratios=None,
                          token_score_mode="importance", prompt_tokens=None):
        """
        硬剪枝前向：真正删除被剪 token，缩短序列长度，使 self-attention 复杂度 O(N^2) 真实下降。
        与 forward() 的软剪枝（mask token 替换）形成对比，用于验证 TTFT 降低。

        x: [B, V, C, H, W]
        keep_ratio: 标量 float in [0,1]，所有视角统一保留率（benchmark 简化版）
        keep_ratios: [B, V] 张量，每视角不同保留率（RL 调度对接用）。优先于 keep_ratio。
        token_score_mode: "random" / "importance"
        prompt_tokens: optional [B, P, D]

        注意：当 keep_ratios 为 None 且 keep_ratio 为标量时，batch 内序列长度一致可批量处理；
        当 keep_ratios 指定每视角不同时，各样本序列长度可能不同，逐样本处理。
        """
        B, V, C, H, W = x.shape
        assert V == self.num_views

        # 0. 无需剪枝时直接走原 forward
        if keep_ratios is None:
            keep_ratio = float(keep_ratio)
            if keep_ratio >= 1.0:
                return self.forward(x, prompt_tokens=prompt_tokens)
            # 标量统一保留率，走批量路径
            return self._hard_prune_uniform(x, keep_ratio, token_score_mode, prompt_tokens)

        # 每视角不同的 keep_ratios，走逐样本路径
        keep_ratios = keep_ratios.to(device=x.device, dtype=torch.float32).clamp(0.0, 1.0)
        # 若所有视角保留率相同且 >= 1.0，直接走原 forward
        if torch.all(keep_ratios >= 1.0):
            return self.forward(x, prompt_tokens=prompt_tokens)

        outputs = []
        for b in range(B):
            single_x = x[b:b+1]  # [1, V, C, H, W]
            single_keep = keep_ratios[b]  # [V]
            single_prompt = prompt_tokens[b:b+1] if prompt_tokens is not None else None
            # 逐视角 gather 不同数量的 token，然后拼接
            out = self._hard_prune_per_sample(single_x, single_keep, token_score_mode, single_prompt)
            outputs.append(out)
        return torch.cat(outputs, dim=0)

    def _hard_prune_uniform(self, x, keep_ratio, token_score_mode, prompt_tokens):
        """标量统一保留率的硬剪枝（批量处理，序列长度一致）。"""
        B, V, C, H, W = x.shape

        # 1. Patch 提取
        x = x.view(B * V, C, H, W)
        x = self.patch_embed(x)
        _, N, D = x.shape
        x = x.view(B, V, N, D)

        # 2. 计算保留数量（标量，所有视角统一）
        keep_count = max(1, int(math.ceil(keep_ratio * N)))
        keep_count = min(keep_count, N)

        # 3. 按重要性选 top-k token 索引
        if token_score_mode == "importance":
            scores = x.detach().norm(dim=-1)
        elif token_score_mode == "random":
            scores = torch.rand(B, V, N, device=x.device)
        else:
            raise ValueError(f"不支持的 token_score_mode: {token_score_mode}")

        top_idx = torch.topk(scores, keep_count, dim=-1).indices
        top_idx_sorted, _ = torch.sort(top_idx, dim=-1)

        # 4. 收集保留的 token + 对应位置编码
        gather_idx = top_idx_sorted.unsqueeze(-1).expand(-1, -1, -1, D)
        x_keep = torch.gather(x, 2, gather_idx)

        spatial_pe = self.spatial_pos_embed.expand(B, V, N, D)
        pe_keep = torch.gather(spatial_pe, 2, gather_idx)
        view_pe = self.view_pos_embed.expand(B, V, keep_count, D)

        x_keep = x_keep + pe_keep + view_pe

        # 5. 拉平成序列
        x_flat = x_keep.reshape(B, V * keep_count, D)

        # 6. 拼接 CLS + Prompt
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos_embed
        if prompt_tokens is not None:
            prompt_tokens = prompt_tokens.to(device=x_flat.device, dtype=x_flat.dtype)
            x_flat = torch.cat((cls_tokens, prompt_tokens, x_flat), dim=1)
        else:
            x_flat = torch.cat((cls_tokens, x_flat), dim=1)

        # 7. Transformer blocks
        for block in self.blocks:
            x_flat = block(x_flat)

        x_flat = self.norm(x_flat)
        return self.head(x_flat[:, 0])

    def _hard_prune_per_sample(self, x, keep_ratios_vec, token_score_mode, prompt_tokens):
        """每视角不同保留率的硬剪枝（单样本处理，序列长度 = sum(各视角 keep_count)）。

        x: [1, V, C, H, W]
        keep_ratios_vec: [V] 张量
        """
        B, V, C, H, W = x.shape  # B=1
        x = x.view(B * V, C, H, W)
        x = self.patch_embed(x)
        _, N, D = x.shape
        x = x.view(B, V, N, D)

        # 每视角保留数量
        keep_counts = torch.ceil(keep_ratios_vec * N).long().clamp(1, N)  # [V]

        # 逐视角 gather
        kept_tokens = []
        for v in range(V):
            kc = int(keep_counts[v].item())
            xv = x[0, v]  # [N, D]
            if token_score_mode == "importance":
                scores_v = xv.detach().norm(dim=-1)  # [N]
            elif token_score_mode == "random":
                scores_v = torch.rand(N, device=xv.device)
            else:
                raise ValueError(f"不支持的 token_score_mode: {token_score_mode}")

            top_idx = torch.topk(scores_v, kc).indices
            top_idx_sorted, _ = torch.sort(top_idx)
            xv_keep = xv[top_idx_sorted]  # [kc, D]

            # 对应位置编码
            pe_v = self.spatial_pos_embed[0, 0, top_idx_sorted]  # [kc, D]
            view_pe_v = self.view_pos_embed[0, v].expand(kc, D)
            xv_keep = xv_keep + pe_v + view_pe_v
            kept_tokens.append(xv_keep)

        # 拼接所有视角的保留 token
        x_flat = torch.cat(kept_tokens, dim=0).unsqueeze(0)  # [1, sum_kc, D]

        # 拼接 CLS + Prompt
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos_embed
        if prompt_tokens is not None:
            prompt_tokens = prompt_tokens.to(device=x_flat.device, dtype=x_flat.dtype)
            x_flat = torch.cat((cls_tokens, prompt_tokens, x_flat), dim=1)
        else:
            x_flat = torch.cat((cls_tokens, x_flat), dim=1)

        for block in self.blocks:
            x_flat = block(x_flat)

        x_flat = self.norm(x_flat)
        return self.head(x_flat[:, 0])
