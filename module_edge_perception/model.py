import torch
import torch.nn as nn
import timm


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

    def forward(self, x, view_mask=None, keep_ratios=None, token_score_mode=None, prompt_tokens=None):
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

        return out
