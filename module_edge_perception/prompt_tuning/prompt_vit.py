import torch
import torch.nn as nn
import timm


class PromptMultiViewViT(nn.Module):
    """
    支持提示词注入的多视角早期融合 ViT
    该类专门用于 prompt_tuning 模块，保持与原 model.py 逻辑隔离
    """

    def __init__(self, model_name='vit_small_patch16_224', num_views=4, num_classes=40, pretrained=False):
        super().__init__()
        self.num_views = num_views

        # 1. 初始化基础 ViT 组件 (与原模型结构保持完全一致，确保权重可直接加载)
        vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = vit.num_features

        self.patch_embed = vit.patch_embed
        num_patches = self.patch_embed.num_patches
        pretrained_pos_embed = vit.pos_embed

        #  stealing/reusing 组件
        self.cls_token = vit.cls_token
        self.cls_pos_embed = nn.Parameter(pretrained_pos_embed[:, 0:1, :])
        self.spatial_pos_embed = nn.Parameter(pretrained_pos_embed[:, 1:, :].unsqueeze(1))

        # 视角位置编码
        self.view_pos_embed = nn.Parameter(torch.randn(1, num_views, 1, self.embed_dim) * 0.02)

        # Transformer 核心层
        self.blocks = vit.blocks
        self.norm = vit.norm

        # 分类头
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x, prompt_tokens=None):
        """
        x: [B, V, C, H, W]
        prompt_tokens: [B, num_prompts, D] (来自 PromptGenerator)
        """
        B, V, C, H, W = x.shape

        # --- 1. Patch 提取 ---
        x = x.view(B * V, C, H, W)
        x = self.patch_embed(x)  # [B*V, N, D]
        _, N, D = x.shape

        # --- 2. 位置编码注入 ---
        x = x.view(B, V, N, D)
        x = x + self.spatial_pos_embed + self.view_pos_embed

        # --- 3. 序列大拼接 (关键修改点) ---
        x = x.reshape(B, V * N, D)  # 展平所有视角的所有 Patch

        # 准备 CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos_embed

        # 注入 Prompt Tokens
        if prompt_tokens is not None:
            # 拼接顺序: [CLS] + [Prompts] + [Image Patches]
            # 这种顺序可以让 CLS Token 同时观察到提示信息和图像信息
            x = torch.cat((cls_tokens, prompt_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        # --- 4. Transformer 交互 ---
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # --- 5. 提取输出 ---
        # 无论插入多少个 Prompt，CLS 永远在索引 0 的位置
        cls_out = x[:, 0]
        out = self.head(cls_out)

        return out