import torch
import torch.nn as nn


class PromptGenerator(nn.Module):
    """
    视觉提示词生成器 (Visual Prompt Generator)
    功能：根据输入的环境条件ID，生成对应的可学习的 Prompt Tokens
    """

    def __init__(self, vit_embed_dim, num_prompt_tokens=4, num_conditions=3):
        """
        参数:
            vit_embed_dim: 主干 ViT 的特征维度 (比如 vit_small 是 384, vit_base 是 768)
            num_prompt_tokens: 每次你想插入几个 Prompt token
            num_conditions: 你的环境条件种类数 (例如 0:正常, 1:变亮, 2:变暗)
        """
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.vit_embed_dim = vit_embed_dim

        # 核心：一个 Embedding 查找表
        # 它维护了 num_conditions 组参数，每组参数长度为 num_prompt_tokens * vit_embed_dim
        self.condition_prompts = nn.Embedding(
            num_conditions,
            num_prompt_tokens * vit_embed_dim
        )

        # 初始化策略：使用正态分布进行轻微扰动，这有助于模型在微调初期更好地收敛
        nn.init.normal_(self.condition_prompts.weight, std=0.02)

    def forward(self, condition_ids):
        """
        前向传播
        参数:
            condition_ids: 形状为 [Batch_size] 的张量，里面是 0, 1, 2 这样的条件标签
        返回:
            形状为 [Batch_size, num_prompt_tokens, vit_embed_dim] 的 Prompt Tokens
        """
        # 从 Embedding 层查找到对应的 prompt 长向量
        # 形状变为: [Batch_size, num_prompt_tokens * vit_embed_dim]
        prompts = self.condition_prompts(condition_ids)

        # 将长向量重塑（Reshape）为 ViT 需要的 [Batch, Length, Dim] 形状
        # 也就是拆分成 num_prompt_tokens 个维度为 vit_embed_dim 的 Token
        return prompts.view(-1, self.num_prompt_tokens, self.vit_embed_dim)