# 当前 worktree 约定

本 worktree 专门验证并实现：

`InternViT-6B frozen backbone -> BoxCars task head -> teacher logits/features -> MV-ViT-S AdaptFormer refresh`

- 不恢复 VLM-conditioned Adapter、Qwen 品牌 soft labels 或 Prompt tuning 主线。
- InternViT 只在云端离线使用，不进入边缘 forward，也不随 adapter 下发。
- prototype 只作五分钟 sanity check；正式 teacher 必须训练 linear/MLP 分类头并在独立漂移验证集评估。
- teacher 未显著超过对应 edge drift baseline 前，不开始 Adapter 蒸馏。
- 旧代码和旧文档统一放在 `local/archive/`，主目录保持当前路线整洁。
