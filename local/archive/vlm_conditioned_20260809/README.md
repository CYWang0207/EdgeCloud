# 已归档：VLM-conditioned Adapter 路线（2026-08-09）

此目录保存被替代、但仍可复核的实验材料；它们不是当前训练或答辩的正式证据。

- `docs/`：旧 VLM-conditioned 架构、Qwen3-VL 品牌 soft-label、条件 Adapter 消融，以及其对应的阶段性 general Adapter 实验记录。
- `results/`：对应 BoxCars 条件 Adapter checkpoint、日志、评估 JSON 与 VLM condition cache 导出日志。

归档原因：Qwen3-VL 的语义 condition 对 illumination、motion blur、sensor noise 等低层相机漂移没有稳定正贡献；三类退化相对无 VLM condition Adapter 的平均变化约为 -0.15 pp。正式路线已改为云端视觉教师生成 task logits / visual features，云端仅更新并下发 edge AdaptFormer。详见 `docs/云端视觉教师Adapter方案_20260809.md`。
