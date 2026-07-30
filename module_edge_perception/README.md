# 模块一：边缘实时感知

## 职责

- MV-ViT 多视角推理（ViT-Small, 2200万参数, 4视角早期融合）
- Token 剪枝：运行时动态保留率 k_t
- 漂移模拟与感知：5种环境漂移 + 香农熵 Edrift
- Prompt 注入：接收云端下发 Prompt Token，插入 ViT 序列
- 性能测量：TTFT、推理延迟、GPU 内存

## 对标指标

| 指标 | 要求 | 方法 |
|------|------|------|
| 边侧参数量 | 千万级 | ViT-Small 2200万 |
| TTFT | 降低 >= 75% | Token 剪枝 |
| 推理内存 | <= 1.5GB | ViT-Small + 剪枝 |

## 目录约定

```
module_edge_perception/
├── model/          # MV-ViT 模型（EarlyFusionMultiViewViT）
├── drift/          # 漂移模拟器（DeterministicDriftWrapper）
├── prompt/         # PromptGenerator + train_prompt
├── benchmarks/     # TTFT / 内存 / 延迟测量脚本
└── train/          # 训练脚本
```

## 代码来源

核心代码由学长提供，位于本地 D:\challenge\Code\code2\，包括：
- model.py — MV-ViT 模型
- dataset.py — ModelNet40 数据加载
- sevd_dataset.py — 第二场景 SEVD 四路同步 RGB + COCO 标注加载
- test_sevd_inference.py — SEVD 到现有 MV-ViT 的无训练前向链路验证
- drift_dataset.py — 漂移模拟器
- prompt_tuning/prompt_model.py — Prompt 生成器
- train.py / train_retrain_drift.py / train_token_prompt.py — 训练脚本
- test.py / evaluate_train_set.py — 评估脚本

## 负责人任务

- 测量 TTFT、推理延迟、GPU 内存
- Token 剪枝前后对比实验
- 换 ViT-Tiny（如需更小模型）只需改一行 model_name 参数
