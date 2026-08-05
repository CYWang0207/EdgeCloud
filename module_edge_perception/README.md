# 模块一：边缘实时感知

## 职责

- MV-ViT 多视角推理（ViT-Small, 2200万参数, 4视角早期融合）
- AdaptFormer adapter：FFN 旁路 PEFT，主干冻结只训 adapter（已落地，8/3 验收通过）
- Token 剪枝：运行时动态保留率 k_t
- 漂移模拟与感知：5种环境漂移 + 香农熵 Edrift + 结构性漂移
- Prompt 注入：旧方案遗留，作"环境漂移快响应"可选辅助（存在两套实现，待清理）
- 性能测量：TTFT、推理延迟、GPU 内存

## 对标指标

| 指标 | 要求 | 方法 |
|------|------|------|
| 边侧参数量 | 千万级 | ViT-Small 2200万 |
| TTFT | 降低 >= 75% | Token 剪枝 |
| 推理内存 | <= 1.5GB | ViT-Small + 剪枝 |

## 目录约定（实际为扁平结构，非子目录）

```
module_edge_perception/
├── model.py                          # MV-ViT 模型（EarlyFusionMultiViewViT）
├── adaptformer.py                    # AdaptFormer PEFT 模块（已落地，8/3 验收通过）
├── train_adapter.py                  # Adapter 蒸馏训练（云端软标签→只训 adapter）
├── verify_adaptformer.py             # AdaptFormer 三点验收脚本（零初始化/参数量/三条前向）
├── boxcars_dataset.py                # 场景二 BoxCars116k 数据加载（4逻辑视图 + view_mask）
├── boxcars_drift_dataset.py          # BoxCars 漂移包装
├── train_boxcars.py                  # BoxCars baseline DDP 训练（test Top-1=88.04%）
├── evaluate_boxcars.py               # BoxCars 官方评估
├── train_boxcars_retrain_drift.py    # BoxCars 漂移全量重训
├── train_boxcars_token_prompt.py     # BoxCars Prompt 适配训练（过渡）
├── test_boxcars_inference.py         # BoxCars→MV-ViT 前向冒烟测试
├── dataset.py                        # 场景一 ModelNet40 数据加载
├── train.py / train_retrain_drift.py / train_token_prompt.py  # 场景一训练
├── test.py / evaluate_train_set.py   # 场景一评估
├── prompt_tuning/                    # PromptGenerator（可选辅助，存在两套实现待清理）
└── benchmarks/                       # TTFT / 内存 / 延迟 / 一体化测量
```


## 负责人任务

- 测量 TTFT、推理延迟、GPU 内存
- Token 剪枝前后对比实验
- 换 ViT-Tiny（如需更小模型）只需改一行 model_name 参数
