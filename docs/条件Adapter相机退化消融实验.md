# 条件 Adapter 相机退化消融实验

本实验把 adjust 作为与 second 的 camera-mixture Adapter 同一实验设置下的模型消融，
不重新设计 VLM 专用退化，也不重调 severity。

## 固定实验口径

| 数据集 | 数据划分 | 相机退化 | 固定 severity | 随机种子 |
| --- | --- | --- | --- | --- |
| BoxCars116k | train / validation | illumination, motion_blur, sensor_noise | 1.0, 0.8, 0.6 | 42 |
| ModelNet40 | train / test | illumination, defocus, sensor_noise | 1.0, 0.2, 0.4 | 42 |

训练、验证、baseline checkpoint、混合权重 `(0.3, 0.3, 0.4)` 与 second 保持一致。

## 三组消融

1. second 的 camera-mixture Adapter 基线。
2. 条件 Adapter，不提供 VLM condition cache。
3. 条件 Adapter，提供 VLM condition cache。

VLM cache 由**施加退化后的四视图**离线导出。Qwen3-VL 只加载视觉编码器，避免语言
主干量化内核进入边缘链路；cache 的 `source_metadata` 会在训练前校验数据集、退化类型、
固定强度和 seed，避免误用干净图或旧合成漂移缓存。

## 入口

| 用途 | BoxCars | ModelNet40 |
| --- | --- | --- |
| 相机退化数据 | `boxcars_camera_drift_dataset.py` | `modelnet_camera_drift_dataset.py` |
| 条件训练 | `train_boxcars_drift_adapter.py` | `train_modelnet_condition_adapter.py` |
| VLM 特征导出 | `export_boxcars_vlm_hidden_states.py` | `export_modelnet_vlm_hidden_states.py` |
| 评估 | `evaluate_boxcars_drift_adapters.py` | `evaluate_modelnet_condition_adapter.py` |

## 已完成结果

BoxCars 三组对照均已完成 validation 评测，权重、manifest、日志和 JSON 已归档到
`results/boxcars_camera_ablation_20260808/README.md`。无 VLM 条件 Adapter 相对原始 baseline
的三项校准退化提升为 illumination +7.09 pp、motion blur +5.70 pp、sensor noise +12.02 pp；
VLM cache 条件 Adapter 的对应提升为 +7.24 pp、+6.01 pp、+11.09 pp。与无 VLM 条件 Adapter
相比，VLM 在 illumination 和 motion blur 上只有 +0.15 pp / +0.31 pp，sensor noise 下降
0.92 pp，三类退化平均下降约 0.15 pp。因此主方案采用无 VLM 条件 Adapter，VLM 作为消融
结果保留。

本轮按交付范围只完成 BoxCars 的 VLM 验证；ModelNet 的无 VLM checkpoint 已保留，但没有
继续执行 ModelNet VLM cache 导出和训练。
