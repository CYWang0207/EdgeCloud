# ModelNet40 Adapter 相机退化实验记录（2026-08-06）

## 目标

在 ModelNet40 多视图分类场景训练 AdaptFormer adapter，并验证它能否在具有实际影响、但不过度失真的相机退化下恢复冻结 MV-ViT baseline 的识别能力。

## 产物

- Adapter 权重：`shared/checkpoints/modelnet40_drift_adapters/camera_mixture_calibrated/best.pth`
- 对比结果：`shared/outputs/modelnet40_camera_adapter_20260806/adapter_impact.json`
- 训练日志：`shared/outputs/modelnet40_camera_adapter_20260806/train_calibrated.log`
- 评测日志：`shared/outputs/modelnet40_camera_adapter_20260806/evaluate.log`

权重为 adapter-only 格式，包含 64 个 adapter/norm/head 参数项，大小约 1.3 MB；SHA-256：
`2c9a4891723454458bbb5b083c611c90c5c52dc37cafcef2560e351ae1307583`。

## 代码改动

1. 新增 `module_edge_perception/train_modelnet_drift_adapter.py`：
   - 冻结 ModelNet40 MV-ViT baseline；
   - 在每个 FFN 旁路挂载 AdaptFormer（r=32）；
   - 仅训练 adapter、norm、head；
   - 使用干净/退化成对前向，优化分类损失、logit 一致性与特征对齐损失；
   - 导出可用于 u=1 下发的 adapter-only checkpoint。
2. 新增 `module_edge_perception/calibrate_modelnet_camera_corruptions.py`：
   - 在完整 ModelNet40 test 集上先扫描 baseline 的退化敏感度；
   - 退化达不到有效下降或下降过猛时先调整映射，避免直接训练未校准的合成扰动。
3. 新增 `module_edge_perception/evaluate_modelnet_drift_adapter.py`：
   - 在同一完整 test 集、同一退化参数下，对比 baseline 与 adapter；
   - 输出 clean 保持与各退化下的恢复量 JSON。
4. 更新 `module_edge_perception/README.md`：补充 ModelNet40 相机退化 adapter 的训练、校准与评测说明。

## 干预设计与校准

未采用简单的全局亮度倍乘，也未把 severity 扩展到 `[0,1]` 之外。初始扫描发现：旧 illumination 影响过弱，而旧噪声在低 severity 下又过强。因此改为更接近相机/渲染链路的三类干预，并在完整 test 集上固定到中等、有效的强度：

| 退化 | 生成方式 | 固定 severity | Baseline 相对 clean 的下降 |
|---|---|---:|---:|
| illumination | 曝光/伽马、色温偏移和局部阴影 | 1.0 | 7.58pp |
| defocus | 失焦模糊 | 0.2 | 7.62pp |
| sensor noise | 按视图独立的 Poisson-Gaussian 传感器噪声 | 0.4 | 12.16pp |

未验证出足够影响的 compression 与 partial-occlusion 没有进入正式训练混合，避免对无效干预进行“凑数”训练。

## 训练配置

```text
数据集：ModelNet40，train=9,843，test=2,468，多视图数=4
基座权重：checkpoints/mv_vit_token_epoch_30.pth
Adapter：AdaptFormer，r=32，299,916 个 adapter 参数
训练轮数：8
batch size：16
退化采样权重：illumination/defocus/sensor_noise = 0.3/0.3/0.4
退化强度：1.0/0.2/0.4（均为已校准的固定值）
干净样本比例：20%
```

## 完整测试集结果

| 条件 | Baseline | Adapter | Adapter 相对 Baseline |
|---|---:|---:|---:|
| clean | 96.76% | 96.47% | -0.28pp |
| illumination=1.0 | 89.18% | 95.14% | +5.96pp |
| defocus=0.2 | 89.14% | 96.43% | +7.29pp |
| sensor-noise=0.4 | 84.60% | 95.66% | +11.06pp |

## 结论

- 三个校准退化均使 baseline 出现约 8–12pp 的可观下降，且没有使用失真的超范围 severity；
- adapter 在三类退化上分别恢复 5.96pp、7.29pp、11.06pp；
- clean 准确率仅下降 0.28pp，说明恢复并非以明显损害干净域能力为代价；
- 该 `best.pth` 可作为 ModelNet40 场景的 adapter 下发权重，与 BoxCars 场景权重并列存放。
