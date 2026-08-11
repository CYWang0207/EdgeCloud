# BoxCars116k 相机退化 Adapter 实验记录（2026-08-06）

## 目标

在真实交通监控车辆品牌识别场景中，先验证相机退化确实会使冻结 MV-ViT baseline
明显下降，再训练一个小型、场景专用的 AdaptFormer 恢复识别。该实验不改动原来的
`boxcars_drift_adapters/general` 权重或旧合成漂移代码。

## 为什么需要 BoxCars 专用配置

每个 BoxCars 样本是**同一物理交通摄像机**拍到的一段车辆轨迹，不是 ModelNet 那样的
四个干净渲染视图。真实问题主要来自夜间/逆光曝光、车辆运动模糊、低照度传感器 gain
和雾霾；同名的 noise 也应有轨迹级的相机设置、帧级的随机观测。故本实验不复用
ModelNet 的失焦/渲染噪声参数，也不再采用全局亮暗倍乘和白高斯噪声。

新增文件：

- `module_edge_perception/boxcars_camera_drift_dataset.py`：相机合理的退化生成器；
- `module_edge_perception/calibrate_boxcars_camera_corruptions.py`：baseline 全集扫描；
- `module_edge_perception/train_boxcars_camera_adapter.py`：冻结主干的成对训练；
- `module_edge_perception/evaluate_boxcars_camera_adapter.py`：固定退化逐类对比。

## 校准

环境为 a3 NVIDIA RTX 3090，BoxCars `make` validation=649，clean baseline=93.84%。
在完整 validation 集上扫描后，正式训练选用下列仍像真实监控、且能造成约 7–15pp
下降的档位：

| 退化 | 生成方式 | severity | Baseline | 相对 clean 下降 |
|---|---|---:|---:|---:|
| illumination | 夜间/逆光曝光、色温、局部阴影 | 1.0 | 81.82% | 12.02pp |
| motion blur | 车辆运动/快门方向模糊 | 0.8 | 84.59% | 9.24pp |
| sensor noise | 低照度 ISO/gain 的 Poisson-Gaussian | 0.6 | 79.51% | 14.33pp |
| haze | 雾霾大气散射 | 1.0 | 86.90% | 6.93pp |

也测试了 compression、单帧 partial occlusion、glare 和 lens obstruction；它们在当前
四视图模型上只造成约 0–2.3pp 下降，因此没有被等权塞入训练混合。

## 训练

```text
train=13,098，validation=649，task=make，view=4
Adapter=AdaptFormer r=32，299,916 参数，冻结 MV-ViT 主干
epochs=8，batch size=8，AdamW lr=2e-4，weight decay=0.05，bf16
训练混合：illumination / motion_blur / sensor_noise / haze
混合权重：0.25 / 0.25 / 0.30 / 0.20
固定 severity：1.0 / 0.8 / 0.6 / 1.0
clean 概率：20%
```

最佳 checkpoint 出现在 epoch 6，混合 validation=90.91%。

## 完整 validation 评测

评测使用相同 validation 全集、相同随机种子、同一批固定退化；加载的是新目录的
`best.pth`。

| 条件 | Baseline | Adapter | Adapter 相对 Baseline |
|---|---:|---:|---:|
| clean | 93.84% | 94.14% | +0.31pp |
| illumination=1.0 | 81.82% | 88.60% | +6.78pp |
| motion-blur=0.8 | 84.59% | 89.98% | +5.39pp |
| sensor-noise=0.6 | 79.51% | 91.06% | +11.56pp |
| haze=1.0 | 86.90% | 93.37% | +6.47pp |

四类退化均得到恢复，且干净域没有回退。

## 产物与隔离验证

- 新权重：`shared/checkpoints/boxcars_drift_adapters/camera_mixture_calibrated/best.pth`
  - SHA-256：`f94c3d83881c7485a06ed91e411bfc6fd3c4723eea2e3129304ad268d75768d6`
- 新评测：`shared/outputs/boxcars_camera_adapter_20260806/adapter_impact.json`
- 本地下载：`models/boxcars_camera_mixture_calibrated/best.pth` 和
  `results/boxcars_camera_adapter_20260806/`
- 保留的旧权重：`shared/checkpoints/boxcars_drift_adapters/general/best.pth`
  - SHA-256：`93c71a73b78cebd208df107b29150870b23950f03c8ee33b89b10f5cbe27dc92`

新旧路径及 SHA-256 均不同；本次没有覆盖旧 Cars Adapter。
