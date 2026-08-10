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
├── verify_adaptformer.py             # AdaptFormer 三点验收脚本（零初始化/参数量/三条前向）
├── boxcars_dataset.py                # 场景二 BoxCars116k 数据加载（4逻辑视图 + view_mask）
├── boxcars_drift_dataset.py          # BoxCars 旧合成漂移包装（保留可复现）
├── boxcars_camera_drift_dataset.py   # BoxCars 监控相机退化包装（当前主线）
├── calibrate_boxcars_camera_corruptions.py # BoxCars 退化敏感度校准
├── train_boxcars_camera_adapter.py   # 校准混合 Adapter 训练
├── evaluate_boxcars_camera_adapter.py # 校准混合逐类对比
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

## BoxCars 相机退化 Adapter 实验（当前主线）

漂移校正直接使用 BoxCars 的真实类别标签与 clean/corrupt 成对监督。
先在完整 validation 集校准真实监控退化，再训练新的隔离权重；不要覆盖旧的
`boxcars_drift_adapters/general`。当前有效组合为非均匀曝光、方向运动模糊、低照度
传感器噪声和雾霾：

```bash
python calibrate_boxcars_camera_corruptions.py \
  --dataset-path /root/autodl-tmp/EdgeCloudRuntime/shared/data/BoxCars116k_kaggle/BoxCars116k \
  --baseline-checkpoint checkpoints/boxcars_make_baseline/best.pth \
  --output-json outputs/boxcars_camera_calibration.json

python train_boxcars_camera_adapter.py \
  --dataset-path /root/autodl-tmp/EdgeCloudRuntime/shared/data/BoxCars116k_kaggle/BoxCars116k \
  --baseline-checkpoint checkpoints/boxcars_make_baseline/best.pth \
  --save-dir checkpoints/boxcars_drift_adapters/camera_mixture_calibrated \
  --drift-types illumination,motion_blur,sensor_noise,haze \
  --drift-weights .25 .25 .30 .20 \
  --fixed-severities illumination=1.0,motion_blur=.8,sensor_noise=.6,haze=1.0

python evaluate_boxcars_camera_adapter.py \
  --dataset-path /root/autodl-tmp/EdgeCloudRuntime/shared/data/BoxCars116k_kaggle/BoxCars116k \
  --baseline-checkpoint checkpoints/boxcars_make_baseline/best.pth \
  --adapter-checkpoint checkpoints/boxcars_drift_adapters/camera_mixture_calibrated/best.pth \
  --corruption-specs illumination=1.0 motion_blur=.8 sensor_noise=.6 haze=1.0 \
  --output-json outputs/boxcars_camera_adapter_impact.json
```

## ModelNet40 相机退化 Adapter 实验

ModelNet40 的规整渲染图对高频噪声特别敏感，但正式 Adapter 不应成为单一噪声补丁。完成全 test
校准后，当前训练混合为曝光/伽马+色偏+局部阴影（30%，固定 `1.0`）、失焦（30%，固定 `.2`）和
每视图独立 Poisson-Gaussian 传感器噪声（40%，固定 `.4`）。compression 的校准下降过强，
partial occlusion 影响不足，二者均未混入。它们替代了全局 bright/dark 倍率和四视图同形噪声的玩具式干预。

severity 始终保持在 `[0, 1]`，并保留 20% 干净样本。每个场景都必须先依据完整 baseline 矩阵
校准各类的参数映射，而不是扩张 severity 定义域或复用另一数据域的档位。

训练前必须先用冻结 baseline 在完整 test 上扫描，并将每类固定到最接近 10pp 准确率下降的参数；若某类
在整个 `[0,1]` 网格都不能达到目标，先改该类生成映射，而不是带着未校准干预训练：

```bash
python calibrate_modelnet_camera_corruptions.py \
  --dataset-path /root/autodl-tmp/EdgeCloudRuntime/shared/data/modelnet40v2png_ori4 \
  --baseline-checkpoint checkpoints/mv_vit_token_epoch_30.pth \
  --output-json outputs/modelnet40_camera_calibration.json
```

```bash
python train_modelnet_drift_adapter.py \
  --dataset-path /root/autodl-tmp/EdgeCloudRuntime/shared/data/modelnet40v2png_ori4 \
  --baseline-checkpoint checkpoints/mv_vit_token_epoch_30.pth \
  --save-dir checkpoints/modelnet40_drift_adapters/camera_mixture

python evaluate_modelnet_drift_adapter.py \
  --dataset-path /root/autodl-tmp/EdgeCloudRuntime/shared/data/modelnet40v2png_ori4 \
  --baseline-checkpoint checkpoints/mv_vit_token_epoch_30.pth \
  --adapter-checkpoint checkpoints/modelnet40_drift_adapters/camera_mixture/best.pth \
  --output-json outputs/modelnet40_camera_adapter_impact.json
```
