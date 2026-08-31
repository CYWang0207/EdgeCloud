# 模块一：边缘实时感知

## 职责

- MV-ViT 多视角推理（ViT-Small, 2200万参数, 4视角早期融合）
- AdaptFormer adapter：FFN 旁路 PEFT，主干冻结只训 adapter（已落地，8/3 验收通过）
- Token 剪枝：运行时动态保留率 k_t
- 漂移模拟与感知：校准后的合成相机漂移（BoxCars：illumination / motion_blur / sensor_noise；ModelNet40：illumination / defocus / sensor_noise）+ 香农熵 Edrift + 结构性漂移
- 云端视觉教师：大 ViT 场景分类头 + 教师缓存，代表性漂移样本不使用真实标签
- Prompt/VLM 条件注入：历史消融方案，不属于当前正式推理链路
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
├── export_boxcars_cloud_teacher_cache.py # BoxCars 云教师缓存
├── train_boxcars_cloud_teacher_adapter.py # BoxCars 无标签 Adapter 刷新
├── evaluate_boxcars_cloud_teacher.py # BoxCars 漂移对比评测
├── evaluate_boxcars_cloud_teacher_quick_test.py # 固定 checkpoint 独立 test：默认 256 条 class-stratified，--samples 0 全量 12,322 条；teacher 参数可省略跳过 6B
├── modelnet_camera_drift_dataset.py  # ModelNet40 相机退化包装
├── modelnet_cloud_teacher_refresh.py # ModelNet40 头训练/刷新/选型/终评
├── calibrate_boxcars_camera_corruptions.py # BoxCars 退化敏感度校准
├── train_boxcars_camera_adapter.py   # 校准混合 Adapter 训练
├── evaluate_boxcars_camera_adapter.py # 校准混合逐类对比
├── train_boxcars.py                  # BoxCars baseline DDP 训练（test Top-1=88.04%）
├── evaluate_boxcars.py               # BoxCars 官方评估
├── train_boxcars_retrain_drift.py    # BoxCars 漂移全量重训
├── test_boxcars_inference.py         # BoxCars→MV-ViT 前向冒烟测试
├── dataset.py                        # 场景一 ModelNet40 数据加载
├── train.py / train_retrain_drift.py # 场景一训练
├── test.py                           # 场景一评估
└── benchmarks/                       # TTFT / 内存 / 延迟 / 一体化测量
```


## 性能评测范围

- TTFT、推理延迟和 GPU 内存；
- Token 剪枝前后对比；
- ViT-Small 正式结果与 ViT-Tiny 资源对照。

## 正式云端视觉教师流程（当前主线）

当前方案不是直接使用漂移样本标签训练 Adapter。大 ViT 的场景分类头只在预先划分的离线标注数据上训练；
刷新阶段只使用教师 logits/feature、干净回放约束和边缘输出。开发集负责选择固定 checkpoint，官方 test
只用于最终一次评测。具体协议与命令见：

- `../docs/实验结果总览_20260809.md`
- `../docs/第一场景_ModelNet40.md`

下列直接使用 clean/corrupt 标签的训练命令保留为监督基线，不再代表正式提交方法。

## BoxCars 相机退化 Adapter 实验（监督基线）

漂移校正直接使用 BoxCars 的真实类别标签与 clean/corrupt 成对监督。
先在完整 validation 集校准面向监控场景的合成相机退化，再训练新的隔离权重；不要覆盖旧的
`boxcars_drift_adapters/general`。当前有效组合为非均匀曝光、方向运动模糊、低照度
传感器噪声和雾霾：

```bash
python calibrate_boxcars_camera_corruptions.py \
  --dataset-path data/BoxCars116k_kaggle/BoxCars116k \
  --baseline-checkpoint models/boxcars_make_baseline/best.pth \
  --output-json artifacts/boxcars/boxcars_camera_calibration.json

python train_boxcars_camera_adapter.py \
  --dataset-path data/BoxCars116k_kaggle/BoxCars116k \
  --baseline-checkpoint models/boxcars_make_baseline/best.pth \
  --save-dir artifacts/boxcars/supervised_camera_adapter \
  --drift-types illumination,motion_blur,sensor_noise,haze \
  --drift-weights .25 .25 .30 .20 \
  --fixed-severities illumination=1.0,motion_blur=.8,sensor_noise=.6,haze=1.0

python evaluate_boxcars_camera_adapter.py \
  --dataset-path data/BoxCars116k_kaggle/BoxCars116k \
  --baseline-checkpoint models/boxcars_make_baseline/best.pth \
  --adapter-checkpoint artifacts/boxcars/supervised_camera_adapter/best.pth \
  --corruption-specs illumination=1.0 motion_blur=.8 sensor_noise=.6 haze=1.0 \
  --output-json artifacts/boxcars/boxcars_camera_adapter_impact.json
```

## ModelNet40 相机退化 Adapter 实验（监督基线）

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
  --dataset-path data/modelnet40v2png_ori4 \
  --baseline-checkpoint models/mv_vit_token_epoch_30.pth \
  --output-json artifacts/modelnet40/modelnet40_camera_calibration.json
```

```bash
python train_modelnet_drift_adapter.py \
  --dataset-path data/modelnet40v2png_ori4 \
  --baseline-checkpoint models/mv_vit_token_epoch_30.pth \
  --save-dir artifacts/modelnet40/supervised_camera_adapter

python evaluate_modelnet_drift_adapter.py \
  --dataset-path data/modelnet40v2png_ori4 \
  --baseline-checkpoint models/mv_vit_token_epoch_30.pth \
  --adapter-checkpoint artifacts/modelnet40/supervised_camera_adapter/best.pth \
  --output-json artifacts/modelnet40/modelnet40_camera_adapter_impact.json
```
历史 Prompt 消融和临时作图脚本已移入 `../archive/`，不属于正式提交链路。
