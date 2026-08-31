# 第二场景：BoxCars116k 交通车辆识别

> 本文是本数据集的结果文档，数值来自云端 `adjust` 分支 2026-08-09/08-10 实际运行并回传的
> `local/results/boxcars_cloud_teacher_development_20260809/`、
> `local/results/boxcars_cloud_teacher_quick_test_20260809/` 与
> `local/results/boxcars_cloud_teacher_full_test_20260810/`，与交付包
> `local/delivery/boxcars_recovery_metrics_20260810/` 同源。方法要点与总体结论见
> [`实验结果总览_20260809.md`](实验结果总览_20260809.md)；完整方法设计归档于
> `local/archive/status_docs_20260810/云端视觉教师Adapter方案_20260809.md`。

## 场景定义

BoxCars116k 完成交通监控车辆品牌识别，16 类品牌，默认官方 `make` 划分。一个样本由同一辆车
轨迹中时间跨度尽可能大的 4 张裁剪图组成，输入形状为 `[4, 3, 224, 224]`。

需要注意：BoxCars116k 不提供跨摄像头车辆身份关联，因此这里的 4 张图是**同一物理摄像头**下的
4 个逻辑视图，不能表述为四台摄像头同时拍摄同一车辆。少于 4 张图的轨迹会重复末张图补齐，并
通过 `view_mask` 标记无效视图。

## 与项目架构的对应关系

- MV-ViT 融合同一车辆不同时间、尺度和可见面的观测；
- Token 剪枝用于动态降低多视图推理开销；
- 漂移模块模拟明暗、模糊、噪声等环境变化；
- 云端视觉教师仅在 `u=1` refresh 时为漂移样本产生监督；边缘运行无条件 AdaptFormer；
- `view_mask` 可表达轨迹补齐或视图故障。

既有 **clean 官方 test** baseline：Top-1 **88.04%**、Top-5 97.13%。注意这是 clean 官方 test
指标，与下文 649 条 dev / 256 条 quick test 的**三漂移平均**不是同一指标，不能直接相减或混写。

## 方法

与 ModelNet40 相同的正式新方法：冻结 `OpenGVLab/InternViT-6B-224px`，在其 3,200D track feature
上训练 16 类 task head，再以 task logits / visual features / clean anchor 监督约 0.3M 参数的
AdaptFormer；正式 cloud-unlabeled refresh 的 CE 权重为 0，不读取上传漂移样本的真实标签。
方法要点与完整公式见 [`实验结果总览_20260809.md`](实验结果总览_20260809.md) 的"方法"一节。

## Task head 消融（为什么需要 noise-weighted MLP）

冻结 6B 主干，在已有 13,098 条 train 特征缓存（含 4,412 条 noise 轨迹，随机强度 0.3–1.0）上
训练四种 head 候选，准入门槛 `clean ≥ 92.8%` 且 `noise ≥ 79.5%`（`noise_head_metrics.json`）：

| Head | Clean | Illumination 1.0 | Blur 0.8 | Noise 0.6 | 是否通过 |
|---|---:|---:|---:|---:|---|
| general linear | 93.37% | 90.91% | 89.37% | 73.34% | 否（noise） |
| noise-weighted linear | 93.99% | 91.06% | 89.68% | 77.35% | 否（noise） |
| noise specialist linear | 92.14% | 87.52% | 83.05% | 71.49% | 否（noise） |
| **noise-weighted MLP（选中）** | **96.15%** | **94.92%** | **91.99%** | **82.59%** | **是** |

较早的 mixed-drift linear head（`head_metrics.json`，best mean drift 84.44%）在 noise 上只有
73.19%，不能担任三漂移 teacher。上述消融在既有缓存上完成，`noise_weighted_mlp` 同时满足
clean 与 noise 门槛，best epoch 118，被固定为 `selected_head.pt`。这一步是**训练任务 head**，
不是微调 6B backbone。

## 开发集证据（649 条 validation，固定强度 illum 1.0 / blur 0.8 / noise 0.6）

head 和 Adapter checkpoint 均在此协议上选择，所以只能称为**开发集**结果：

| 模型 | Clean | Illumination 1.0 | Motion blur 0.8 | Sensor noise 0.6 | 漂移平均 |
|---|---:|---:|---:|---:|---:|
| Edge baseline | 93.84% | 81.82% | 84.59% | 79.51% | 81.97% |
| InternViT-6B + selected task head | 96.15% | 94.92% | 91.99% | 82.59% | 89.83% |
| label-only Adapter（有标签上界） | 94.30% | 83.05% | 87.52% | 87.67% | 86.08% |
| cloud hybrid Adapter（消融） | 94.30% | 83.36% | 86.75% | 86.90% | 85.67% |
| **cloud-unlabeled Adapter（正式 u=1）** | **94.61%** | **83.05%** | **86.13%** | **86.44%** | **85.21%** |

三组 Adapter 使用相同抽样顺序和优化配置（4 epoch、每 epoch 256 batch ≈ 1,024 条代表轨迹、
seed 42）；差异只在 loss 权重：label-only（CE 1.0）、hybrid（CE 1.0 + KD 0.35 + feature 0.25 +
anchor 0.2）、cloud-unlabeled（CE 0 + KD 0.7 + feature 0.3 + anchor 0.2）。总结性 gate 见
`summary.json`：

- **unlabeled refresh test：PASS**。相对未适配 Edge 三漂移平均 +3.24 pp；距有标签 label-only
  上界仅 0.87 pp；clean 保持且三种漂移均不低于 Edge。
- **hybrid additive test：FAIL**。有标签 CE 时叠加当前 KD+feature 反而比 label-only 低 0.41 pp，
  因此不能宣称 cloud teacher 在有标签训练上必然增益；它的价值是替代真实环境中缺失的人工标签。

## 独立 quick test（官方 test split，固定 checkpoint）

官方 `test` split 有 12,322 条且类别极不均衡（Porsche 仅 3 条）。为控制 6B 单卡推理时间，
固定 seed `20260809` 抽取 **256 条 class-stratified 子集**（每类至少一条，其余按官方类别容量
分配），固定 checkpoint 后对 Edge baseline、cloud-unlabeled Adapter、InternViT teacher 三个模型
做完全不训练的独立检查。Edge 与 Adapter 使用同一组样本、同一 corruption seed，逐样本配对。
评估耗时 288 秒，产物含 3,072 条逐样本预测（256 × 4 条件 × 3 模型）。

| 模型 | Clean | Illumination 1.0 | Motion blur 0.8 | Sensor noise 0.6 | 漂移平均 |
|---|---:|---:|---:|---:|---:|
| Edge baseline | 85.55% | 74.61% | 73.83% | 69.92% | 72.79% |
| **cloud-unlabeled Adapter** | **87.50%** | **76.17%** | **74.22%** | **77.73%** | **76.04%** |
| InternViT-6B teacher | 92.97% | 91.41% | 91.02% | 80.86% | 87.76% |

对同一条样本的 Edge→Adapter 差异做 2,000 次 paired bootstrap：

| 条件 | Adapter − Edge | 95% CI |
|---|---:|---:|
| Clean | +1.95 pp | [−0.39, +4.69] pp |
| Illumination | +1.56 pp | [−1.17, +4.30] pp |
| Motion blur | +0.39 pp | [−2.73, +3.12] pp |
| Sensor noise | +7.81 pp | **[+3.91, +11.72] pp** |

结论：

1. 独立 test 子集复现了开发集方向：cloud-unlabeled 三漂移平均提升 **+3.26 pp**（与开发集
   +3.24 pp 一致），clean 也提升 +1.95 pp。
2. teacher 在四个条件均明显高于 Edge（平均漂移高 14.97 pp），有资格承担云端监督。
3. 最大且区间不跨 0 的改善来自 sensor noise（+7.81 pp）；illumination / blur 的点估计为正，但
   256 条快速样本上 CI 仍跨 0，不宣称单项目已强统计显著。

边界：这是 12,322 条 official test 中的 256 条快速独立检查，样本量不足以让 illumination / blur
的单项结论达到强统计显著；全量官方 test 结论见下一节。

## 完整官方 test（12,322 条，固定 checkpoint）

全量跑在 official test 全部 12,322 条 × 4 条件上，固定开发阶段 checkpoint（Edge baseline 与
`cloud_unlabeled/best.pth`），未按 test 重选 head、epoch 或 loss。评估只覆盖主系统对照所需的
Edge / cloud-unlabeled 两组模型（跳过 6B teacher 推理），耗时约 7 分钟，产物为 98,576 条逐
样本配对预测（12,322 × 4 × 2）。

| 模型 | Clean | Illumination 1.0 | Motion blur 0.8 | Sensor noise 0.6 | 漂移平均 |
|---|---:|---:|---:|---:|---:|
| Edge baseline | 88.04% | 74.03% | 77.57% | 71.77% | 74.46% |
| **cloud-unlabeled Adapter** | **88.09%** | **78.45%** | **79.51%** | **79.10%** | **79.02%** |

同一样本的 Edge→Adapter 配对 bootstrap（2,000 次）：

| 条件 | Adapter − Edge | 95% CI |
|---|---:|---:|
| Clean | +0.06 pp | [−0.20, +0.31] pp |
| Illumination | +4.42 pp | [+3.95, +4.90] pp |
| Motion blur | +1.94 pp | [+1.57, +2.31] pp |
| Sensor noise | +7.33 pp | [+6.77, +7.90] pp |

结论：

1. 全量官方 test 上，无标签 Adapter 三漂移平均提升 **+4.56 pp**，且**三个漂移条件的 95% CI 全部
   不跨 0**——补齐了 256 条快速检查中 illumination / blur 区间跨 0 的限制。
2. clean 基本持平（+0.06 pp，CI 覆盖 0 两侧），没有 clean 保持代价；与 256 条小样本上的 +1.95 pp
   点估计不同，正式口径以全量为准。
3. 最大增益仍来自 sensor noise（+7.33 pp），与开发阶段必须训练 noise-weighted head 的发现一致。

运行入口 `evaluate_boxcars_cloud_teacher_quick_test.py --samples 0`，可省略 teacher 参数跳过
6B 推理（本次即此配置）。完整证据位于 `local/results/boxcars_cloud_teacher_full_test_20260810/`。

## VLM 消融（旧路线，负向消融）

历史 Qwen3-VL 路线已归档为消融（`local/archive/vlm_conditioned_20260809/`），不进入当前 edge
forward。BoxCars validation 同协议消融（2026-08-08）：

| 方法 | Clean | Illumination | Motion blur | Sensor noise | Mean drift |
|---|---:|---:|---:|---:|---:|
| camera-mixture Adapter（second baseline） | 94.14% | 88.60% | 89.98% | 91.06% | 89.88% |
| condition Adapter，无 VLM | 93.84% | 88.91% | 90.29% | 91.53% | 90.24% |
| condition Adapter，VLM cache | 93.68% | 89.06% | 90.60% | 90.60% | 90.09% |

VLM cache 只在 illumination / blur 上增加约 0.15 / 0.31 pp，却使 noise 下降约 0.92 pp，
三漂移平均反而比无 VLM 低约 0.15 pp，不能归因于 VLM。且 Qwen3-VL 16 类 soft-label Top-1 仅
62.25%，远低于既有 Edge clean baseline 88.04%。因此旧路线被保留为负向消融，不恢复。

## 已验证的工程约束

| 项目 | 已验证事实 |
|---|---|
| 云端 teacher | `OpenGVLab/InternViT-6B-224px`，冻结、BF16、仅云端离线/刷新期使用 |
| Edge 模型 | `EarlyFusionMultiViewViT(vit_small_patch16_224)`，4 个逻辑视图 |
| Adapter | `r=32` 的 AdaptFormer，299,916 参数 |
| 下发大小 | 实测 `1,216,745 bytes`，约 1.2 MB |
| 下发内容 | 仅 Adapter 参数；不含 backbone、norm、head、teacher 或训练期 projector |
| 数据对齐 | teacher cache 保存原始 index、drift type、severity；训练时逐 batch 校验 |
| clean 保持 | anchor 将“开启 Adapter 的 clean replay”与“关闭 Adapter 的冻结 baseline”对齐 |

## 文件与证据位置

| 内容 | 路径 |
|---|---|
| 开发集证据（pipeline summary + head 消融 metrics） | `local/results/boxcars_cloud_teacher_development_20260809/` |
| 独立 quick test 证据（summary + 3,072 条 predictions） | `local/results/boxcars_cloud_teacher_quick_test_20260809/` |
| **完整官方 test 证据（summary + 98,576 条 predictions）** | `local/results/boxcars_cloud_teacher_full_test_20260810/` |
| 轻量交付（README + 可画图 JSON，能力保持曲线） | `local/delivery/boxcars_recovery_metrics_20260809/` |
| 最终下发 Adapter | `models/boxcars_cloud_teacher_adapter_20260809/cloud_unlabeled/best.pth` |
| 旧 VLM 消融归档 | `local/archive/vlm_conditioned_20260809/` |
| 主要代码 | `module_edge_perception/`（`boxcars_dataset.py`、`train_boxcars.py`、`evaluate_boxcars.py`、`boxcars_camera_drift_dataset.py`、`export_boxcars_cloud_teacher_cache.py`、`train_boxcars_cloud_teacher_adapter.py`、`evaluate_boxcars_cloud_teacher_quick_test.py`） |

数据集不在 Git 中重复分发；官方来源、目录结构和准备方法见 `data/README.md`。正式权重通过 GitHub Release 提供，版本与校验值见 `models/README.md`。
