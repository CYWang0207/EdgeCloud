# 第一场景：ModelNet40 大 ViT 云教师

> 本文是本数据集的结果文档，数值来自云端 `adjust` 分支 2026-08-09 实际运行并回传的
> `local/results/modelnet40_cloud_teacher_full_test_20260809/`，与交付包
> `local/delivery/modelnet40_recovery_metrics_20260809/` 同源。方法要点与总体结论见
> [`实验结果总览_20260809.md`](实验结果总览_20260809.md)；完整方法设计归档于
> `local/archive/status_docs_20260810/云端视觉教师Adapter方案_20260809.md`。

## 场景定义

ModelNet40 是白底、光滑、轮廓主导的规范化 CAD 渲染，40 类。本场景按官方 `train` / `test`
划分，一个样本是同一物体四个视角组成的轨迹，形状为 `[4, 3, 224, 224]`。四个视角之间包含
互补的形状信息，因此模型保留每个视角的 3,200D InternViT 表示，**逐视角分类后再平均 track
logits**，而不是在进入 task head 前直接平均四视图 feature。

## 最终路线

本场景采用与 BoxCars 相同的正式新方法：

```text
冻结 InternViT-6B
  → ModelNet40 离线标签训练新的 40 类 task head
  → 独立 dev 上执行 teacher gate
  → 代表性漂移轨迹生成 teacher logits / visual features
  → 不读取 refresh 标签，训练 Edge AdaptFormer
  → 固定 checkpoint 后评估完整 official test
```

InternViT-6B 主干始终冻结，没有被任何场景回写；BoxCars 的 16 类 task head 不复用，
ModelNet40 使用独立的 40 类 head。

## 数据划分

官方 train 共 9,843 条，按类确定性拆为三份互斥集合，解决类别数 64 → 889 的长尾差异：

| 用途 | 每类条数 | 实际条数 | 说明 |
|---|---:|---:|---|
| task head 离线训练 | 余量 | **9,043** | 使用原 train 标签，同时看到 clean 与随机 illumination / defocus / sensor-noise feature |
| refresh（Adapter 训练） | 12 | **480** | cloud-unlabeled loss 不读取这 480 条轨迹的真实标签 |
| dev（选模 + teacher gate） | 8 | **320** | 与 head、refresh 互斥 |

所有 head、loss profile 与 Adapter checkpoint 都在 train-internal dev 上固定，
最后才运行 2,468 条 official test。

## Teacher gate（320 条 dev）

| 模型 | Clean | Illumination | Defocus | Sensor noise | Mean drift |
|---|---:|---:|---:|---:|---:|
| Edge baseline | 100.000% | 92.188% | 95.000% | 88.750% | 91.979% |
| InternViT-6B + 40 类 head | 97.188% | 95.000% | 97.188% | 94.375% | **95.521%** |

teacher mean drift 领先 **3.542 pp**，准入通过。数值来自 `teacher_gate.json` /
`teacher_head_metrics.json`（best epoch 37，`best_mean_drift=0.955208`）。

## 完整 official test（2,468 条四视图轨迹）

这是完整 `test` split，不是 quick subset。同一 split、同一 corruption seed 下逐样本配对评估：

| 方法 | Clean | Illumination | Defocus | Sensor noise | Mean drift |
|---|---:|---:|---:|---:|---:|
| Edge baseline | 96.759% | 89.141% | 88.412% | 84.400% | 87.318% |
| Cloud unlabeled（初始） | 95.827% | 88.898% | 93.598% | 91.370% | 91.289% |
| Label-only upper bound | 96.313% | 90.032% | 94.692% | 93.152% | 92.625% |
| Hybrid | 96.070% | 90.113% | 94.368% | 92.707% | 92.396% |
| **Cloud unlabeled（最终 illumination 专项版）** | **95.381%** | **90.154%** | **94.327%** | **92.788%** | **92.423%** |

本数据集实际评估了**多种方法对照**，不是只有一种：`cloud_unlabeled`（无上传标签主系统）、
`label_only`（有标签监督上界）、`hybrid`（CE+teacher 同时使用的消融），以及后续
illumination 专项调优得到的最终 `cloud_unlabeled_illumination_tuned`。

### 最终专项版相对 Edge baseline

| 条件 | 提升 | 2,000 次配对 bootstrap 95% CI |
|---|---:|---:|
| Clean | −1.378 pp | [−1.945, −0.810] pp |
| Illumination 1.0 | +1.013 pp | [−0.001, +2.066] pp |
| Defocus 0.2 | +5.915 pp | [+4.903, +7.010] pp |
| Sensor noise 0.4 | +8.388 pp | [+7.131, +9.684] pp |
| 三漂移平均 | **+5.105 pp** | — |

无标签 Adapter 相对 baseline 的三漂移平均提升为 **5.105 pp**，defocus、noise 的增益在
2,468 条全量样本上区间不跨 0；illumination 从初始版的 −0.243 pp 修正为 +1.013 pp（见下），
clean 的代价为 −1.378 pp，能力保持曲线中如实展示。

### 为什么需要 illumination 专项调优（dev-only 选择）

初始 cloud-unlabeled Adapter 已显著恢复 defocus 和 noise，但 illumination 在完整 test 上仍
比 baseline 低 0.243 pp。因此只用 train-internal dev 选择 illumination 强化配置：对
illumination refresh 轨迹过采样、增强 task-logit KD、降低通用 feature 对齐权重，并约束
clean / defocus / noise 不低于准入条件。正式 refresh loss 的 CE 权重仍为 0，不读取这 480 条
轨迹的真实标签。三个候选 profile：

| Profile | illum 权重 | KD | feature | anchor | epochs | dev mean drift |
|---|---:|---:|---:|---:|---:|---:|
| illum_balanced | 2.0 | 1.0 | 0.2 | 0.15 | 8 | 96.354% |
| illum_focus | 4.0 | 1.0 | 0.1 | 0.15 | 8 | 96.458% |
| illum_strong（选中） | 6.0 | 1.2 | 0.05 | 0.1 | 10 | 96.563% |

最终选中 `illum_strong`，即下发物 `cloud_unlabeled_illumination_tuned`。完整选型记录见
`illumination_tuned_summary.json` 与 `illumination_tune.log`。

## 最终权重与云端 artifact

`models/` 只放边缘最终部署权重：

- `models/modelnet40_cloud_teacher_adapter_20260809/cloud_unlabeled/best.pth`
  是最终向边缘下发的约 1.2 MB Adapter（实测 1,219,465 bytes，299,916 参数；
  与云端 `cloud_unlabeled_illumination_tuned_adapter.pth` 同哈希，即 illumination 专项版）。

云端 40 类 task head 不进入 Edge 部署，作为实验 artifact 放在：

- `local/results/modelnet40_cloud_teacher_full_test_20260809/artifacts/teacher_head.pth`

Edge baseline 和冻结 InternViT-6B 是输入依赖，不属于本次新训练权重，也不在上述目录重复存放。
其余对照 Adapter（`label_only` / `hybrid` / `cloud_unlabeled` 初始版 / illumination 三 profile）
只保存在 `local/results/`，不得当作正式部署权重。

## 文件与证据位置

| 内容 | 路径 |
|---|---|
| 完整结果证据（summary / gate / head metrics / predictions / manifest / 日志 / 对照权重） | `local/results/modelnet40_cloud_teacher_full_test_20260809/` |
| 轻量交付（README + 可画图 JSON，含逐条件 bootstrap CI） | `local/delivery/modelnet40_recovery_metrics_20260809/` |
| 最终下发 Adapter | `models/modelnet40_cloud_teacher_adapter_20260809/cloud_unlabeled/best.pth` |
| 主程序 | `module_edge_perception/modelnet_cloud_teacher_refresh.py` |

正式运行需要在 GPU 服务器提供 ModelNet40 数据、ModelNet40 Edge baseline 和冻结
InternViT-6B 路径。脚本会固定 head/refresh/dev split，先生成或复用 feature cache，再依次
完成 teacher gate、Adapter 训练和 official test。`--illumination-tune` 启用最终专项配置的
dev-only 选择流程。
