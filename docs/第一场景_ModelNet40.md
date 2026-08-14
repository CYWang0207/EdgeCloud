# 第一场景：ModelNet40 大 ViT 云教师

> 2026-08-12 对最终无标签 Adapter 进行一次 clean-guard 再训练：复用既有
> InternViT feature cache，不改变数据划分、教师或 Adapter 结构；新权重在同一完整 official
> test 上将 clean 提升至 95.705%，illumination 提升至 92.301%。本节中的“最终版”均指该权重。

> 本文是本数据集的结果文档；8/9 初始实验来自云端 `adjust` 分支
> `local/results/modelnet40_cloud_teacher_full_test_20260809/`，8/12 clean-guard 复训结果
> 固化在 `shared/local/results/modelnet40_cloud_teacher_clean_guard_20260812/`。方法要点与总体结论见
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
| **Cloud unlabeled（最终 clean-guard 专项版）** | **95.705%** | **92.301%** | **94.449%** | **92.990%** | **93.247%** |

本数据集实际评估了**多种方法对照**，不是只有一种：`cloud_unlabeled`（无上传标签主系统）、
`label_only`（有标签监督上界）、`hybrid`（CE+teacher 同时使用的消融），以及后续
illumination 专项调优得到的最终 `cloud_unlabeled_illumination_tuned`。

### 最终专项版相对 Edge baseline

| 条件 | 相对 Edge baseline 的完整测试提升 | 备注 |
|---|---:|---:|
| Clean | −1.053 pp | — |
| Illumination 1.0 | +3.160 pp | — |
| Defocus 0.2 | +6.037 pp | — |
| Sensor noise 0.4 | +8.590 pp | — |
| 三漂移平均 | **+5.929 pp** | — |

无标签 Adapter 相对 baseline 的三漂移平均提升为 **5.929 pp**。相比 8/9 的最终专项版，
clean 额外恢复 **0.324 pp**，illumination 额外提升 **2.147 pp**；clean 仍低于 frozen Edge
baseline 1.053 pp，能力保持曲线中如实展示。

### clean-guard 专项调优（dev-only 选择）

8/9 的 6× illumination profile 虽使 illumination 转正，但 clean 代价扩大至 −1.378 pp。
本轮因此将 illumination 过采样收敛至 2–4×，把 clean replay anchor 提升至 0.35–0.40，
并将其温度降至 1；同时将 clean 准入收紧至不低于 dev baseline 0.3 pp。正式 refresh loss
的 CE 权重仍为 0，不读取这 480 条轨迹的真实标签。三个候选 profile：

| Profile | illum 权重 | KD | feature | anchor | epochs | dev mean drift |
|---|---:|---:|---:|---:|---:|---:|
| illum_clean_guard | 2.0 | 1.0 | 0.10 | 0.35 | 8 | 96.667% |
| illum_balanced_guard | 3.0 | 1.0 | 0.08 | 0.40 | 8 | 96.979% |
| illum_focus_guard（选中） | 4.0 | 1.1 | 0.05 | 0.40 | 10 | 97.083% |

最终选中 `illum_focus_guard`，即下发物 `cloud_unlabeled_illumination_tuned`。完整选型记录
见本地 `shared/local/results/modelnet40_cloud_teacher_clean_guard_20260812/illumination_tuned_summary.json`。

## 最终权重与云端 artifact

`shared/models/` 只放边缘最终部署权重：

- `shared/models/modelnet40_cloud_teacher_adapter_20260812/cloud_unlabeled/best.pth`
  是最终向边缘下发的约 1.2 MB Adapter（1,219,859 bytes，299,916 参数；SHA-256
  `1e24728b3ffa1f44f0dfd1db64c7b0f2e195566df8cb2b38e2dbebb037f4d82a`）。

云端 40 类 task head 不进入 Edge 部署，作为实验 artifact 放在：

- `local/results/modelnet40_cloud_teacher_full_test_20260809/artifacts/teacher_head.pth`

Edge baseline 和冻结 InternViT-6B 是输入依赖，不属于本次新训练权重，也不在上述目录重复存放。
其余对照 Adapter（`label_only` / `hybrid` / `cloud_unlabeled` 初始版 / illumination 三 profile）
只保存在 `local/results/`，不得当作正式部署权重。

## 文件与证据位置

| 内容 | 路径 |
|---|---|
| 完整结果证据（summary / gate / head metrics / predictions / manifest / 日志 / 对照权重） | `local/results/modelnet40_cloud_teacher_full_test_20260809/` |
| 轻量交付（README + 可画图 JSON，含逐条件 bootstrap CI） | `shared/local/delivery/modelnet40_recovery_metrics_20260812/` |
| 最终下发 Adapter | `shared/models/modelnet40_cloud_teacher_adapter_20260812/cloud_unlabeled/best.pth` |
| 主程序 | `module_edge_perception/modelnet_cloud_teacher_refresh.py` |

正式运行需要在 GPU 服务器提供 ModelNet40 数据、ModelNet40 Edge baseline 和冻结
InternViT-6B 路径。脚本会固定 head/refresh/dev split，先生成或复用 feature cache，再依次
完成 teacher gate、Adapter 训练和 official test。`--illumination-tune` 启用最终专项配置的
dev-only 选择流程。
