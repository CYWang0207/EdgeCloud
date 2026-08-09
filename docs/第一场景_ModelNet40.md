# 第一场景：ModelNet40 大 ViT 云教师

## 最终路线

本场景采用与 BoxCars 相同的正式新方法，而不是历史有标签 Adapter：

```text
冻结 InternViT-6B
  → ModelNet40 离线标签训练新的 40 类 task head
  → 独立 dev 上执行 teacher gate
  → 代表性漂移轨迹生成 teacher logits / visual features
  → 不读取 refresh 标签，训练 Edge AdaptFormer
  → 固定 checkpoint 后评估完整 official test
```

InternViT-6B 主干始终冻结，没有被 BoxCars 或 ModelNet40 回写。BoxCars 的 16 类 task
head 也没有复用；ModelNet40 使用独立的 40 类 head。

## 针对 ModelNet40 的调整

ModelNet40 是白底、光滑、轮廓主导的规范化 CAD 渲染，四个视角之间包含互补的形状信息。
因此没有在进入 task head 前直接平均四视图 feature，而是保留每个视角的 3,200D
InternViT 表示，逐视角分类后再平均 track logits。

官方 train 共 9,843 条。每类确定性保留 12 条 refresh 和 8 条 dev，其余 9,043 条全部
用于离线 task-head 训练，解决类别数从 64 到 889 的长尾差异。task head 同时看到 clean
与随机 illumination / defocus / sensor-noise feature。

初始 cloud-unlabeled Adapter 已显著恢复 defocus 和 noise，但 illumination 在完整 test 上
比 baseline 低 0.243 pp。专项版只用 train-internal dev 选择：对 illumination refresh
轨迹过采样，增强 task-logit KD、降低通用 feature 对齐权重，并约束 clean、defocus、noise
不低于准入条件。正式 refresh loss 的 CE 权重仍为 0，不读取这 480 条轨迹的真实标签。

## Teacher gate

独立 dev 共 320 条：

| 模型 | Clean | Illumination | Defocus | Sensor noise | Mean drift |
|---|---:|---:|---:|---:|---:|
| Edge baseline | 100.000% | 92.188% | 95.000% | 88.750% | 91.979% |
| InternViT-6B + 40类 head | 97.188% | 95.000% | 97.188% | 94.375% | 95.521% |

teacher mean drift 领先 3.542 pp，准入通过。

## 完整 official test

测试集为全部 2,468 条四视图轨迹，不是 quick subset：

| 方法 | Clean | Illumination | Defocus | Sensor noise | Mean drift |
|---|---:|---:|---:|---:|---:|
| Edge baseline | 96.759% | 89.141% | 88.412% | 84.400% | 87.318% |
| Cloud unlabeled（初始） | 95.827% | 88.898% | 93.598% | 91.370% | 91.289% |
| Label-only upper bound | 96.313% | 90.032% | 94.692% | 93.152% | 92.625% |
| Hybrid | 96.070% | 90.113% | 94.368% | 92.707% | 92.396% |
| **Cloud unlabeled（最终专项版）** | **95.381%** | **90.154%** | **94.327%** | **92.788%** | **92.423%** |

最终无标签 Adapter 相对 baseline 的三漂移平均提升为 5.105 pp；illumination、defocus、
sensor noise 分别提升 1.013、5.915、8.388 pp。clean 下降 1.378 pp。逐条件配对
bootstrap 95% CI 在交付 JSON 中。

## 正式权重（只有两个新产物）

1. `models/internvit6b_modelnet40_cloud_teacher_20260809/task_head/selected_head.pth`
   是云端 40 类 task head。
2. `models/internvit6b_modelnet40_cloud_teacher_20260809/cloud_unlabeled_illumination_tuned/best.pth`
   是最终向边缘下发的约 1.2 MB Adapter。

Edge baseline 和冻结 InternViT-6B 是输入依赖，不属于本次新训练权重，也不在上述目录重复存放。
对照 checkpoint 只保存在 `local/results/`，不得当作正式部署权重。

## 文件与复现入口

- 主程序：`module_edge_perception/modelnet_cloud_teacher_refresh.py`
- 轻量交付：`local/delivery/modelnet40_recovery_metrics_20260809/`
- 完整证据：`local/results/internvit6b_modelnet40_cloud_teacher_20260809/`

正式运行需要在 GPU 服务器提供 ModelNet40 数据、ModelNet40 Edge baseline 和冻结
InternViT-6B 路径。脚本会固定 head/refresh/dev split，先生成或复用 feature cache，再依次
完成 teacher gate、Adapter 训练和 official test。`--illumination-tune` 启用最终专项配置的
dev-only 选择流程。
