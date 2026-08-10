# 边缘感知与云端视觉教师

本 worktree 只维护当前正式路线：冻结 `InternViT-6B-224px` 云端视觉主干，分别训练
ModelNet40 / BoxCars 任务分类头，用漂移样本上的 teacher logits/features 监督
`MV-ViT-S + AdaptFormer`，最终只向边缘下发 adapter 权重。

ModelNet40 已完成完整 official test。其新方法入口是
`modelnet_cloud_teacher_refresh.py`：保留四视图各自的 3,200D InternViT feature，使用
9,043 条离线有标签 train 轨迹训练 40 类 task head；320 条互斥 dev 执行 teacher gate，
480 条互斥 refresh 轨迹只提供 cloud logits/features，不在正式 Adapter loss 中读取标签。
针对白底、光滑、轮廓主导的渲染物体，最终版对 illumination 轨迹过采样并增强 task-logit
KD，由 dev 在保持 clean/defocus/noise 的约束下选择配置。

## 当前主线

1. `evaluate_boxcars_cloud_teacher.py`：五分钟以内的 prototype 小样本排雷；不作为正式结果。
2. `train_boxcars_cloud_teacher_head.py`：云端冻结 InternViT，缓存 clean + 相机退化训练集特征并训练 linear/MLP 分类头。
3. 在独立 validation 的 clean、illumination、motion blur、sensor noise 上验证 teacher。
4. teacher 明显超过对应 edge baseline 后，导出 task logits/features cache。
5. `train_boxcars_cloud_teacher_adapter.py`：只训练无条件 AdaptFormer 和训练期 projector。
6. 下发 adapter-only checkpoint；InternViT 和 projector 均不进入 edge forward。

这里的“无标签”是指代表性上传轨迹在云端刷新时不需要人工品牌标签；并不表示训练阶段没有
BoxCars 标签。标签只用于预先训练冻结 InternViT feature 上的任务 head，以及构造
`label-only` 对照上界。

## 已有开发集结果的正确用法

`cloud-unlabeled` 是当前正式 `u=1` 产物：它使用 `KD + feature + clean anchor`，不使用
上传样本的 CE。在完整 validation 上，它的三漂移平均为 85.21%，高于 Edge baseline 的
81.97%，且只低于有标签 label-only 上界 0.87 pp。另一方面，`CE + KD + feature + anchor`
的 hybrid 为 85.67%，低于 label-only 的 86.08%。因此不要把 hybrid 当主路线，也不要声称
teacher 在已有标签时必然提升 CE；它的实际作用是替代缺失的上传样本标签。

这些开发集数值用于模型选择；独立的全量官方 test 结论见下一节。固定 checkpoint 的检查
入口是 `evaluate_boxcars_cloud_teacher_quick_test.py`：在官方 `test` split 中抽取可复核的
class-stratified 子集（默认 256 条）或（`--samples 0`）全量评估，保存 sample indices、
逐条预测和配对 bootstrap 区间，且不会按 test 重选任何模型。完整的方法、结果边界见
[`docs/第二场景_BoxCars116k.md`](../docs/第二场景_BoxCars116k.md)。

## 完整官方 test（12,322 条）

固定开发阶段 checkpoint（Edge baseline 与 `cloud_unlabeled/best.pth`），在官方 `test`
split 全量 12,322 条 × 4 条件的逐样本配对评估；未按 test 重选 head、epoch 或 loss。
`evaluate_boxcars_cloud_teacher_quick_test.py --samples 0` 可省略 teacher 参数跳过 6B
推理，只评 Edge/Adapter 两组模型，约 7 分钟：

| Model | Clean | Illumination | Blur | Noise | Mean drift |
|---|---:|---:|---:|---:|---:|
| Edge baseline | 88.04% | 74.03% | 77.57% | 71.77% | 74.46% |
| Cloud unlabeled Adapter | 88.09% | 78.45% | 79.51% | 79.10% | 79.02% |

Adapter 相对 Edge 的三漂移平均提升 **+4.56 pp**，逐条件 2,000 次配对 bootstrap 95% CI
全部不跨 0：illumination +4.42 pp（[+3.95, +4.90]）、blur +1.94 pp（[+1.57, +2.31]）、
noise +7.33 pp（[+6.77, +7.90]）；clean 基本持平 +0.06 pp（[-0.20, +0.31]）。这补齐了
256 条快速检查中 illumination/blur 区间跨 0 的限制。完整证据位于
`local/results/boxcars_cloud_teacher_full_test_20260810/`。

ModelNet40 则已有 2,468 条完整 official test：最终无标签 Adapter 将三漂移平均准确率从
87.318% 提升至 92.423%（+5.105 pp）；illumination / defocus / sensor noise 分别提升
+1.013 / +5.915 / +8.388 pp，clean 代价为 -1.378 pp。详细协议、权重和证据见
[`docs/第一场景_ModelNet40.md`](../docs/第一场景_ModelNet40.md)。

## 保留代码

- `model.py`、`adaptformer.py`：边缘 MV-ViT 和 Adapter。
- `dataset.py`、`modelnet_camera_drift_dataset.py`：ModelNet40 四视图数据与相机漂移。
- `modelnet_cloud_teacher_refresh.py`：ModelNet40 大 ViT task head、teacher gate、无标签刷新、对照及完整 test。
- `boxcars_dataset.py`、`boxcars_camera_drift_dataset.py`：四视图数据与固定相机漂移。
- `train_boxcars.py`、`evaluate_boxcars.py`：edge baseline。
- `evaluate_boxcars_cloud_teacher.py`：限时零训练 sanity check。
- `evaluate_boxcars_cloud_teacher_quick_test.py`：固定 checkpoint 的独立 test 评估，默认 256 条 class-stratified 子集，`--samples 0` 跑全量 12,322 条；teacher 参数可省略；不是训练，也不会按 test 重选模型。
- `train_boxcars_cloud_teacher_head.py`：正式 task head 训练与四种条件的独立准入验证。
- `retrain_boxcars_cloud_teacher_head_from_cache.py`：复用已提取特征，快速比较 noise 加权/专用 head。
- `build_boxcars_teacher_cache_from_feature_cache.py`：复用随机强度 train features，以 selected head 生成正式监督。
- `export_boxcars_cloud_teacher_cache.py`：加载已训练 task head，导出严格对齐的云端监督缓存。
- `train_boxcars_cloud_teacher_adapter.py`：新 u=1 Adapter refresh。
- `summarize_boxcars_cloud_teacher_pipeline.py`：汇总 condition-wise 对照并执行最终增益 gate。
- `benchmarks/`：边缘时延、显存和端到端评测。

旧 VLM condition、VLM soft-label、Prompt、全量漂移重训和旧 Adapter 专家实验均已移至
`local/archive/legacy_methods_20260809/`，不再从这里调用。

方法要点与实验总览见 `docs/实验结果总览_20260809.md`；完整方法设计见归档文档
`local/archive/status_docs_20260810/云端视觉教师Adapter方案_20260809.md`。
