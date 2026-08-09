# 边缘感知与云端视觉教师

本 worktree 只维护当前正式路线：冻结 `InternViT-6B-224px` 云端视觉主干，训练
BoxCars 任务分类头，用漂移样本上的 teacher logits/features 监督
`MV-ViT-S + AdaptFormer`，最终只向边缘下发 adapter 权重。

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

这些数值用于开发选择，尚不是全量官方 test 最终结论。快速独立检查使用
`evaluate_boxcars_cloud_teacher_quick_test.py`：固定已有 checkpoint，在官方 `test` split
中抽取可复核的 class-stratified 子集，保存 sample indices、逐条预测和配对 bootstrap 区间。
完整的方法、结果边界和当前建议见
[`docs/实验状态与证据边界_20260809.md`](../docs/实验状态与证据边界_20260809.md)。

## 保留代码

- `model.py`、`adaptformer.py`：边缘 MV-ViT 和 Adapter。
- `boxcars_dataset.py`、`boxcars_camera_drift_dataset.py`：四视图数据与固定相机漂移。
- `train_boxcars.py`、`evaluate_boxcars.py`：edge baseline。
- `evaluate_boxcars_cloud_teacher.py`：限时零训练 sanity check。
- `evaluate_boxcars_cloud_teacher_quick_test.py`：固定 checkpoint 的快速独立 test 检查；不是训练，也不会按 test 重选模型。
- `train_boxcars_cloud_teacher_head.py`：正式 task head 训练与四种条件的独立准入验证。
- `retrain_boxcars_cloud_teacher_head_from_cache.py`：复用已提取特征，快速比较 noise 加权/专用 head。
- `build_boxcars_teacher_cache_from_feature_cache.py`：复用随机强度 train features，以 selected head 生成正式监督。
- `export_boxcars_cloud_teacher_cache.py`：加载已训练 task head，导出严格对齐的云端监督缓存。
- `train_boxcars_cloud_teacher_adapter.py`：新 u=1 Adapter refresh。
- `summarize_boxcars_cloud_teacher_pipeline.py`：汇总 condition-wise 对照并执行最终增益 gate。
- `benchmarks/`：边缘时延、显存和端到端评测。

旧 VLM condition、VLM soft-label、Prompt、全量漂移重训和旧 Adapter 专家实验均已移至
`local/archive/legacy_methods_20260809/`，不再从这里调用。

详细方案见 `docs/云端视觉教师Adapter方案_20260809.md`。
