# Legacy method archive (2026-08-09)

这些文件是已退出当前 worktree 主线的可恢复归档，不再参与新实验。

- `code/vlm_condition/`：Qwen/VLM hidden-state condition、soft-label 蒸馏及 ModelNet condition 实验。
- `code/prompt_and_retrain/`：Prompt tuning、全量漂移重训及其旧数据/评估入口。
- `code/old_drift_adapter/`：label/clean-feature/VLM-condition 混合的旧 BoxCars Adapter 实现与评估。
- `code/soft_label_baseline/`：旧 baseline soft-label 导出器。
- `docs/CLAUDE_legacy.md`：旧 VLM/Prompt 项目叙事和任务记录。

保留它们仅用于追溯消融和历史结果。当前实现入口在
`module_edge_perception/{evaluate_boxcars_cloud_teacher.py,export_boxcars_cloud_teacher_cache.py,train_boxcars_cloud_teacher_adapter.py}`。
