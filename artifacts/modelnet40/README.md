# ModelNet40 正式证据

实验：完整 official `test`，2,468 条四视图轨迹；结果标签为**实机测量（GPU 推理）**，
相机退化为**合成相机退化**。运行环境：RTX 3090、PyTorch 2.8.0+cu128、CUDA 12.8、
随机种子 42；正式 checkpoint 在测试前固定。复现提交为 `49c3f7a`，原始运行命令与数据隔离
由 `split_manifest.json`、`run.log` 和 `scripts/reproduce_modelnet40.py` 共同记录。

`modelnet40_recovery_metrics.json` 与 `metrics.csv` 是正式绘图数据；
`illumination_tuned_test_predictions.jsonl` 是逐样本记录；`teacher_gate.json`、
`teacher_head_metrics.json` 和 `split_manifest.json` 保留 teacher 准入与数据隔离证据。
