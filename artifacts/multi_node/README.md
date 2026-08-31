# 多节点仲裁正式证据

`multi_node_eval_weighted_full.csv` 与 `multi_node_eval_bayesian_full.csv` 为 BoxCars116k
官方 `make/test` 全量 12,322 轨迹的逐次记录。结果标签为**实机模型推理 + 仲裁仿真**：
四个逻辑节点各观察同一车辆轨迹的不同三视图，而非四个物理摄像头。

固定 seed 42、RTX 3090、PyTorch 2.8.0+cu128、CUDA 12.8；复现提交为 `49c3f7a`，命令见
`scripts/reproduce_multinode.py`；`summary.json` 给出冲突率、
解决率、回滚数及 weighted/bayesian 两种融合的汇总。
