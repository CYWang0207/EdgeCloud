# 弱网正式证据

本目录包含 ModelNet40 完整 official test（2,468 时隙）下 static、jitter、
jitter_outage、markov 四档的逐时隙 CSV。结果标签为**网络仿真**：`T_edge=80 ms`
为固定参数，默认 Adapter refresh 异步执行、不阻塞当前前台链路。

随机种子为 42（Markov 固定种子），环境为 RTX 3090、PyTorch 2.8.0+cu128、CUDA 12.8；复现提交
为 `49c3f7a`，命令见 `scripts/reproduce_network.py`。
`summary.json` 同时记录多节点摘要；各 CSV 的字段记录决策、连通性、队列和端到端时延。
