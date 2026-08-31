# 正式实验佐证材料

本目录只放与最终提交口径一致、可公开复核的轻量证据；真实数据集和 checkpoint
通过数据说明与 GitHub Release 获取。每个目录的 `README.md` 说明实验类型、输入、
随机种子、硬件/软件口径和结果边界。

- `modelnet40/`：完整 official test 的 JSON、逐样本预测、split/teacher gate。
- `boxcars/`：完整 official make/test 的指标、置信区间和绘图数据。
- `network/`：四档网络仿真的逐时隙 CSV 与汇总。
- `multi_node/`：加权/贝叶斯仲裁的逐样本记录与汇总。
- `performance/`：TTFT 代理、显存和端到端计时的 CSV/正式图。
- `archive/`：不属于正式结论的历史结果索引，禁止与上述目录混用。

所有网络时延均标明为网络仿真：`T_edge=80 ms` 是固定模拟参数；默认异步口径不将
Adapter 下发计入当前前台请求，显式前台下发对照见网络设计文档。
