# 模块二：云边协同调度

## 职责

- Actor-Critic：Actor 生成候选动作，Critic 通过注水算法评估资源分配。
- Lyapunov 带宽队列：约束长期平均通信量并管理后台更新积压。
- 协同模式：`u=0` 本地自治、`u=1` Adapter refresh、`u=2` 结构性模型更新。
- 网络韧性：模拟带宽波动、丢包和断连，并在弱网时执行本地回退。
- 多节点协同：检测重叠感知结果冲突，执行加权投票或贝叶斯融合。

## 正式指标

| 指标 | 要求 | 结果与口径 |
|---|---:|---|
| 前台端到端时延 | ≤200 ms | 异步口径 80 ms；含 1.2 MB Adapter 前台下发为 92–111 ms，均为网络仿真 |
| 业务连续性 | ≥90% | 四档网络 90.28%–90.32% |
| 决策冲突 | ≤5% | 4.06% |
| 冲突解决 | ≥90% | 100% |

> 网络模拟器中的 `T_edge=80 ms` 是固定参数，不是真实边缘硬件的 wall-clock 实测值。默认架构将 Adapter 生成与下发作为后台异步任务，因此不阻塞当前前台推理；92–111 ms 是显式将 Adapter 下发计入前台后的对照仿真。

## 目录

```text
module_scheduling/
├── EdgeCloud_RL/
│   ├── main_edge_cloud_new.py          # 当前单节点调度主循环
│   ├── main_edge_cloud_real_model.py   # 接入真实模型轨迹的全链路评测
│   ├── ../../archive/scheduling/main_edge_cloud.py # 历史接口版本，仅用于结果追溯
│   ├── actor_memory.py
│   ├── critic_water_filling.py
│   ├── network_sim.py
│   └── run_network_resilience_tests.py
├── comparison_baselines/               # LSCI、VBRD、Hyperion 对照实现
└── multi_node/
    ├── arbiter.py
    └── multi_node_eval.py
```

## 复现入口

正式运行命令、所需输入和预期输出见仓库根目录的 [REPRODUCE.md](../REPRODUCE.md)。算法定义和完整结果见：

- [网络波动模拟器设计](../docs/网络波动模拟器设计.md)
- [多节点冲突仲裁测试结果](../docs/多节点冲突仲裁测试结果_20260811.md)
- [接口契约](../docs/接口契约.md)

`comparison_baselines/` 中的基线只用于统一输入与指标口径下的对照。正式结论必须同时报告参数、数据划分和限制，不以简化实现替代原论文完整系统。
