# 模块二：云边协同调度

## 职责

- Actor-Critic RL：Actor（DNN）生成候选动作，Critic（注水算法）精确评估
- Lyapunov 带宽队列：长期平均带宽约束优化
- VLM Oracle：模拟 DeepSeek 场景分析行为
- 多节点协同：冲突检测 + 云端一致性仲裁
- 网络韧性：带宽波动 + 断连 fallback

## 对标指标

| 指标 | 要求 | 方法 |
|------|------|------|
| 端到端时延 | <= 0.2s | 视角休眠 + Token 剪枝 + O(V²) 注水 |
| 业务连续性 | >= 90% | Lyapunov 稳定性 + ut=0 本地自治 |
| 决策冲突 | < 5% | 云端 VLM 全局仲裁 |
| 冲突解决 | >= 90% | 置信度加权投票 + 回滚 |

## 目录约定（部分目录待建）

```
module_scheduling/
├── EdgeCloud_RL/             # Actor-Critic RL + Lyapunov 主循环（实际代码在此，非 scheduler/ 子目录）
│   ├── main_edge_cloud_new.py  # 单节点主循环（已接 adapter u=1 口径 + network_sim）
│   ├── main_edge_cloud.py      # 【已失效】旧版，Critic 调用签名不匹配，勿用
│   ├── actor_memory.py / critic_water_filling.py
│   ├── generate_real_trajectory.py / evaluate_rl_policy_on_mvvit.py
│   ├── network_sim.py           # 网络波动模拟器
│   └── plot_*.py
├── comparison_baselines/     # LSCI/VBRD/Hyperion 基线（有意弱化，答辩公平性有风险）
├── multi_node/                # 【待建】arbiter.py / overlap_manager.py / rollback.py
```


## 负责人任务

### A（组长）— 多节点协同 + 集成
- 在 main_edge_cloud_new.py 中扩展多节点仿真（多个路口节点并行跑）
- 实现冲突检测算法：同一目标两个节点判断不一致 + 双方置信度都 > 0.7 = 冲突
- 实现冲突解决策略：云端仲裁 + 置信度加权投票
- 目标：冲突比例 < 5%，解决成功率 >= 90%
- 预估代码量：100-150 行，纯本地仿真

### D — 网络韧性 + 评测
- 在调度仿真中加入带宽波动 + 断连场景模拟
- 实现业务连续性保持率指标 >= 90%
- 端到端 wall-clock 延迟测量 <= 0.2s
- 全指标评测自动化脚本
- 注意：和 A 改同一个文件（main_edge_cloud_new.py），注意分支管理
