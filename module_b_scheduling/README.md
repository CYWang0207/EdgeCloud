# 模块 B · 云边协同推理与任务调度（周老师组）

> 技术底座：MV-ViT + 云端 VLM 知识预言机 + 解耦提示 + Lyapunov 混合 Actor-Critic

## 职责

- 边侧 MV-ViT 实时推理；云端 VLM 作为"知识预言机"下发解耦提示（Penv 环境提示 + ωt 空间先验）
- 漂移感知 `Edrift(t)`（香农熵，零额外算力）
- Lyapunov 混合 Actor-Critic 在带宽约束下在线求解 `a_t = {v_t, u_t, k_t}`
  （Actor：NOP 保序量化离散探索；Critic：KKT 注水闭式解）
- 运行时从模块 A 的操作点菜单（LUT）中选择 SubNet 并切换

## 对标指标

| 指标 | 要求 |
|---|---|
| 平均端到端时延 | ≤ 0.2 s（与模块 A 共同保证） |
| 网络波动期间基本业务保持率 | ≥ 90% |

## 对外接口（详见 `docs/接口契约_v0.md`）

- 消费模块 A 的 `{SubNet_id → (TTFT, mem, acc_surrogate)}` LUT 作为代价/效用模型
- `u_t ≠ 0` 时触发上行 Squery（关键帧）→ 云端（对接模块 C）

## 目录约定（待补充）

```
module_b_scheduling/
├── edge_inference/   # 边侧 MV-ViT 推理
├── drift/            # 漂移感知 Edrift
├── scheduler/        # Lyapunov 混合 Actor-Critic
└── prompts/          # 解耦提示注入
```
