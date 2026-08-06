# CLAUDE.md — 挑战杯 EdgeCloud 项目

> 本文件为"面向云边协同场景的分布式人工智能感知与决策关键技术研究"（赛题 XH-202606）参赛项目的 Claude Code 工作指南。任何新会话打开本项目时应先读本文件，以快速恢复团队决策、技术架构与本周推进上下文。

---

## 一、赛题与硬指标

- **赛题**：XH-202606，面向云边协同场景的分布式人工智能感知与决策关键技术研究
- **发榜单位**：山东浪潮数据库技术有限公司
- **提交截止**：2026-08-31（邮件提交至 wangqun@inspur.com）
- **关键时间节点**：8/3 MVP、8/10 硬指标数字确认、8/17 代码冻结、8/21 最终交付、8/31 提交截止
- **仓库**：https://github.com/CYWang0207/EdgeCloud（私有仓）

### 三大核心功能（赛题要求）
1. 边缘实时感知与轻量化推理——基于国产开源全量大模型（如 DeepSeek）压缩构建边侧轻量大模型，毫秒级感知，离线/弱网可用
2. 云边协同推理与任务调度——边缘轻量模型 + 云端全量模型互补协同，感知网络波动与任务复杂度，在线动态调度最优计算路径
3. 全局决策优化与一致性保障——云端跨域认知，模型分发与更新机制，长周期更新同步至边缘，解决"局部贪婪 vs 全局最优"冲突

### 硬指标
| 指标 | 要求 |
|---|---|
| 边侧轻量模型保持满血能力 | 80%-90%（数学/代码/自然语言推理） |
| TTFT 降低 | ≥75% |
| 单次推理内存 | ≤1.5GB |
| 网络波动期业务功能保持率 | ≥90% |
| 端到端时延（≥2类场景） | ≤0.2s |
| 多节点决策冲突比例 | ≤5% |
| 冲突解决成功率 | ≥90% |

### 评分（百分制）
- 技术效果 40（实时性 15 + 感知决策效果 15 + 资源通信效率 10）
- 方案完整性与可扩展性 25（完整性 15 + 可扩展性 10）
- 系统稳定性与一致性 20（稳定性 10 + 决策一致性 10）
- 创新性与应用价值 15（创新性 10 + 应用价值 5）

---

## 二、团队分工

| 成员 | 模块 | 职责 |
|---|---|---|
| 王成洋（队长） | 系统集成 + 多节点协同 | 总体架构、接口统筹、AdaptFormer 模块、主循环集成、多节点冲突检测与仲裁 |
| 钟捷杭（B） | 模型评测 | TTFT / 内存 / 延迟测量、集中式基准对比、端到端时延、80-90% 能力保持曲线、硬指标汇总 |
| 张晨（C） | 第二场景 | SEVD 数据集适配、全流程训练、train_adapter 训练管线（云端 VLM 蒸馏→adapter） |
| 唐凤玲 (D) | 网络韧性 + 评测 | 网络波动模拟器 network_sim.py、业务连续性指标、网络韧性实验 |
| 苏程鑫 （E）| 文档 / 实验统筹 | 技术报告、对比图表、Demo 视频、进度追踪、方案文档维护 |

**指导老师**：周睿婷、东方、孟杰
**发榜单位联系人**：顾问专家 金老师 15628776777 / 林老师 18811226033；联络专员 王老师 18654565200 / 赵老师 13399579966（工作日 9:00-17:00）

**约定**：每日 21:00 飞书站会 10 分钟；分支命名 `feat/<模块>-<功能>`；合并前至少 1 人 Code Review；权重（*.pth/*.onnx）和数据集不进 Git，统一网盘管理；临时脚本放 tmp 目录。

---

## 三、技术架构与核心决策

### 总体方案（2026-08-03 团队拍板：采用 Adapter 方案）

```
         +---------- 云端 Cloud ----------+
         | DeepSeek / VLM "知识预言机"     |   ← 冻结，不训练
         | · 分析关键帧 → 判断环境状态     |
         | · 蒸馏监督 AdaptFormer adapter  |   ← 云端大模型知识压缩进 adapter
         | · 多节点冲突仲裁                |
         +---+--------------------+------+
   上行关键帧 |                    | 下行 adapter 参数（几百KB~MB）/ 重训权重
             |                    v
         +--------------------------------+
         |    边缘 Edge（路口盒子）        |
         | MV-ViT (ViT-Small, 2200万) 主干冻结
         | + AdaptFormer adapter (可训练) |
         | · Token 剪枝: k_t               |
         | · 视角选择: v_t                 |
         | · 协同模式: u_t ∈ {0,1,2}      |
         | · 漂移感知: E_drift (香农熵)    |
         | · Lyapunov 带宽队列             |
         +--------------------------------+
```

### 压缩叙事（答辩核心）
**云端 VLM 的场景知识，经蒸馏压缩进 AdaptFormer 适配器；adapter 即压缩产物，下发的是压缩后的场景知识，非整个模型。** 压缩的不是参数，是云端大模型蒸馏出来的场景知识；传输的不是模型，是这个压缩后的小适配器。比"把 DeepSeek 量化蒸馏成迷你模型部署边缘"更可行（大模型再压缩也跑不动毫秒级多视角融合），比"纯 Prompt 注入"更贴赛题"基于全量大模型压缩"字面、且能力保持率可测。

### 协同动作语义（u_t）
- **u=0 本地自治**：边缘冻结主干+adapter 直接推理，c_comm=0，离线/弱网可用
- **u=1 adapter 同步**：云端下发 adapter 参数（几百KB~MB），治环境漂移；接现有 EdgeCloud_RL 的 u=1 动作
- **u=2 重训权重同步**：云端下发重训权重（~50MB），治结构性漂移；Dora 调度的"重训练"对象=只重训 adapter（尺度匹配时间槽）

### 调度与理论
- **Actor-Critic + Lyapunov + 注水算法**：Actor(DNN) 生成候选 (v,u)，Critic(注水闭式解) 算最优 k_t，Lyapunov 虚拟队列 Y_bw 保证长期平均带宽约束
- **理论证明**：Slater 可行性、虚拟队列强稳定（Theorem 2）、O(1/V) 效用逼近界（Theorem 3）、KKT 注水闭式解（Proposition 1）。证明建立在**单一虚拟队列 Y_bw** 上，不可随意加第二罚项队列
- **Dora**：原论文假设每时间槽可重训练模型（对小视觉模型成立）；本项目把 Dora 调度的重训练对象换成小 adapter，使尺度重新匹配时间槽/数据量/epoch/资源

---

## 四、代码结构

```
EdgeCloud/
├── module_edge_perception/          # 模块一：边缘实时感知
│   ├── model.py                     # EarlyFusionMultiViewViT（timm ViT + 多视角早期融合 + token剪枝 + 视角掩码）
│   ├── adaptformer.py               # AdaptFormer PEFT 模块（王成洋 8/3 实现，挂每个block的FFN旁路并行；含 attach/freeze/计数工具）
│   ├── boxcars_dataset.py           # BoxCars116k 四逻辑视图车辆品牌识别（张晨第二场景，替代 SEVD）
│   ├── boxcars_drift_dataset.py     # BoxCars 专用漂移数据包装
│   ├── train_boxcars.py             # BoxCars baseline DDP 训练（已 30 轮，test Top-1=88.04%）
│   ├── evaluate_boxcars.py          # BoxCars 官方 validation/test 评估
│   ├── train_boxcars_retrain_drift.py  # BoxCars 漂移全量重训
│   ├── train_boxcars_token_prompt.py   # BoxCars Prompt 适配训练（adapter 落地前的过渡）
│   ├── test_boxcars_inference.py     # BoxCars→MV-ViT 前向冒烟测试
│   ├── dataset.py / drift_dataset.py
│   ├── prompt_tuning/               # PromptGenerator（PEFT，现作为可选辅助；存在两套实现待清理）
│   ├── train.py / train_retrain_drift.py / train_token_prompt.py  # 场景一 ModelNet40 训练管线
│   ├── train_adapter.py             # C 8/4 起，云端 VLM 软标签蒸馏→只训 adapter（8/3 晚骨架已写，待服务器跑）
│   └── benchmarks/                  # B 的评测脚本
│       ├── benchmark_latency.py     # TTFT/延迟（已达标）
│       ├── benchmark_memory.py      # 内存（已达标）
│       ├── benchmark_full.py        # 精度+延迟+内存一体化（已支持 modelnet40|boxcars 双场景）
│       ├── build_scheduler_latency.py  # 接口1：延迟喂调度器
│       ├── demo_mvp.py              # 8/3 MVP 演示（内存为硬编码估值，非真测）
│       └── results/                 # CSV + PNG 实测结果
├── module_scheduling/               # 模块二：云边协同调度
│   ├── EdgeCloud_RL/
│   │   ├── main_edge_cloud_new.py   # 单节点主循环（已接 generate_real_trajectory 轨迹，待接 network_sim + adapter u=1）
│   │   ├── main_edge_cloud.py       # 【已失效】旧版，调用 Critic 仅 6 参，现 Critic 签名为 8 参，勿用
│   │   ├── critic_water_filling.py  # 注水闭式解 Critic（8/3 晚已改：u=1 走 S_adapter 口径）
│   │   ├── actor_memory.py          # Actor DNN + 经验回放
│   │   ├── generate_real_trajectory.py  # 场景一轨迹生成（KL/置信度算 w_t + 归一化熵算 E_drift）
│   │   ├── evaluate_rl_policy_on_mvvit.py  # RL 策略真实评估（8/3 晚已改：u=1 常挂 adapter + 加载下发权重）
│   │   └── plot_*.py                # 准确率/token 数时序画图
│   ├── comparison_baselines/        # LSCI/VBRD/Hyperion 基线（有意弱化，答辩公平性有风险）
│   └── multi_node/                  # 【待建】arbiter.py / overlap_manager.py / rollback.py
├── common/drift_dataset.py          # 确定性漂移主实现（5类×6档 schedule）；注意 module_scheduling/EdgeCloud_RL/ 下有副本，导入路径不统一
├── data/  data/README.md           # 数据集不进 Git（ModelNet40 ~2GB、BoxCars116k）
├── models/ models/README.md         # 权重不进 Git（注：checkpoints/*.pth 当前为 0 字节占位，真权重在服务器）
├── docs/
│   ├── 方案总览.md                   # 苏程鑫维护，待改为 Adapter 叙事
│   ├── 接口契约.md                   # 接口1-4（u=1 需改为 adapter 同步）
│   ├── 第二场景_BoxCars116k.md       # 张晨第二场景定义（baseline 88.04% Top-1）
│   └── 环境配置避坑指南.md
├── scripts/verify_env.py
└── requirements.txt
```

### 已达标硬指标（B 的 benchmark 实测，2026-07-30 快照）
| 指标 | 实测 | 状态 |
|---|---|---|
| 参数量 | 21.7M (ViT-Small) | ✅ |
| TTFT 降幅 | 86.8%（287→38ms，keep=0.1） | ✅ |
| 推理内存 | 0.08GB | ✅ |
| 单帧延迟 | 38ms | ✅ |
| 精度列 | BoxCars116k make 16 类 Top-1=88.04% / Top-5=97.13%（30 轮 baseline，场景二） | ✅ 场景二 |

---

## 五、网络韧性注入设计（唐凤玲方案，2026-08-03）

在单节点主循环外加一层"时变网络 + 时延 + 业务可用性"，不动原有 Lyapunov 证明。

### 核心原则
- **Q_net 是物理传输积压队列（状态量），不进效用目标 G 的罚项**；长期平均带宽约束仍由 Y_bw 单一虚拟队列保证，Lyapunov 证明不变。文档须显式写明此句锁死理论故事。
- **Y_bw 更新**：`Y_bw[t+1] = max(Y_bw[t] + c_comm - b_avg, 0)`
- **Q_net 更新**：`Q_net[t+1] = max(Q_net[t] + c_comm - B_t, 0)`（追踪瞬时积压，需加上限/丢弃防无限增长）
- **B_t = R_t × slot_duration / 8**（R_t 单位 Mbps，B_t 单位 MB/slot）

### 四档网络模式（含 static 基线做消融）
- `static`：不模拟弱网，基线对照
- `jitter`：带宽抖动，不断联
- `jitter_outage`：jitter + 随机断联(disconnect_prob) + 周期断联(outage_period+outage_duration)
- `markov`：GOOD/WEAK/DOWN 三态弱网（主模型，转移矩阵须写死）

### 关键处理
- **断联 → 强制 u=0**（只保留本地自治候选）
- **超带宽 → 软罚或硬过滤**：`G_effective = G_raw - overflow_penalty × (comm_overflow / B_t)`；`--strict-bandwidth` 直接过滤
- **u=1 / u=2 默认异步后台**：本时隙先用本地旧模型出决策，adapter 参数包(u=1)或重训权重(u=2)进 Q_net 积压队列，后续容量充足再消化；`--sync-u2` 才让 u=2 实时同步
- **端到端时延**：`T_e2e = T_edge + T_comm + T_cloud`；`T_comm = 8 × realtime_comm / R_t × 1000 + RTT`；u=0 时 T_comm=0（特判，不加 RTT）；T_cloud 按动作分（u=0→0、u=1异步→0、u=2异步→0、u=2同步(`--sync-u2`)→重训时延）。u=1/u=2 异步时当步 e2e≈T_edge，权重包后台传输完成情况由业务保持率反映
- **业务可用四条件（防"断联切本地=100%可用"虚高）**：`business_available = decision_success AND e2e≤deadline AND active_views≥min AND proxy_acc≥acc_floor`

### NetworkSimulator 接口（王成洋 8/4 接主循环对齐）
```python
class NetworkSimulator:
    def __init__(self, mode, slot_duration, ...): ...
    def step(self) -> dict:                    # 采样 R_t, B_t, net_state, is_disconnected
    def filter_candidates(self, candidates, c_comm_map):  # 断联/超带宽过滤
    def compute_e2e(self, u, realtime_comm, t_edge):      # 返回 comm_delay, e2e_delay
    def update_queues(self, c_comm, b_avg):               # 更新 Y_bw, Q_net
    def is_business_available(self, decision_success, e2e_ms, active_views, proxy_acc):  # 业务可用四条件
```

---

## 六、本周（8/3-8/9）关键路径与依赖

### 主关键路径（最长，决定 8/10 能否出数）
```
王成洋 adaptformer.py(8/4) → C train_adapter 训练(8/5) → C adapter 权重(8/5-8/8)
  → B 全指标评测(8/6-8/7) → B 80-90% 能力曲线(8/7) → 苏程鑫 报告填数(8/8) → 8/10 指标定稿
```
**原最大瓶颈 adaptformer.py 已于 8/3 晚 verify 跑通（卡点解除）；当前瓶颈转为 C 的 adapter 训练权重产出（8/5-8/8）。**

### 每人本周主线
- **王成洋**：adaptformer.py(8/3-8/4) → u=1 adapter sync 接入(8/5) → Dora 重训→adapter 重构(8/6) → 多节点 arbiter(8/7-8/8) → 系统联调(8/9)
- **张晨 C**：~~SEVD 任务定义~~ → 已改用 BoxCars116k（baseline 已 88% 跑通）→ train_adapter.py(8/4-8/5，待 adaptformer.py 落地) → 两场景 adapter 权重(8/5-8/8) → 漂移重训(8/7)
- **钟捷杭 B**：benchmark_full 扩 adapter 口径 + benchmark_e2e 骨架(8/3) → 集中式基准+e2e(8/4-8/5) → 真权重全指标(8/6) → 80-90% 能力保持曲线(8/7) → 冲突率统计(8/8) → 指标对照表(8/9)
- **唐凤玲**：network_sim 6 点决策+开写(8/3) → network_sim.py 完成(8/4) → 弱网实验(8/5-8/6) → 接入 e2e(8/7) → 仲裁下发韧性(8/8) → 数据定稿(8/9)
- **苏程鑫**：方案文档改 Adapter 叙事(8/3) → 接口契约 v2(8/4) → 答辩叙事(8/5) → 报告框架(8/7) → 填数据出图(8/8) → 汇报材料(8/9)

### 8/10 硬指标交付清单
| 指标 | 负责人 |
|---|---|
| 参数量 / TTFT / 内存 / 单帧延迟 | B |
| 80-90% 能力保持曲线（adapter vs 云端 VLM） | B + C |
| 端到端时延 ≤0.2s（含网络） | B + 唐凤玲 |
| 业务保持率 ≥90% | 唐凤玲 |
| 冲突 ≤5% / 解决 ≥90% | 王成洋 + B |
| ≥2 类场景（ModelNet40 + BoxCars116k） | C |

---

## 七、关键风险与待确认

1. **VLM Oracle 现在是模拟的**——adapter 蒸馏需要真"老师"，本周内至少接一个真 VLM（InternVL/Qwen-VL）产软标签，否则"云端大模型压缩→adapter"叙事不实。需和老师确认 VLM 推理资源。
2. **~~SEVD 任务语义错配~~ → 已解决**：张晨已放弃 SEVD，改选 **BoxCars116k 交通车辆品牌识别（16 类 make 分类任务）**。BoxCars 是分类任务，与 MV-ViT 分类模型语义匹配，错配问题消除。baseline 已 30 轮训练，官方 test Top-1=88.04%、Top-5=97.13%（详见 `docs/第二场景_BoxCars116k.md`）。注意 BoxCars 不提供跨摄像头身份，4 张图是同一摄像头下同车轨迹的 4 个时间观测 + `view_mask`，叙事上不得表述为"四台摄像头同拍一辆车"。
3. **"80-90% 能力保持"指标口径**——赛题写"数学/代码/NLP"，本项目是视觉任务。需让老师/队长问发榜单位：adapter 方案算不算"基于全量大模型压缩"？80-90% 指标在视觉任务上能否按"边缘 adapter 保持云端 VLM 能力百分比"理解？问清就踏实。
4. **基线公平性**——comparison_baselines 的 LSCI/VBRD/Hyperion 被有意弱化，README 已明文承认"简化版突出缺点"；答辩若被追问公平性有风险，至少一个基线用未弱化实现。
5. **prompt/adapter 50/50 已于 8/3 定 adapter**——prompt 代码不删，作"环境漂移快响应"可选辅助，但本周不铺开做，先把 adapter 主线打透。当前 prompt 存在两套互不兼容实现（顶层 `train_token_prompt.py` 用 `EarlyFusionMultiViewViT`+5类漂移；`prompt_tuning/` 子目录用独立 `PromptMultiViewViT`+仅亮度3类），属待清理技术债。
6. **代码与文档同步进度**——8/3 拍板的 adapter 方案：`adaptformer.py` 王成洋 8/3 已实现，**8/3 晚本机 py3.11 verify_adaptformer.py 三点验收全通过**（①零初始化 Δ=0 ②adapter 0.300M<1M ③三条前向路径+wrapper hook 全触发）；`train_adapter.py` 骨架 8/3 晚已写（预计算软标签蒸馏，待服务器跑通）；u=1 已在 critic_water_filling / baseline_common / main_edge_cloud_new / evaluate_rl_policy 四处由 S_prompt/prompt 注入改为 S_adapter/adapter 加载（8/3 晚代码层完成，待真轨迹/权重验证）；方案总览.md、README.md、接口契约.md 叙事已对齐 Adapter。

---

## 八、AdaptFormer 模块技术规格（王成洋 8/3 实现；8/3 晚本机 py3.11 verify_adaptformer.py 验收通过）

- **挂载点**：每个 timm Transformer block 的 mlp（FFN）旁路并行
- **瓶颈维度 r**：32（vit_small embed_dim=384，压缩比 12:1，每层 ~24K 参数，12 层共 ~290K）
- **结构**：`down(D→r) → GELU → up(r→D)`，W_up 零初始化使启动时 adapter 输出为 0（不破坏预训练主干）
- **实现方式**：用 `AdaptFormerMLPWrapper` 包裹 timm 原生 `block.mlp`，`forward = mlp(x) + scale × adapter(x)`；替换 `block.mlp` 后三条前向路径（forward / forward_hard_prune / _hard_prune_per_sample）自动生效，无需改前向逻辑
- **冻结策略**：主干全部 `requires_grad=False`，只训 adapter + norm + head
- **验收**（verify_adaptformer.py）：①零初始化挂上后输出差异 < 1e-3；②可训参 < 1M；③三条前向路径均生效。8/3 晚本机实测：① Δ=0.000e+00；② adapter=299,916 (0.300M)、冻结后可训 316,084；③ forward / forward_hard_prune / _hard_prune_per_sample 全过 + wrapper hook 触发。运行注意：Windows 控制台是 GBK，需 `PYTHONUTF8=1 py -3.11 verify_adaptformer.py`（否则打 ✅ 崩）
- **参考论文**：AdaptFormer (ICCV 2022)；相关对比：LoRA (ICLR 2022)、Houlsby Adapter (ICML 2019)、Hinton 蒸馏 (2014)

---

## 九、环境与访问

- **训练环境**：AutoDL 云 GPU（数据路径默认 `/root/autodl-tmp/`），N 卡
- **本会话运行环境**：WSL，工作目录默认 `/home/AIDD`（AIDD 项目），挑战杯代码读绝对路径（如 `/mnt/d/edgeCloud` 或 `/mnt/d/Challenge/EdgeCloud_clone`）
- **GitHub 私仓访问**：需 PAT 或 SSH key；PAT 已用于 clone（注意：PAT 出现在对话后建议 revoke 重生成）
- **bash 安全分类器**：本环境 glm-5.2 分类器偶发不可用，期间 git clone / pip 等需分类的命令会失败，但 Read/Grep/Glob 文件读取不受影响。clone 等操作改由用户在自己的 WSL 终端执行。

---

## 十、给新会话的快速恢复提示

1. 先读本文件恢复团队/架构/决策上下文
2. 读 `docs/方案总览.md` 和 `docs/接口契约.md` 看最新方案文档（注意：苏程鑫 8/3 后改为 Adapter 叙事，若仍为 Prompt 注入版说明文档未更新）
3. 读 `module_edge_perception/model.py` + `adaptformer.py`（已实现）看模型与 adapter 实现，`verify_adaptformer.py` 是其验收脚本
4. 读 `module_scheduling/EdgeCloud_RL/main_edge_cloud_new.py` + `critic_water_filling.py` 看调度主循环与注水算法
5. 用 `git log --oneline -20` 看最近提交，了解各人最新进度
6. 所有回复用中文
7. 若需接续进行中任务，看 `docs/交接_<日期>.md`（如 `docs/交接_2026-08-03.md`）了解当前卡点与下一步
