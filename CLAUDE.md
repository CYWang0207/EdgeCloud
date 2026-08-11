# CLAUDE.md — 挑战杯 EdgeCloud 项目

> 本文件为"面向云边协同场景的分布式人工智能感知与决策关键技术研究"（赛题 XH-202606）参赛项目的 Claude Code 工作指南。任何新会话打开本项目时应先读本文件，以快速恢复团队决策、技术架构与本周推进上下文。
> **最后更新：2026-08-10**（合入 PR#30 cloud-teacher 正式方案 + PR#31 网络文档第二次测试版；FiLM/VLM-conditioned 降为消融。张晨 8/9 pivot 到云端视觉教师 Adapter，云端 InternViT-6B 真入训练回路，"VLM 压缩"叙事坐实）

---

## 一、赛题与硬指标

- **赛题**：XH-202606，面向云边协同场景的分布式人工智能感知与决策关键技术研究
- **发榜单位**：山东浪潮数据库技术有限公司
- **提交截止**：2026-08-31（邮件提交至 wangqun@inspur.com）；**答辩时间未定，不急**
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
| 王成洋（队长） | 系统集成 + 多节点协同 | 总体架构、接口统筹、AdaptFormer 模块、主循环集成、多节点冲突检测与仲裁、CLAUDE.md 维护 |
| 钟捷杭（B） | 模型评测 | TTFT / 内存 / 延迟测量、集中式基准对比、端到端时延、能力保持曲线、硬指标汇总 |
| 张晨（C） | 第二场景 + 云端视觉教师 Adapter | BoxCars116k 适配、cloud-teacher 训练管线（InternViT-6B 教师→无标签 adapter 刷新） |
| 唐凤玲 (D) | 网络韧性 + 评测 | network_sim.py、业务连续性指标、网络韧性实验、proxy_acc 口径 |
| 苏程鑫 （E）| 文档 / 实验统筹 | 技术报告、对比图表、Demo 视频、进度追踪、方案文档维护 |

**指导老师**：周睿婷、东方、孟杰
**发榜单位联系人**：顾问专家 金老师 15628776777 / 林老师 18811226033；联络专员 王老师 18654565200 / 赵老师 13399579966（工作日 9:00-17:00）

**约定**：每日 21:00 飞书站会 10 分钟；分支命名 `feat/<模块>-<功能>`；合并前至少 1 人 Code Review；权重（*.pth/*.onnx）和数据集不进 Git，统一网盘管理；临时脚本放 tmp 目录。

---

## 三、技术架构与核心决策

### 总体方案（2026-08-03 Adapter；2026-08-05 漂移校正；2026-08-09 云端视觉教师 cloud-teacher 正式落地）

```
         +---------- 云端 Cloud ----------+
         | InternViT-6B-224px 视觉教师     |   ← 冻结，5.9B 参数，国产开源（OpenGVLab）
         | · 提视觉特征 + BoxCars 分类头  |   ← 训 linear/MLP head 产 task logits
         | · 为漂移样本产 {features,logits} cache  |   ← 真在训练回路产监督
         | · 仅训练边缘 MV-ViT-S 的 AdaptFormer     |
         | · 多节点冲突仲裁                |
         +---+--------------------+------+
   上行 representative 漂移样本 |    | 下行 adapter-only checkpoint（~1.2MB）
             |                  v
         +--------------------------------+
         |    边缘 Edge（路口盒子）        |
         | MV-ViT (ViT-Small, 2200万) 主干冻结
         | + AdaptFormer adapter (可训练) |
         | · 云端教师蒸馏：KL(学生,教师) + 特征对齐 + CE + clean anchor
         | · 边缘 forward 不含大模型/不依赖云端实时返回
         | · Token剪枝: k_t               |
         | · 视角选择: v_t                 |
         | · 协同模式: u_t ∈ {0,1,2}      |
         | · 漂移感知: E_drift (香农熵)    |
         | · Lyapunov 带宽队列             |
         | · network_sim 网络韧性层        |
         +--------------------------------+
```

### 压缩叙事（答辩核心，2026-08-09 cloud-teacher 正式版坐实）

**当前正式方案（cloud-teacher）**：云端冻结 **InternViT-6B**（全量视觉大模型）在独立离线标注集上训练场景分类头，再为代表性漂移样本生成 `{features, logits}` 教师监督；边缘 AdaptFormer 用该监督蒸馏训练，产出约 1.2MB 的 adapter-only checkpoint 下发。**云端大模型真在训练回路里产监督，adapter 是其蒸馏压缩产物**——"基于全量大模型压缩"叙事字面坐实。边缘 forward 只跑 MV-ViT-S + Adapter，不引入大 ViT/VLM 在线推理，不依赖云端实时返回。

**为什么 cloud-teacher（8/9 张晨 pivot）**：
- Qwen3-VL 当分类 teacher 失败（Top-1 56-62% < baseline 88-94%），VLM 不擅长细粒度车辆品牌；
- 把 VLM 当"环境表征经 FiLM 调制 adapter"也验证为**无明显贡献或负增益的消融**——VLM 擅长高层语义，错误用作 illumination/blur/noise 等**低层协变量漂移**的控制器；
- 改用大型**视觉**基础模型 InternViT-6B 当 teacher：它对退化域判别监督更稳健（cloud teacher 自身 mean drift 89.83% vs edge 81.97%），蒸馏出的 adapter 在无标签下 +3.24pp。

**FiLM / VLM-conditioned（已降为消融）**：`export_boxcars_vlm_hidden_states.py` / `prepare_vlm_condition_cache.py` / adaptformer 的 FiLM 扩展 / model 的 drift_conditioner 代码保留作消融证据，**不再作正式方案**。张晨 PR#30 body 明确"未带入旧 VLM-conditioned"。

### 协同动作语义（u_t，cloud-teacher 口径）
- **u=0 本地自治**：边缘冻结主干+adapter 直接推理，c_comm=0，离线/弱网可用
- **u=1 cloud-teacher adapter refresh**：上传代表性漂移样本，云端 InternViT 产监督+更新约 0.3M 参数，下发约 1.2MB adapter；与既有通信预算兼容
- **u=2 重训权重同步**：结构性漂移或任务变化，更新主干/分类头等更大范围；默认异步后台进 Q_net 跨槽消化

### 调度与理论
- **Actor-Critic + Lyapunov + 注水算法**：Actor(DNN) 生成候选 (v,u)，Critic(注水闭式解) 算最优 k_t，Lyapunov 虚拟队列 Y_bw 保证长期平均带宽约束
- **理论证明**：Slater 可行性、虚拟队列强稳定、O(1/V) 效用逼近界、KKT 注水闭式解。证明建立在**单一虚拟队列 Y_bw** 上，Q_net 不进 G 罚项
- **Dora**：把重训练对象换成小 adapter，尺度匹配时间槽

---

## 四、代码结构

```
EdgeCloud/
├── module_edge_perception/          # 模块一：边缘实时感知
│   ├── model.py                     # EarlyFusionMultiViewViT（含 drift_conditioner/bad_view_token/return_features，FiLM 消融用）
│   ├── adaptformer.py               # AdaptFormer PEFT（含 FiLM 扩展；cloud-teacher 用 condition_dim=0 无条件）
│   ├── verify_adaptformer.py        # adapter 三点验收脚本
│   ├── boxcars_dataset.py / boxcars_camera_drift_dataset.py / boxcars_drift_dataset.py  # BoxCars 数据 + 相机退化
│   ├── train_boxcars.py             # BoxCars baseline DDP 训练（30 轮，test Top-1=88.04%）
│   ├── calibrate_boxcars_camera_corruptions.py / calibrate_modelnet_camera_corruptions.py  # 退化档校准
│   │ ===== cloud-teacher 正式管线（张晨 8/9，PR#30 已合）=====
│   ├── train_boxcars_cloud_teacher_head.py        # 训云端 InternViT-6B 上的 BoxCars linear/MLP head
│   ├── export_boxcars_cloud_teacher_cache.py       # 冻结 InternViT 提特征 + head 产 {features,logits} cache
│   ├── build_boxcars_teacher_cache_from_feature_cache.py  # 从特征缓存建教师缓存
│   ├── train_boxcars_cloud_teacher_adapter.py      # 用教师 cache 蒸馏 AdaptFormer（CE+KL+特征对齐+anchor）
│   ├── retrain_boxcars_cloud_teacher_head_from_cache.py    # 从缓存重训 head
│   ├── modelnet_cloud_teacher_refresh.py          # ModelNet40 cloud-teacher 刷新（549行）
│   ├── modelnet_camera_drift_dataset.py           # ModelNet 相机漂移数据集
│   ├── evaluate_boxcars_cloud_teacher.py / evaluate_boxcars_cloud_teacher_quick_test.py  # cloud-teacher 评估
│   ├── summarize_boxcars_cloud_teacher_pipeline.py  # 流水线汇总
│   │ ===== FiLM / VLM-conditioned 消融（保留，不作正式方案）=====
│   ├── export_boxcars_vlm_hidden_states.py  # Qwen3-VL 视觉特征导出（消融）
│   ├── prepare_vlm_condition_cache.py       # VLM 特征→PCA 128→z_vlm（消融）
│   ├── train_boxcars_camera_adapter.py     # 漂移校正 adapter（label-only 上界，8/6，+5-11pp）
│   ├── train_boxcars_drift_adapter.py / evaluate_boxcars_drift_adapters.py  # 旧 drift adapter
│   ├── train_modelnet_drift_adapter.py / evaluate_modelnet_drift_adapter.py  # ModelNet drift adapter
│   ├── train_adapter.py                     # 旧 VLM 蒸馏版（已弃，保留对照）
│   ├── export_boxcars_soft_labels.py / export_boxcars_vlm_soft_labels.py  # 软标签导出（旧）
│   ├── train_boxcars_retrain_drift.py / train_boxcars_token_prompt.py  # 旧管线
│   ├── dataset.py / drift_dataset.py / train.py / train_retrain_drift.py / train_token_prompt.py  # 场景一 ModelNet40 旧管线
│   ├── prompt_tuning/               # PromptGenerator（可选辅助；两套实现待清理）
│   └── benchmarks/                  # B 的评测脚本
│       ├── benchmark_latency.py / benchmark_memory.py / benchmark_full.py  # TTFT/内存/延迟（已达标）
│       ├── benchmark_centralized.py # 集中式基准对比（B）
│       ├── benchmark_e2e.py         # 端到端时延评测（B）
│       ├── build_scheduler_latency.py / demo_mvp.py
│       └── results/                 # CSV + PNG 实测结果
├── module_scheduling/               # 模块二：云边协同调度
│   ├── EdgeCloud_RL/
│   │   ├── main_edge_cloud_new.py   # 单节点主循环（已接 network_sim）
│   │   ├── main_edge_cloud.py       # 【已失效】旧版，勿用
│   │   ├── critic_water_filling.py  # 注水闭式解 Critic
│   │   ├── actor_memory.py          # Actor DNN + 经验回放
│   │   ├── network_sim.py           # 网络韧性模拟器（唐凤玲，四档+Q_net+TTL+业务五条件）
│   │   ├── run_network_resilience_tests.py  # 【8/9新增】网络韧性测试主脚本（产出四档结果表）
│   │   ├── generate_real_trajectory.py / generate_boxcars_trajectory.py  # 两场景轨迹生成
│   │   ├── evaluate_rl_policy_on_mvvit.py / evaluate_train_policy_on_mvvit.py
│   │   └── plot_*.py
│   ├── comparison_baselines/        # LSCI/VBRD/Hyperion 基线（有意弱化，答辩公平性有风险）
│   └── multi_node/                  # 多节点仲裁
│       ├── arbiter.py               # Arbiter 类（冲突检测+加权投票/贝叶斯融合+回滚+统计）
│       └── multi_node_eval.py       # 真实多节点多视角评估（4节点各看3缺1，待AutoDL跑）
├── common/drift_dataset.py          # 确定性漂移主实现；注意 EdgeCloud_RL/ 下有副本
├── data/  data/README.md           # 数据集不进 Git（ModelNet40 ~2GB、BoxCars116k）
├── models/ models/README.md         # 权重不进 Git（checkpoints/*.pth 为占位，真权重在 AutoDL）
├── docs/
│   ├── 云端视觉教师Adapter方案_20260809.md  # 【正式方案】cloud-teacher 总方案（PR#30）
│   ├── 第一场景_ModelNet40.md                # 【新】ModelNet 场景定义（PR#30）
│   ├── 网络波动模拟器设计.md         # 网络韧性设计（PR#31 第二次测试结果版）
│   ├── 方案总览.md / 接口契约.md     # 待补 cloud-teacher 叙事
│   ├── 第二场景_BoxCars116k.md       # BoxCars 场景定义（baseline 88.04%）
│   ├── 漂移校正Adapter方案.md / 多场景漂移校正Adapter实验总览_20260806.md  # 8/6 drift-correction（label-only 上界）
│   ├── BoxCars116k_相机退化Adapter实验记录_20260806.md / ModelNet40_Adapter相机退化实验_20260806.md
│   ├── VLM软标签验证.md             # 【已过时】VLM 品牌软标签（pivot 前）
│   └── 环境配置避坑指南.md
├── scripts/verify_env.py / deploy_a3.sh
└── requirements.txt
```

### 已达标硬指标（实测，2026-08-10 快照）
| 指标 | 实测 | 状态 |
|---|---|---|
| 参数量 | 21.7M (ViT-Small) | ✅ |
| TTFT 降幅 | 86.8%（287→38ms，keep=0.1） | ✅ |
| 推理内存 | 0.08GB | ✅ |
| 单帧延迟 | 38ms | ✅ |
| BoxCars baseline 精度 | Top-1=88.04% / Top-5=97.13%（30 轮 test，场景二） | ✅ 场景二 |
| ModelNet40 baseline 精度 | clean 96.76%（test 全集） | ✅ 场景一 |
| **cloud-teacher Adapter（BoxCars，正式）** | cloud-unlabeled mean drift 85.21% vs edge 81.97%（**+3.24pp**）；noise +6.93pp 显著，illum +1.23/blur +1.54（256条 CI 跨0）；clean 94.61% 不退化 | ✅ 真数（叙事合法） |
| cloud-teacher 教师（InternViT-6B 本身） | mean drift 89.83%，clean 96.15% | ✅ 教师稳健 |
| label-only Adapter（有标签上界，8/6） | mean drift 86.08%，+5-11pp 逐项 | ✅ 上界对照 |
| 能力保持（漂移损失恢复率口径） | cloud-teacher 无标签口径已交，label-only 口径待统一 | ✅ 口径待发榜确认 |
| adapter 下发体积 | 299,916 参 / 1,216,745 bytes ≈ 1.2MB（不含 backbone/norm/head/projector） | ✅ |
| 业务保持率（四档） | 见网络文档第二次测试结果（PR#31 已合） | ⚠️ e2e 口径见下 |
| 端到端时延 | 80ms（T_edge 写死，u=1/u=2 异步 realtime_comm=0，**主实验未含通信**） | ⚠️ 待含通信 |
| 冲突率/解决率 | arbiter 模拟达标，真 multi_node 待跑 | ⚠️ 待真数 |
| ≥2 类场景 | ModelNet40 + BoxCars116k | ✅ |

---

## 五、网络韧性注入设计（唐凤玲方案；PR#31 已合第二次测试结果）

在单节点主循环外加一层"时变网络 + 时延 + 业务可用性"，不动原有 Lyapunov 证明。

### 核心原则
- **Q_net 是物理传输积压队列（状态量），不进效用目标 G 的罚项**；长期平均带宽约束仍由 Y_bw 单一虚拟队列保证，Lyapunov 证明不变。
- **Y_bw 更新**：`Y_bw[t+1] = max(Y_bw[t] + c_comm - b_avg, 0)`
- **Q_net 更新**：追踪瞬时积压，带上限(Q_net_max=4×b_avg)+TTL(50时隙)丢弃防无限增长
- **B_t = R_eff_t × slot_duration / 8**（R_eff 已含 1-p 折扣）

### 四档网络模式（含 static 基线做消融）
- `static`：不模拟弱网，基线对照
- `jitter`：带宽抖动，不断联
- `jitter_outage`：jitter + 随机断联(disconnect_prob) + 周期断联(outage_period+outage_duration)
- `markov`：GOOD/WEAK/DOWN 三态弱网（主模型，转移矩阵：GOOD→GOOD 0.92/WEAK 0.07/DOWN 0.01 等）

### 关键处理
- **断联 → 强制 u=0**（只保留本地自治候选）
- **超带宽 → 软罚或硬过滤**：`G_effective = G_raw - overflow_penalty × (comm_overflow / B_t)`；`--strict-bandwidth` 直接过滤
- **u=1/u=2 均异步后台**：实时链路只计下发 T_comm，生成/重训时延 T_cloud=0；仅 `--sync-u2` 才把 u=2 重训时延计入
- **端到端时延**：`T_e2e = T_edge + T_comm + T_cloud`；`T_comm = 8 × realtime_comm / R_eff × 1000 + RTT`；u=0 时 T_comm=0（特判）
- **业务可用五条件**：`business_available = decision_success AND transmission_success AND e2e≤deadline AND active_views≥min AND proxy_acc≥acc_floor`。acc_floor 默认 0.8；transmission_success 不可省

### 第二次测试结果摘要（2026-08-09，PR#31）
- 四档 × 两场景业务保持率 99.83-99.96%，avg_e2e 80ms（**注：默认异步口径 realtime_comm=0，主实验 e2e 不含通信**）
- 通信探针（强制实时传 payload）：1.2MB adapter 在 GOOD 下 170ms 达标；6.2MB/55MB 严重超标 → 大包异步队列必要性成立
- u=2 高漂移验证：触发率 5.22%，max_Q_net 49.96MB，SCL 完成率 54.64%、drop 45.26%
- **已知口径风险**：e2e 主实验不含通信（写死 edge_delay=80ms），业务保持率因 u=0 transmission_success 特判仍偏高——8/10 前需落地"含通信 e2e"一版

---

## 六、关键路径与依赖（8/10 更新）

### 主关键路径
```
adaptformer(✅8/3) → 漂移校正 label-only 上界(✅8/6) → cloud-teacher 正式方案(✅8/9 PR#30) → 8/10 硬指标定稿
ModelNet cloud-teacher + 全量 test(⏳待跑) → 8/17-8/21 收尾
```
**当前状态：8/10 用 cloud-teacher（cloud-unlabeled）真数交硬指标，叙事合法。label-only 作有标签上界对照。**

### 实际进度
- ✅ adaptformer.py + verify（8/3）
- ✅ VLM 分类 teacher 失败 → 漂移校正 pivot（8/5）→ FiLM/VLM-conditioned 验证为消融（8/6-8/9）
- ✅ 两场景相机退化校准 + label-only adapter（8/6，+5-11pp 上界）
- ✅ cloud-teacher 正式方案：InternViT-6B 教师 + 无标签 adapter 刷新，BoxCars 出真数（8/9，PR#30 已合）
- ✅ network_sim.py + 接主循环 + 四档验收 + 第二次测试结果文档（8/9，PR#31 已合）
- ✅ multi_node arbiter + multi_node_eval 入库（真数待跑）
- ⏳ ModelNet cloud-teacher 全量（modelnet_cloud_teacher_refresh.py 已入库，结果待跑）
- ⏳ cloud-unlabeled 全量 test（当前仅 256 条快速 check）
- ⏳ label-only 8/6 数与 cloud-teacher 文档 label-only 数口径对齐（待张晨统一）
- ⏳ B 能力保持曲线 + 硬指标汇总
- ⏳ 含通信 e2e + 唐凤玲 acc_floor 0.8 重跑
- ⏳ 王成洋 multi_node 真冲突率 + 系统联调

### 8/10 硬指标交付清单
| 指标 | 负责人 | 当前 |
|---|---|---|
| 参数量/TTFT/内存/延迟 | B | ✅ 达标 |
| 能力保持（80-90%） | B+C | ✅ cloud-teacher cloud-unlabeled 真数（+3.24pp），口径待发榜确认 |
| 端到端时延≤0.2s | B+唐凤玲 | ⚠️ 80ms 未含通信，待落地通信探针口径 |
| 业务保持率≥90% | 唐凤玲 | ✅ 99.8%（PR#31），但 e2e 口径有风险 |
| 冲突≤5%/解决≥90% | 王成洋+B | ⚠️ arbiter 模拟达标，真 multi_node 待跑 |
| ≥2类场景 | C | ✅ ModelNet40+BoxCars（ModelNet cloud-teacher 待跑） |

---

## 七、关键风险与待确认（8/10 更新）

1. **~~"VLM 压缩"叙事未落实~~ → ✅ 已解决（8/9 cloud-teacher 落地）**：云端 InternViT-6B 真在训练回路产 {features,logits} 监督，adapter 是其蒸馏压缩产物（1.2MB），"基于全量大模型压缩"叙事字面坐实。FiLM/VLM-conditioned 降为消融。**新代价**：cloud-unlabeled（85.21%）绝对精度低于 label-only 上界（86.08%）0.87pp，是"用精度换叙事合法性"——交数时主推 cloud-unlabeled，label-only 作有标签上界对照，不能把 label-only 的 +5-11pp 算到云端大模型头上。
2. **"80-90% 能力保持"指标口径**——赛题写"数学/代码/NLP"，本项目视觉任务。现用"漂移下相对 edge baseline 的提升/损失恢复率"口径。**需老师问发榜单位：视觉任务能否按此口径算 80-90% 能力保持？**
3. **cloud-unlabeled 全量 test 未跑**——当前 test 只 256 条快速 check，illumination/blur 的小提升 CI 跨 0，**仅 noise +7.81pp（95% CI [+3.91,+11.72]）显著**。8/17 前需跑全量 test（官方 12,322 条）定终稿。
4. **label-only 数字口径不一致**——8/6 文档 label-only illumination +6.78pp，cloud-teacher 文档 label-only illumination 83.05 vs 81.82 = +1.23pp。同一 label-only 两个数差很大，大概率 drift 严重度/baseline 口径不同。**需张晨统一口径**，否则两份文档数字打架。
5. **业务保持率/e2e 口径风险**——e2e 主实验写死 edge_delay=80ms + 异步 realtime_comm=0，**不含通信**；业务保持率因 u=0 transmission_success 特判仍偏高。8/10 前需落地"含通信 e2e"一版（详见第五章口径风险）。
6. **ModelNet cloud-teacher 未跑**——ModelNet 场景 cloud-teacher 结果待跑（modelnet_cloud_teacher_refresh.py 已入库）。≥2 类场景硬指标需 ModelNet 也有数。
7. **基线公平性**——comparison_baselines 的 LSCI/VBRD/Hyperion 被有意弱化；至少一个基线用未弱化实现。
8. **prompt/adapter 双轨**——prompt 代码不删作可选辅助，两套互不兼容实现属待清理技术债。
9. **InternViT-6B 资源**——5.9B 参数，RTX 3090 BF16 batch=1 离线特征提取。需确认张晨服务器跑得动、教师 cache 已产完。DINOv3 作低层 OOD 对照（7B 不适合 30GB 磁盘环境）。
10. **代码与文档同步**——cloud-teacher 全套入库+PR#30 合(✅)、网络文档第二次测试版+PR#31 合(✅)、CLAUDE.md 本文件已对齐 cloud-teacher(✅8/10)。**方案总览.md / 接口契约.md 待补 cloud-teacher 叙事**。

---

## 八、AdaptFormer 模块技术规格（王成洋 8/3 实现；张晨扩展；verify 验收通过）

- **挂载点**：每个 timm Transformer block 的 mlp（FFN）旁路并行
- **瓶颈维度 r**：32（vit_small embed_dim=384，压缩比 12:1，每层 ~24K 参数，12 层共 ~290K，加 scale 共 299,916）
- **结构**：`down(D→r) → GELU → up(r→D)`，W_up 零初始化使启动时 adapter 输出为 0（不破坏预训练主干）
- **FiLM 条件化扩展（8/6 张晨，已降为消融）**：`AdaptFormerMLP` 加 `condition_dim` + `film` 层（`Linear(condition_dim, 2r)`，零初始化）；forward 里 `h = h*(1+gamma) + beta`。**cloud-teacher 用 `condition_dim=0` 无条件 adapter**，FiLM 路径不激活
- **实现方式**：`AdaptFormerMLPWrapper` 包裹 `block.mlp`，`forward = mlp(x) + scale × adapter(x, condition)`；替换后三条前向路径自动生效
- **冻结策略**：主干全部 `requires_grad=False`，只训 adapter + norm + head；cloud-teacher 训练期还开放 drift_conditioner/bad_view_token（FiLM 消融用）
- **save/load/enabled**：`save_adapter_checkpoint`（存 1.2MB adapter-only，**cloud-teacher 支持不含 norm/head 的纯 adapter 导出**）、`load_adapter_checkpoint`、`set_adapter_enabled`、`collect_adapter_state`
- **验收**（verify_adaptformer.py）：①零初始化 Δ=0；②adapter 0.300M<1M；③三条前向路径+wrapper hook 全触发
- **cloud-teacher 训练目标**（train_boxcars_cloud_teacher_adapter.py）：`L = CE(z_S,y) + λ_KD·KL(z_S‖z_T) + λ_f·(1-cos(P(h_S),h_T)) + λ_a·KL(z_S‖z_0)`。z_T/h_T=云端 InternViT 教师，z_0=clean replay baseline anchor，P=云端训练 projector（不下发）。teacher 用真实标签 CE 兜底防教师品牌误判主导
- **参考论文**：AdaptFormer (ICCV 2022)、FiLM、LoRA (ICLR 2022)、Houlsby Adapter (ICML 2019)、Hinton 蒸馏 (2014)、InternViT (OpenGVLab)

---

## 九、环境与访问

- **训练环境**：AutoDL 云 GPU（数据路径默认 `/root/autodl-tmp/`），N 卡。张晨有账号；王成洋/B/唐凤玲可自注册
- **公司本机限制**：Python 3.14 的 torch DLL 被应用控制策略阻止（import torch 失败）；github.com 443 命令行/TCP 直连被拦但 Invoke-WebRequest 走 HTTPS 放行；没装 git/gh。公司本机只能 Read/Grep/Glob 文件 + 浏览器操作 GitHub 网页 + AutoDL 网页 JupyterLab + PowerShell 调 GitHub REST API
- **GitHub 私仓访问（8/10 已通）**：fine-grained PAT 存用户级环境变量 GITHUB_TOKEN（注册表 HKCU\Environment，子进程读不到要直读注册表）；走 GitHub REST API + Invoke-WebRequest（不装 git）；权限 Contents/Pull requests/Metadata read+write，7 天有效。详见 memory/github-private-repo-access.md
- **AutoDL 自注册**：autodl.com 注册+实名+充值，镜像选 PyTorch2.x+Python3.11，卡选 4090 24G；InternViT-6B 教师需更大显存（3090 BF16 batch=1 离线）

---

## 十、给新会话的快速恢复提示

1. 先读本文件恢复团队/架构/决策上下文
2. 读 `docs/云端视觉教师Adapter方案_20260809.md`（cloud-teacher 正式方案，PR#30）
3. 读 cloud-teacher 训练管线：`train_boxcars_cloud_teacher_head.py` → `export_boxcars_cloud_teacher_cache.py` → `train_boxcars_cloud_teacher_adapter.py` → `evaluate_boxcars_cloud_teacher.py`
4. 读 `module_edge_perception/adaptformer.py`（condition_dim=0 无条件版）+ `model.py` + `verify_adaptformer.py`
5. 读 `module_scheduling/EdgeCloud_RL/main_edge_cloud_new.py` + `critic_water_filling.py` + `network_sim.py` + `run_network_resilience_tests.py`
6. 读 `module_scheduling/multi_node/arbiter.py` + `multi_node_eval.py`
7. 读 `docs/网络波动模拟器设计.md`（第二次测试结果版）
8. 若需消融背景：`docs/漂移校正Adapter方案.md`（8/6 label-only 上界）+ FiLM 消融脚本（export_boxcars_vlm_hidden_states.py / prepare_vlm_condition_cache.py）
9. 用 `git log --oneline -20` 看最近提交
10. 所有回复用中文
11. 若需接续进行中任务，看 `docs/交接_<日期>.md` 了解当前卡点与下一步

**当前最紧要**：8/10 用 cloud-teacher（cloud-unlabeled）真数交硬指标，叙事合法。待办：ModelNet cloud-teacher 跑全量、cloud-unlabeled 全量 test、label-only 口径统一、含通信 e2e、multi_node 真冲突率。答辩不急（8/31 提交，答辩时间未定）。
