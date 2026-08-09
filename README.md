# EdgeCloud · 面向端边云协同推理的分布式感知与全局优化决策系统

> 2026"揭榜挂帅"擂台赛 · 赛题 XH-202606  
> **当前核心思路**：云端冻结大型视觉基础模型 InternViT-6B，并训练 BoxCars 任务头；其漂移样本 logits/features 被压缩进轻量 AdaptFormer。边缘只运行 MV-ViT-S + adapter，调度层通过 Actor-Critic + Lyapunov 决定何时刷新。
> **压缩的不是参数，是云端大模型蒸馏出来的场景知识；传输的不是模型，是压缩后的小适配器。**  
> 当前 worktree 不再维护 Prompt/VLM-conditioned 路线；历史实现统一放在 `local/archive/`。

## 团队成员

| 成员 | 负责模块 | 职责 |
|------|---------|------|
| 王成洋 | 系统集成 + 多节点协同 | 总体架构、接口统筹、多节点冲突检测与仲裁 |
| 钟捷杭 | 模型评测 | TTFT / 内存 / 延迟测量 |
| 张晨 | 第二场景 | 第二应用场景数据集适配与全流程训练 |
| 唐凤玲 | 网络韧性 + 评测 | 网络波动模拟、连续性指标、评测自动化 |
| 苏程鑫 | 文档 / 实验统筹 | 技术报告、对比图表、Demo 视频、进度追踪 |

指导老师：周睿婷、东方、孟杰

## 目录结构

```
EdgeCloud/
├── module_edge_perception/         # 模块一：边缘实时感知
│   ├── model.py                     #   EarlyFusionMultiViewViT（多视角早期融合 + token剪枝 + 视角掩码）
│   ├── adaptformer.py              #   AdaptFormer PEFT 模块（已落地，8/3 验收通过）
│   ├── export_boxcars_cloud_teacher_cache.py # 云端视觉监督导出
│   ├── train_boxcars_cloud_teacher_adapter.py # 只训练/下发 adapter
│   ├── verify_adaptformer.py        #   AdaptFormer 三点验收脚本（零初始化/参数量/三条前向）
│   ├── boxcars_dataset.py           #   场景二 BoxCars116k 数据加载
│   ├── dataset.py / drift_dataset.py
│   ├── train*.py                    #   Edge baseline 与 cloud-guided Adapter
│   └── benchmarks/                  #   TTFT / 内存 / 延迟 / 一体化测量
│
├── module_scheduling/              # 模块二：云边协同调度
│   ├── EdgeCloud_RL/                #   Actor-Critic RL + Lyapunov 主循环 + 注水 Critic + 网络模拟器
│   ├── comparison_baselines/       #   LSCI/VBRD/Hyperion 基线
│   ├── multi_node/                 #   【待建】多节点冲突检测与仲裁
│
├── common/                          # 漂移模拟器（5种×6档 schedule）
├── data/                            # 数据集（不进 Git：ModelNet40、BoxCars116k）
├── models/                          # 模型权重（不进 Git；checkpoints 当前为 0 字节占位）
├── scripts/                         # 一键运行脚本
├── docs/                            # 方案文档
└── requirements.txt                 # Python 依赖
```

## 技术架构

```
         +---------- 云端 Cloud ----------+
         | InternViT-6B 视觉教师           |   ← 冻结 backbone
         | · BoxCars task head             |
         | · 输出漂移 logits / features    |
         | · 多节点冲突仲裁                |
         +---+--------------------+------+
   上行关键帧 |                    | 下行 adapter 参数（几百KB~MB）/ 重训权重
             |                    v
         +--------------------------------+
         |    边缘 Edge（路口盒子）        |
         | MV-ViT (ViT-Small, 2200万) 主干冻结
         | + AdaptFormer adapter (可训练) |
         | · Token剪枝: k_t               |
         | · 视角选择: v_t                |
         | · 协同模式: ut in {0,1,2}      |
         | · 漂移感知: Edrift (香农熵)     |
         | · Lyapunov 带宽队列            |
         +--------------------------------+
                  ^ 4路摄像头
                  |
         +--------+--------+
         |   端 Device      |
         |   多源感知数据    |
         +-----------------+
```

## 方案亮点

1. **鲁棒视觉监督压缩**：云端大型视觉模型产生任务对齐 logits/features，只把约 0.3M 参数的 AdaptFormer 增量下发边缘
2. **多粒度资源调度**：每时隙联合决策视角选择、Token 剪枝率、协同模式，KKT 注水闭式解保证 ms 级决策
3. **理论保证**：Lyapunov 强稳定性证明、O(1/V) 效用逼近界、KKT 最优性条件
4. **多源异构**：四路摄像头天然异构（光照/遮挡/视角各异），MV-ViT 早期融合统一处理

## 快速开始

```bash
git clone https://github.com/CYWang0207/EdgeCloud.git
cd EdgeCloud
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 协作规范

- 分支命名：`feat/<模块>-<功能>`
- 合并前需至少 1 人 Code Review
- 每日 21:00 站会 10 分钟（飞书）
- 每日 GitHub Issue 更新进度

## 时间线

| 日期 | 里程碑 |
|------|--------|
| 7/30 | 环境跑通；延迟方案确认；第二场景数据集确认 |
| 8/3 | MVP 可运行 |
| 8/10 | 硬指标数字确认 |
| 8/17 | 代码冻结；报告初稿 |
| 8/21 | 最终交付 |
| 8/31 | 正式提交截止 |

## 大文件红线

权重（*.pth/*.onnx）和数据集不进 Git，统一用网盘管理。
