# EdgeCloud · 面向端边云协同推理的分布式感知与全局优化决策系统

> 2026"揭榜挂帅"擂台赛 · 赛题 XH-202606  
> **核心思路**：云端国产大模型（DeepSeek/VLM）充当"知识预言机"，将场景理解**压缩为轻量 Prompt Token** 注入边缘小模型；边缘 MV-ViT 负责实时多视角推理；调度层通过 Actor-Critic + Lyapunov 优化在线决策。  
> **知识压缩替代参数压缩，Prompt 注入替代模型蒸馏。**

## 团队成员

| 成员 | 负责模块 | 职责 |
|------|---------|------|
| 王成洋（队长） | 系统集成 + 多节点协同 | 总体架构、接口统筹、多节点冲突检测与仲裁 |
| 待定 | 模型评测 | TTFT / 内存 / 延迟测量 |
| 待定 | 第二场景 | 第二应用场景数据集适配与全流程训练 |
| 待定 | 网络韧性 + 评测 | 网络波动模拟、连续性指标、评测自动化 |
| 待定 | 文档 / 实验统筹 | 技术报告、对比图表、Demo 视频、进度追踪 |

指导老师：周睿婷、东方、孟杰

## 目录结构

```
EdgeCloud/
├── module_edge_perception/    # 模块一：边缘实时感知
│   ├── model/                 #   MV-ViT + Token 剪枝
│   ├── drift/                 #   漂移模拟（5种）
│   ├── prompt/                #   Prompt 生成与注入
│   └── benchmarks/            #   TTFT / 内存 / 延迟测量
│
├── module_scheduling/         # 模块二：云边协同调度
│   ├── scheduler/             #   Actor-Critic RL + Lyapunov
│   ├── vlm_oracle/            #   云端 VLM 模拟接口
│   ├── multi_node/            #   多节点冲突检测与仲裁
│   └── network_sim/           #   网络波动模拟
│
├── data/                      # 数据集（不进 Git）
├── models/                    # 模型权重（不进 Git）
├── experiments/               # 实验脚本与结果
├── scripts/                   # 一键运行脚本
├── docs/                      # 方案文档
└── requirements.txt           # Python 依赖
```

## 技术架构

```
         +---------- 云端 Cloud ----------+
         | DeepSeek / VLM "知识预言机"    |
         | · 分析关键帧 -> 判断环境状态   |
         | · P_env (2KB) + w_t (4标量)   |
         | · 多节点冲突仲裁               |
         +---+--------------------+------+
   上行关键帧 |                    | 下行 Prompt (几KB)
             |                    v
         +--------------------------------+
         |    边缘 Edge（路口盒子）        |
         | MV-ViT (ViT-Small, 2200万)     |
         | · Token剪枝: k_t               |
         | · 视角选择: v_t                |
         | · 协同模式: ut in {0,1,2}      |
         | · 漂移感知: Edrift (香农熵)    |
         | · Lyapunov 带宽队列             |
         +--------------------------------+
                  ^ 4路摄像头
                  |
         +--------+--------+
         |   端 Device      |
         |   多源感知数据    |
         +-----------------+
```

## 方案亮点

1. **知识压缩替代参数压缩**：DeepSeek 在云端分析场景，把理解压缩成 4 个 Prompt Token（2KB）注入边缘 ViT，而非把 175B 模型蒸馏到 10M
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