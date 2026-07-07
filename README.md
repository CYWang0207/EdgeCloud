# EdgeCloud · 面向端边云协同推理的分布式感知与全局优化决策系统

> 2026"揭榜挂帅"擂台赛 · 赛题 XH-202606「面向云边协同场景的分布式人工智能感知与决策关键技术研究」
>
> 一句话简介：打破以云为中心的集中式 AI，让**感知、推理、决策**在端 / 边 / 云三层之间
> 合理分工——边侧轻量化实时推理 + 云边协同在线调度 + 云端持续学习与全局一致性仲裁，
> 三者咬合成一个自洽闭环。

## 团队成员

| 成员 | 分组 | 负责模块 |
|---|---|---|
| 王成洋（队长） | 周老师组 | 系统集成 + 决策优化 |
| 待定 | 周老师组 | 模块 B · 云边协同 |
| 待定 | 周老师组 | 模块 C · 持续学习与全局决策 |
| 待定 | 东方老师组 | 模块 A · 轻量化-压缩 |
| 待定 | 东方老师组 | 模块 A · 预测器/部署 |

指导老师：周睿婷、东方、孟杰

## 目录结构

```
EdgeCloud/
├── module_a_lightweight/   # 模块 A：模型轻量化/压缩（AutoTailor）
├── module_b_scheduling/    # 模块 B：云边协同推理与任务调度（Lyapunov Actor-Critic）
├── module_c_global/        # 模块 C：持续学习与全局决策（Replay Buffer + 一致性仲裁）
├── common/                 # 三模块共用代码（数据格式定义、接口约定实现、公共工具）
├── docs/                   # 规划文档、接口契约文档
├── data/                   # 数据集（不进 Git，网盘管理，目录内有下载说明）
├── models/                 # 模型权重（不进 Git，网盘管理，目录内有下载说明）
└── requirements.txt        # Python 依赖（待补充）
```

各模块职责与对标指标见对应目录下的 README；三模块间的接口约定见
[docs/接口契约_v0.md](docs/接口契约_v0.md)；整体规划见
[docs/整体规划_v1.md](docs/整体规划_v1.md)。

## 如何克隆和开始开发

```bash
# 1. 克隆仓库（私有仓库，需先被添加为协作者并配置 GitHub 认证）
git clone https://github.com/CYWang0207/EdgeCloud.git
cd EdgeCloud

# 2. 创建并激活虚拟环境（Python 3.10+ 建议）
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载数据集与模型权重（不在 Git 中！）
#    按 data/README.md 和 models/README.md 中的网盘链接下载并放到对应目录

# 5. 开发流程
#    - 从 main 拉出功能分支：git checkout -b feat/<模块>-<功能>
#    - 提交后推送并发起 Pull Request，由模块负责人评审合并
#    - 涉及跨模块接口的改动，先更新 docs/接口契约_v0.md 达成一致
```

## ⚠️ 大文件红线

**模型权重（\*.pth / \*.onnx / \*.bin 等）和数据集一律不进 Git**，统一用团队网盘管理，
仓库只放代码与下载说明。`.gitignore` 已做拦截，但提交前请自查 `git status`。
