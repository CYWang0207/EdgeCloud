# EdgeCloud 实验复现导航

## 为什么需要本文件

`docs/` 中的实验报告负责解释“为什么这样设计、实验如何开展、结果意味着什么”；本文件只负责回答“评委需要准备什么、运行哪条命令、到哪里查看输出”。因此它是实验报告的统一入口，不替代也不重复各项报告。

## 统一环境

推荐环境：Ubuntu 22.04、Python 3.11、NVIDIA GPU、CUDA 12.8、PyTorch 2.8.0、timm 1.0.15。依赖与路径默认值见 `requirements.txt`、`configs/submission.yaml` 和 `RELEASE_ASSETS.md`。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_env.py
```

`verify_env.py` 默认使用随机初始化，不下载数据和模型。增加 `--pretrained` 会通过 timm 下载 ImageNet 预训练权重。

## 输入材料约定

```text
data/
├── modelnet40v2png_ori4/
└── BoxCars116k_kaggle/BoxCars116k/

models/
├── InternViT-6B/
├── mv_vit_token_epoch_30.pth
├── boxcars_make_baseline/best.pth
├── modelnet40_cloud_teacher_adapter/cloud_unlabeled/best.pth
└── boxcars_cloud_teacher_adapter/cloud_unlabeled/best.pth

artifacts/inputs/
├── trajectory_modelnet40.csv
└── trajectory_boxcars.csv
```

数据来源、目录结构、权重版本与 SHA-256 分别见 `data/README.md`、`models/README.md` 和 GitHub Release。

## 复现矩阵

| 实验 | 入口命令 | 主要输出 | 对应报告 |
|---|---|---|---|
| 环境与模型结构 | `python scripts/verify_env.py` | 四视图前向与计时 | README |
| AdaptFormer 结构 | `python module_edge_perception/verify_adaptformer.py` | 参数量、零初始化、前向检查 | 感知模块 README |
| ModelNet40 | `python scripts/reproduce_modelnet40.py` | gate、checkpoint、完整 test 指标 | `docs/第一场景_ModelNet40.md` |
| BoxCars116k | `python scripts/reproduce_boxcars.py` | 完整 test 指标与 CI | `docs/第二场景_BoxCars116k.md` |
| 网络韧性 | `python scripts/reproduce_network.py` | 四档网络 CSV/JSON | `docs/网络波动模拟器设计.md` |
| 多节点仲裁 | `python scripts/reproduce_multinode.py` | 冲突、融合与回滚记录 | `docs/多节点冲突仲裁测试结果_20260811.md` |

## 口径说明

1. ModelNet40 与 BoxCars 的正式精度均使用完整 official test，不使用 quick subset 作为最终结论。
2. 两类场景中的光照、模糊、失焦和传感器噪声均为可控合成退化。
3. 网络韧性结果来自仿真。默认异步口径下 Adapter 后台下发不阻塞当前前台业务。
4. `T_edge=80 ms` 是网络模拟器固定参数；含 1.2 MB Adapter 前台下发的 92–111 ms 也是仿真结果。
5. 15 秒网页回放用于展示阶段变化，不代表训练或推理真实耗时。

## 完整性检查

```bash
shasum -a 256 -c MANIFEST.sha256
git status --short
git rev-parse HEAD
```

输出目录应包含运行配置、Git commit、随机种子、硬件信息、汇总指标和逐样本或逐时隙记录。不要覆盖随提交包提供的正式证据，建议写入新的 `artifacts/*/reproduced/` 目录。
