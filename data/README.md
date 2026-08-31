# data · 数据集目录（真实数据不进 Git）

本目录只提交说明文件。数据集、解压文件、生成的索引和模型权重均由
`.gitignore` 排除，禁止提交到仓库。

## 两个应用场景

```text
data/
├── README.md
├── modelnet40v2png_ori4/                 # 场景 1：四视角 3D 物体分类
└── BoxCars116k_kaggle/
    ├── BoxCars116k.zip                   # 原始压缩包，约 6.1 GB
    └── BoxCars116k/                      # 场景 2：监控车辆多视图分类
        ├── images/
        ├── json_data/
        │   ├── dataset.json
        │   ├── classification_splits.json
        │   └── verification_splits.json
        ├── matlab_data/
        ├── dataset.pkl
        └── INFO.txt
```

推荐在仓库根目录使用以下相对路径：

```text
data/BoxCars116k_kaggle/BoxCars116k
```

## 场景 2：BoxCars116k 交通监控车辆识别

第二场景已经从 SEVD 调整为 BoxCars116k。该数据集包含 27,496 辆独立车辆的
116,286 张监控裁剪图、693 个细粒度车型标签；服务器解压目录约 9.2 GB。
`images/` 下还为每张图保存一个 `_mask.png`，因此直接统计 PNG 会得到
232,572 个文件，训练图像数仍是 116,286。

- 官方项目与格式说明：https://github.com/JakubSochor/BoxCars
- 数据集论文：https://arxiv.org/abs/1703.00686

### 项目中的任务定义

第一版采用官方 `make` 划分做 **16 类车辆品牌识别**，而不是直接训练长尾明显的
693 个细粒度类别。官方划分按摄像头隔离训练集和测试集，比自行随机切分图片更能
检验跨摄像头泛化，也避免同一轨迹泄漏到不同集合。

BoxCars116k 的真实采集方式必须说明清楚：同一 `vehicle_id` 是一辆车在**同一个
物理交通摄像头**下形成的一段轨迹，数据集不提供跨摄像头车辆身份关联。因此项目
不能声称四个物理摄像头同时看见同一辆车。这里把一条轨迹中时间跨度尽可能大的
四张裁剪图作为四个逻辑视图：

```python
views, view_mask, label, metadata = dataset[index]

views.shape     # [4, 3, 224, 224]
view_mask.shape # [4]，真实图像为 1，补齐位置为 0
label           # 品牌类别索引，默认 0–15
```

抽样规则为首帧到末帧之间等间隔取 4 张，以获得车辆尺度、位置和可见面的最大变化。
若轨迹少于 4 张，保留全部真实图像并重复末张图补齐张量，同时将补齐位置的
`view_mask` 设为 0。模型已有 missing-view token，可用该掩码模拟摄像头缺失；
雨雾、明暗、模糊、遮挡等环境变化继续由漂移模块在线合成，不伪造数据集原本没有
的天气标签。

这种定义与项目目标的对应关系是：

- 多视图融合：融合同一车辆轨迹中不同时间、尺度和可见面的观测；
- Token 剪枝：对四个逻辑视图的 patch token 动态保留；
- 视图选择：根据 `view_mask` 和调度动作关闭低价值或故障视图；
- 场景漂移：在输入图像上模拟光照、雨雾、模糊、遮挡和设备故障；
- 输出任务：品牌分类，后续再评估 `medium`（79 类）或 `hard`（107 类）车型任务。

## DataLoader

加载器位于 `module_edge_perception/boxcars_dataset.py`：

```python
from module_edge_perception.boxcars_dataset import BoxCarsMultiView

dataset = BoxCarsMultiView(
    root_dir="data/BoxCars116k_kaggle/BoxCars116k",
    split="train",          # train / validation / test
    task="make",            # make / body / medium / hard
    num_views=4,
    transform=transform,
)
views, view_mask, label, metadata = dataset[0]
```

官方分类任务规模如下（数量单位为车辆轨迹）：

| 任务 | 类别数 | train | validation | test | 用途 |
|---|---:|---:|---:|---:|---|
| `make` | 16 | 13,098 | 649 | 12,322 | 第一版品牌分类，推荐 |
| `body` | 6 | 13,432 | 771 | 12,650 | 车身类型分类 |
| `medium` | 79 | 12,084 | 611 | 11,456 | 中等粒度车型分类 |
| `hard` | 107 | 11,653 | 637 | 11,125 | 较细粒度车型分类 |

不要自行按单张图片随机切分，也不要把 `_mask.png` 当作 RGB 输入。

## DataLoader 验证

从仓库根目录运行：

```bash
cd module_edge_perception
python test_boxcars_inference.py \
  --dataset-path ../data/BoxCars116k_kaggle/BoxCars116k \
  --task make --split test
```

该命令只验证真实 DataLoader、`view_mask` 和 MV-ViT 前向链路，不训练模型；随机
初始化分类头的输出不能作为识别精度。正式精度评测必须加载 BoxCars116k 训练得到
的 checkpoint。

正式训练默认使用 16 类品牌任务。双卡全视图 baseline（无 Token 剪枝）运行：

```bash
cd module_edge_perception
torchrun --standalone --nproc_per_node=2 train_boxcars.py \
  --task make --batch-size 4 --accumulation-steps 2
```

### Baseline 结果（2026-07-31）

使用 ImageNet 预训练的 `vit_small_patch16_224`、官方 `make` 划分和双卡 BF16
训练 30 轮。最佳 validation Top-1 为 93.85%；加载 `best.pth` 在官方 test 集
12,322 条车辆轨迹上评估，结果为：

| 指标 | 结果 |
|---|---:|
| Test loss | 0.651585 |
| Test Top-1 | 88.04% |
| Test Top-5 | 97.13% |

validation 与 test 的差距符合官方跨摄像头划分的难度：测试摄像头不出现在训练集。
复现实验使用：

```bash
torchrun --standalone --nproc_per_node=2 evaluate_boxcars.py \
  --checkpoint checkpoints/boxcars_make_baseline/best.pth \
  --task make --split test --batch-size 16
```
