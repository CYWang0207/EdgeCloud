# data · 数据集目录（内容不进 Git）

**真实数据集不进 Git 仓库**（体积过大），本目录在仓库中只保留本说明文件。

## 使用方式

1. 按下文链接下载所需数据集。
2. 解压到本目录下，例如：

```
data/
├── README.md        # 本文件（唯一进 Git 的文件）
├── modelnet40v2png_ori4/  # 场景 1：ModelNet40 四视角分类
└── SEVD/                    # 场景 2：四路固定交通摄像头感知
```

`.gitignore` 已配置忽略本目录下除 README 外的所有内容，正常 `git add .` 不会误传数据。

## 场景 2：SEVD Fixed Perception RGB

SEVD 是公开的 CARLA 合成自动驾驶多模态数据集。项目第二场景只使用其中四路
固定摄像头的同步 RGB 与 2D 标注，适配多视角边缘感知、视角选择、Token 剪枝
和场景漂移实验。原始来源：

- 项目页：https://eventbasedvision.github.io/SEVD/
- 论文：https://arxiv.org/abs/2404.10540
- 官方代码：https://github.com/eventbasedvision/SEVD
- 本项目使用的数据下载目录：https://www.dropbox.com/scl/fo/mhn4ykfhlk3as46sg8mhs/AB_LqoZeea2yjESIoa3B2Ps?rlkey=fq4t643tc6q6p0b85aj2p7a3o&dl=0

可从官方下载链接获取。当前服务器已整理到：

```text
/root/autodl-tmp/SEVD/
├── scene_002/organized/
├── scene_003/organized/
├── scene_030/organized/
└── scene_032/organized/
```

每个 `organized/` 均包含 `metadata.json` 和 `camera_01` 至 `camera_04`；每路
相机包含 `images/*.png` 与 `annotations/coco.json`。同名 PNG 是同一个同步
时刻。整理后共有 5704 个四视角时刻、22816 张 RGB 图像，约 36 GB；类别为
`car`、`truck`、`van`、`pedestrian`、`motorcycle`、`bicycle`。

如需放在仓库目录，保持同样结构复制或软链接到 `data/SEVD`。不要提交真实数据。

DataLoader 位于 `module_edge_perception/sevd_dataset.py`：

```python
from module_edge_perception.sevd_dataset import SEVDMultiView

dataset = SEVDMultiView("data/SEVD", split="train", transform=transform)
views, targets, metadata = dataset[0]
# views.shape == [4, 3, H, W]
```

`train/val/test` 在每个场景内按同步时刻顺序切分为 70%/15%/15%，不可按单路
图片随机切分。检测框按相机保留在 `targets` 中，类别标签沿用原 COCO 文件的
`category_id` 1–6（0 保留为背景）。当前 `transform` 只变换图像；若后续训练
检测头并使用会改变几何尺寸的增强，需要同步变换 `targets[view]["boxes"]`。

在服务器上验证真实数据和现有 MV-ViT 前向链路：

```bash
cd /path/to/EdgeCloud/module_edge_perception
/root/miniconda3/bin/python test_sevd_inference.py \
  --dataset-path /root/autodl-tmp/SEVD --scene scene_030
```

该命令不训练、不下载权重；6 类分类头为随机初始化，只用于确认 DataLoader 的
图像张量能通过现有模型。原始 COCO 目标仍然保留，后续若做目标检测应接检测头，
不能把该随机分类输出当作检测精度结论。

### 已验证结论（2026-07-30）

已在服务器的真实 `scene_030` 数据上执行上述命令：测试段识别到 271 个同步
样本，输入批张量为 `[1, 4, 3, 224, 224]`，MV-ViT 输出为 `[1, 6]`，首个样本
四路分别读取到 8、12、6、5 个标注目标。由此确认第二场景数据目录、四路同步
组合、COCO 标注索引、DataLoader 拼批和现有模型前向数据流均可用；这只是连通性
验证，不代表模型已经在 SEVD 上训练或具备有效精度。
