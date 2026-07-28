import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

class ModelNet40MultiView(Dataset):
    def __init__(self, root_dir, split='train', transform=None, num_views=4):
        """
        多视角数据集加载器
        :param root_dir: 数据集的根目录，包含 40 个类别文件夹
        :param split: 'train' 或 'test'
        :param transform: torchvision 数据预处理管道
        :param num_views: 每个物体提取的视角数量 (默认 4)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_views = num_views

        # 1. 获取所有类别 (40个文件夹) 并按字母排序以确保一致性
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 存储样本的列表：[(实例的共同路径前缀, 标签), ...]
        self.samples = []

        # 2. 遍历文件夹，提取同一个物体的4个视角的公共前缀
        for cls_name in self.classes:
            split_dir = os.path.join(root_dir, cls_name, split)
            if not os.path.exists(split_dir):
                continue

            # 获取该类别下所有的图片路径
            all_imgs = glob.glob(os.path.join(split_dir, '*.png'))

            # 使用集合去重，提取实例的主名称 (例如把 airplane_0001_001.png 变成 airplane_0001)
            instance_prefixes = set()
            for img_path in all_imgs:
                # 找到最后一个 '_' 之前的部分作为该 3D 实例的唯一标识符
                prefix = img_path.rsplit('_', 1)[0]
                instance_prefixes.add(prefix)

            # 将该类别下的所有唯一实例加入总样本集
            for prefix in sorted(list(instance_prefixes)):
                self.samples.append((prefix, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix, label = self.samples[idx]

        views = []
        # 按顺序读取 1 到 num_views 的图片
        for v in range(1, self.num_views + 1):
            # 兼容两种常见的多视角命名格式
            img_path = f"{prefix}_{v:03d}.png"  # 尝试 _001.png 格式
            if not os.path.exists(img_path):
                img_path = f"{prefix}_{v}.png"  # 退回 _1.png 格式

            # 如果某张图片确实丢失，可以抛出异常或用全黑图片代替，这里选择严格读取
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"找不到视角图片: {img_path}")

            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            views.append(image)

        # 核心步骤：将4个 [C, H, W] 的张量沿着新的维度(视角维度)堆叠起来
        # 最终形状为 -> [num_views, C, H, W]
        views_tensor = torch.stack(views, dim=0)

        return views_tensor, label