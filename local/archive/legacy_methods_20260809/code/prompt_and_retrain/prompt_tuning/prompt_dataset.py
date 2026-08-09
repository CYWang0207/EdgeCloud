import torch
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

# ==========================================
# 🌟 核心扩展区：场景注册表 (Registry)
# 未来你要加任何新 Prompt 场景，只需要在这里加一行！
# ==========================================
CONDITION_REGISTRY = {
    0: {
        "name": "正常光照",
        "transform": lambda img: img  # 正常情况什么都不做
    },
    1: {
        "name": "光照变亮",
        "transform": lambda img: TF.adjust_brightness(img, 1.5)
    },
    2: {
        "name": "光照变暗",
        "transform": lambda img: TF.adjust_brightness(img, 0.5)
    },
    # 🔮 未来的扩展例子 (你可以随时取消注释并实现它们)：
    # 3: {
    #     "name": "高斯模糊",
    #     "transform": lambda img: TF.gaussian_blur(img, kernel_size=5)
    # },
    # 4: {
    #     "name": "随机噪声",
    #     "transform": lambda img: img + torch.randn_like(img) * 0.1
    # }
}


class ConditionalDatasetWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        # 获取所有已注册的场景 ID (例如 [0, 1, 2])
        self.condition_ids = list(CONDITION_REGISTRY.keys())
        # ⚠️ 修复点：将归一化剥离到这里，必须在所有物理破坏之后执行
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        images, label = self.base_dataset[idx]

        # 动态从注册表中随机抽取一个场景
        condition_id = random.choice(self.condition_ids)
        # 获取该场景对应的处理函数
        transform_func = CONDITION_REGISTRY[condition_id]["transform"]

        modified_images = []
        for img in images:
            # 1. 应用物理破坏 (如变亮、变暗、模糊等)
            img = transform_func(img)
            # 2. ⚠️ 必须在最后一步进行数学归一化！
            img = self.normalize(img)
            modified_images.append(img)

        images = torch.stack(modified_images, dim=0)
        return images, label, condition_id