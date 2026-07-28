import torch
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 导入我们之前写好的数据集和模型
from dataset import ModelNet40MultiView
from model import EarlyFusionMultiViewViT


def evaluate_single_checkpoint(model, test_loader, weight_path, device):
    """
    加载指定权重并在测试集上进行评估
    """
    if not os.path.exists(weight_path):
        print(f"❌ 找不到权重文件: {weight_path}")
        return

    # 加载权重；strict=False 用于兼容旧权重或剪枝感知训练新增的 mask token
    missing, unexpected = model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    if missing or unexpected:
        print(f"权重兼容加载: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()  # 切换到推理模式（关闭 Dropout 等）

    correct = 0
    total = 0

    print(f"正在评估权重: {os.path.basename(weight_path)} ...")

    # 关闭梯度计算，节省显存并加速
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向推理
            outputs = model(images)

            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"测试集准确率: {accuracy:.2f}% ({correct}/{total})\n")
    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的设备: {device}")

    # ================== 配置区 ==================
    dataset_path = "./data/modelnet40v2png_ori4"
    batch_size = 32  # 测试时不计算梯度，Batch Size 可以比训练时大一倍
    num_classes = 40
    num_views = 4

    # 你想测试的具体权重文件路径
    # 你可以把它改成比如 "./checkpoints/mv_vit_token_epoch_30.pth"
    weight_to_test = "./checkpoints/mv_vit_token_epoch_30.pth"
    # ============================================

    # 1. 定义纯净的测试数据预处理（没有任何数据增强）
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 加载测试集 (split='test')
    try:
        test_dataset = ModelNet40MultiView(
            root_dir=dataset_path,
            split='test',  # 注意这里改成了 test
            transform=test_transform,
            num_views=num_views
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # 测试集不需要打乱
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        print(f"成功加载测试集，共有 {len(test_dataset)} 个样本。")
    except Exception as e:
        print(f"数据加载失败，请检查路径。错误信息: {e}")
        return

    # 3. 实例化模型
    # 注意：这里 pretrained=False，因为我们不需要从网上下载预训练权重了，
    # 我们接下来会直接用自己训练好的 .pth 文件覆盖它。
    model = EarlyFusionMultiViewViT(
        model_name='vit_small_patch16_224',
        num_views=num_views,
        num_classes=num_classes,
        pretrained=False
    ).to(device)

    # 4. 执行测试
    # 模式一：测试单个指定的权重文件
    evaluate_single_checkpoint(model, test_loader, weight_to_test, device)

    # ==========================================================
    # 模式二（可选）：如果你想一次性测试 checkpoints 文件夹下的所有权重
    # 并找出准确率最高的一个，你可以取消下面这段代码的注释：
    # ==========================================================
    """
    print("="*40)
    print("开始批量测试所有保存的权重...")
    checkpoint_dir = "./checkpoints"
    best_acc = 0.0
    best_weight = ""

    # 遍历所有 .pth 文件
    for file in sorted(os.listdir(checkpoint_dir)):
        if file.endswith(".pth"):
            full_path = os.path.join(checkpoint_dir, file)
            acc = evaluate_single_checkpoint(model, test_loader, full_path, device)
            if acc > best_acc:
                best_acc = acc
                best_weight = file

    print(f"🏆 测试完毕！最佳权重是 {best_weight}，最高准确率为: {best_acc:.2f}%")
    """


if __name__ == "__main__":
    main()
