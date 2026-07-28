import sys
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import ModelNet40MultiView
from prompt_tuning.prompt_model import PromptGenerator
from prompt_tuning.prompt_vit import PromptMultiViewViT
from prompt_tuning.prompt_dataset import CONDITION_REGISTRY  # 引入注册表


def test_specific_condition(vit_model, prompt_gen, test_loader, device, condition_id, condition_info):
    vit_model.eval()
    prompt_gen.eval()

    condition_name = condition_info["name"]
    transform_func = condition_info["transform"]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)

    correct, total = 0, 0
    print(f"\n 正在评估: [{condition_name}] (ID: {condition_id})")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            B, V, C, H, W = images.shape

            modified_images = []
            for b in range(B):
                batch_imgs = []
                for v in range(V):
                    img = images[b, v]
                    # 1. 动态调用该场景的破坏函数
                    img = transform_func(img)
                    # 2. 归一化
                    img = normalize(img)
                    batch_imgs.append(img)
                modified_images.append(torch.stack(batch_imgs, dim=0))
            images = torch.stack(modified_images, dim=0)

            condition_ids = torch.full((B,), condition_id, dtype=torch.long, device=device)
            outputs = vit_model(images, prompt_tokens=prompt_gen(condition_ids))

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f" [{condition_name}] 推理完成！最终准确率: {acc:.2f}% ({correct}/{total})")
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "./data/modelnet40v2png_ori4"
    pretrained_vit_path = "./checkpoints/mv_vit_epoch_30.pth"
    prompt_weight_path = "./checkpoints/prompt_generator_best.pth"

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 不在这里 Normalize
    ])

    test_dataset = ModelNet40MultiView(root_dir=dataset_path, split='test', transform=test_transform, num_views=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    num_conditions = len(CONDITION_REGISTRY)
    vit_model = PromptMultiViewViT(model_name='vit_small_patch16_224', num_views=4, pretrained=False).to(device)
    prompt_gen = PromptGenerator(vit_embed_dim=384, num_prompt_tokens=4, num_conditions=num_conditions).to(device)

    vit_model.load_state_dict(torch.load(pretrained_vit_path, map_location=device))

    if os.path.exists(prompt_weight_path):
        checkpoint = torch.load(prompt_weight_path, map_location=device)
        prompt_gen.load_state_dict(checkpoint['prompt_gen'])
        vit_model.norm.load_state_dict(checkpoint['vit_norm'])
        vit_model.head.load_state_dict(checkpoint['vit_head'])
        print(" 成功加载全套微调权重")

    print("\n" + "=" * 50)
    # 🌟 自动化测试所有注册的场景！
    for cid, info in CONDITION_REGISTRY.items():
        test_specific_condition(vit_model, prompt_gen, test_loader, device, cid, info)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()