import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import ModelNet40MultiView
from prompt_tuning.prompt_model import PromptGenerator
from prompt_tuning.prompt_vit import PromptMultiViewViT
from prompt_tuning.prompt_dataset import ConditionalDatasetWrapper, CONDITION_REGISTRY


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "./data/modelnet40v2png_ori4"
    pretrained_vit_path = "./checkpoints/mv_vit_epoch_30.pth"
    save_dir = "./checkpoints"
    batch_size = 16
    epochs = 5
    num_views = 4
    vit_embed_dim = 384

    # 🌟 动态获取当前注册了多少个 Prompt 场景
    num_conditions = len(CONDITION_REGISTRY)
    print(f" 检测到 {num_conditions} 种 Prompt 场景，正在初始化训练管线...")

    # 去掉 Normalize，交由 Wrapper 在破坏图像后处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    base_train_dataset = ModelNet40MultiView(root_dir=dataset_path, split='train', transform=train_transform,
                                             num_views=num_views)
    train_dataset = ConditionalDatasetWrapper(base_train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    vit_model = PromptMultiViewViT(model_name='vit_small_patch16_224', num_views=num_views, num_classes=40,
                                   pretrained=False).to(device)
    vit_model.load_state_dict(torch.load(pretrained_vit_path, map_location=device))

    # ⚠️ 修复精度下降：冻结主干，但必须解冻最后的 Norm 和 Head 以适应新的 Prompt
    for param in vit_model.parameters():
        param.requires_grad = False
    for param in vit_model.norm.parameters():
        param.requires_grad = True
    for param in vit_model.head.parameters():
        param.requires_grad = True

    vit_model.eval()
    vit_model.norm.train()
    vit_model.head.train()

    prompt_gen = PromptGenerator(vit_embed_dim=vit_embed_dim, num_prompt_tokens=4, num_conditions=num_conditions).to(
        device)
    prompt_gen.train()

    # 将解冻的模块和 Prompt 一起交给优化器
    optimizer = torch.optim.AdamW([
        {'params': prompt_gen.parameters()},
        {'params': vit_model.norm.parameters(), 'lr': 1e-4},
        {'params': vit_model.head.parameters(), 'lr': 1e-4}
    ], lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels, condition_ids in pbar:
            images, labels, condition_ids = images.to(device), labels.to(device), condition_ids.to(device)
            optimizer.zero_grad()
            outputs = vit_model(images, prompt_tokens=prompt_gen(condition_ids))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{acc:.2f}%"})

    # ⚠️ 保存时，将更新后的 head 和 norm 一起打包
    save_path = os.path.join(save_dir, "prompt_generator_best.pth")
    torch.save({
        'prompt_gen': prompt_gen.state_dict(),
        'vit_norm': vit_model.norm.state_dict(),
        'vit_head': vit_model.head.state_dict()
    }, save_path)
    print(f"\n"
          f" Prompt 微调完成！权重已保存至: {save_path}")


if __name__ == "__main__":
    main()