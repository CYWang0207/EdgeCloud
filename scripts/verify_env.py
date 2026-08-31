import argparse
import time

import torch

from module_edge_perception.model import EarlyFusionMultiViewViT


parser = argparse.ArgumentParser(description="EdgeCloud 无数据四视图环境验证")
parser.add_argument(
    "--pretrained",
    action="store_true",
    help="通过 timm 下载并加载 ImageNet 预训练权重；默认随机初始化以支持离线验证",
)
parser.add_argument("--iterations", type=int, default=3, help="CPU 参考计时迭代次数")
args = parser.parse_args()

print("=" * 50)
print("环境验证")
print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
print("=" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

print("\n[1/4] 构建模型...")
model = EarlyFusionMultiViewViT(
    "vit_small_patch16_224", 4, 40, pretrained=args.pretrained
).to(device)
print(f"  参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

print("\n[2/4] 随机输入推理 (2,4,3,224,224)...")
images = torch.randn(2, 4, 3, 224, 224, device=device)
model.eval()
with torch.no_grad():
    out = model(images)
print(f"  输出: {list(out.shape)}, 预测: {out.argmax(1).tolist()}")

print("\n[3/4] Token剪枝 + 视角休眠...")
keep = torch.tensor([[0.7,0.3,0.8,0.0],[0.5,0.5,0.5,0.5]], device=device)
vmask = torch.tensor([[1,1,1,0],[1,1,1,1]], device=device)
with torch.no_grad():
    out2 = model(images, view_mask=vmask, keep_ratios=keep, token_score_mode="importance")
print(f"  协同输出: {list(out2.shape)}, 预测: {out2.argmax(1).tolist()}")

print("\n[4/4] CPU单帧计时...")
imgs = torch.randn(1, 4, 3, 224, 224, device=device)
with torch.no_grad():
    t0 = time.time()
    for _ in range(args.iterations):
        _ = model(imgs)
    t1 = time.time()
print(f"  平均: {(t1-t0)/args.iterations*1000:.0f} ms/帧 (CPU参考值)")

print("\n" + "=" * 50)
print("环境验证全部通过！代码链路完整可运行。")
print("=" * 50)
