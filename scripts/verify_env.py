import sys, os, time
sys.path.insert(0, ".")
import torch
from model import EarlyFusionMultiViewViT
from prompt_tuning.prompt_model import PromptGenerator

print("=" * 50)
print("环境验证")
print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
print("=" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

print("\n[1/4] 构建模型...")
model = EarlyFusionMultiViewViT("vit_small_patch16_224", 4, 40, pretrained=True).to(device)
print(f"  参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

print("\n[2/4] 随机输入推理 (2,4,3,224,224)...")
images = torch.randn(2, 4, 3, 224, 224, device=device)
model.eval()
with torch.no_grad():
    out = model(images)
print(f"  输出: {list(out.shape)}, 预测: {out.argmax(1).tolist()}")

print("\n[3/4] Token剪枝 + Prompt注入 + 视角休眠...")
keep = torch.tensor([[0.7,0.3,0.8,0.0],[0.5,0.5,0.5,0.5]], device=device)
vmask = torch.tensor([[1,1,1,0],[1,1,1,1]], device=device)
pg = PromptGenerator(384, 4, 7).to(device)
pt = pg(torch.tensor([0,4], device=device))
with torch.no_grad():
    out2 = model(images, view_mask=vmask, keep_ratios=keep, token_score_mode="importance", prompt_tokens=pt)
print(f"  协同输出: {list(out2.shape)}, 预测: {out2.argmax(1).tolist()}")

print("\n[4/4] CPU单帧计时...")
imgs = torch.randn(1, 4, 3, 224, 224, device=device)
with torch.no_grad():
    t0 = time.time()
    for _ in range(10): _ = model(imgs)
    t1 = time.time()
print(f"  平均: {(t1-t0)/10*1000:.0f} ms/帧 (CPU参考值)")

print("\n" + "=" * 50)
print("环境验证全部通过！代码链路完整可运行。")
print("=" * 50)
