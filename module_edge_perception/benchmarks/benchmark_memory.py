"""
内存测量脚本：测峰值推理内存，验证 ≤1.5GB 目标。

用法（从 module_edge_perception/ 目录运行）：
    py -3.11 benchmarks/benchmark_memory.py
    py -3.11 benchmarks/benchmark_memory.py --model-name vit_tiny_patch16_224
    py -3.11 benchmarks/benchmark_memory.py --batch-size 1 --keep-ratios 1.0,0.5,0.2,0.1

输出：
    benchmarks/results/memory_results.csv
"""
import argparse
import csv
import gc
import os
from pathlib import Path

import torch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import EarlyFusionMultiViewViT


def parse_args():
    p = argparse.ArgumentParser(description="推理内存 profiling：验证 ≤1.5GB")
    p.add_argument("--model-name", default="vit_small_patch16_224",
                   choices=["vit_small_patch16_224", "vit_tiny_patch16_224"])
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=1, help="推理 batch size（赛题'单次推理'口径）")
    p.add_argument("--keep-ratios", default="1.0,0.5,0.2,0.1", help="逗号分隔的保留率")
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    return p.parse_args()


def measure_gpu_memory(model, fn):
    """GPU: 用 torch.cuda.max_memory_allocated 测峰值。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = fn()
    peak_bytes = torch.cuda.max_memory_allocated()
    return peak_bytes


def measure_cpu_memory(model, fn):
    """CPU: 用 psutil 测进程 RSS 峰值（包含 PyTorch 张量内存，比 tracemalloc 准确）。"""
    if not HAS_PSUTIL:
        raise RuntimeError("CPU 内存测量需要 psutil：pip install psutil")
    gc.collect()
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    peak = rss_before
    with torch.no_grad():
        # 多次推理取峰值，捕捉中间张量分配
        for _ in range(5):
            _ = fn()
            rss = proc.memory_info().rss
            if rss > peak:
                peak = rss
    # 峰值增量 = 推理峰值 - 推理前基线
    return peak - rss_before


def get_model_weight_mb(model):
    """模型权重占用（参数量 × 4 bytes / 1024^2）。"""
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 4 / (1024 ** 2)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  模型: {args.model_name}  batch: {args.batch_size}")

    print(f"\n构建模型 {args.model_name} ...")
    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=args.num_views,
        num_classes=args.num_classes,
        pretrained=False,
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    weight_mb = get_model_weight_mb(model)
    print(f"参数量: {n_params:,} ({n_params/1e6:.1f}M)  权重占用: {weight_mb:.1f} MB")

    images = torch.randn(args.batch_size, args.num_views, 3, 224, 224, device=device)
    keep_ratios = [float(x) for x in args.keep_ratios.split(",")]
    rows = []

    # 1.5GB 阈值（bytes）
    THRESHOLD_GB = 1.5
    THRESHOLD_BYTES = THRESHOLD_GB * 1024 ** 3

    print(f"\n阈值: {THRESHOLD_GB} GB = {THRESHOLD_BYTES/1024**3:.2f} GB")

    for kr in keep_ratios:
        # baseline（无剪枝）
        if kr >= 1.0:
            fn = lambda: model(images)
            mode = "baseline"
        else:
            fn = lambda: model.forward_hard_prune(images, keep_ratio=kr, token_score_mode="importance")
            mode = "hard_prune"

        if device.type == "cuda":
            peak_bytes = measure_gpu_memory(model, fn)
        else:
            peak_bytes = measure_cpu_memory(model, fn)

        # 总占用 = 模型权重 + 推理峰值增量（赛题"单次推理内存"口径）
        weight_bytes = int(weight_mb * 1024 ** 2)
        total_bytes = peak_bytes + weight_bytes
        total_gb = total_bytes / (1024 ** 3)
        within = "✅" if total_bytes <= THRESHOLD_BYTES else "❌"
        print(f"[{mode} keep={kr}] 推理增量: {peak_bytes/1024**2:.1f} MB  "
              f"总占用(权重+推理): {total_gb:.3f} GB  {within}")

        rows.append({
            "mode": mode,
            "keep_ratio": kr,
            "batch_size": args.batch_size,
            "weight_mb": weight_mb,
            "inference_peak_mb": peak_bytes / (1024 ** 2),
            "total_gb": total_gb,
            "within_1_5gb": total_bytes <= THRESHOLD_BYTES,
        })

    # 写 CSV
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "memory_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "keep_ratio", "batch_size", "weight_mb",
            "inference_peak_mb", "total_gb", "within_1_5gb"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV 已保存: {csv_path}")

    # 汇总
    print("\n" + "=" * 60)
    print(f"汇总（{args.model_name}, batch={args.batch_size}, device={device}）")
    print(f"模型权重占用: {weight_mb:.1f} MB ({n_params/1e6:.1f}M 参数)")
    print("=" * 60)
    print(f"{'mode':<14}{'keep_ratio':<12}{'推理增量':>12}{'总占用':>12}{'达标':>8}")
    for r in rows:
        ok = "✅" if r["within_1_5gb"] else "❌"
        print(f"{r['mode']:<14}{r['keep_ratio']:<12}{r['inference_peak_mb']:>10.1f}MB"
              f"{r['total_gb']:>10.3f}GB{ok:>8}")


if __name__ == "__main__":
    main()
