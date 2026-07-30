"""
一体化评测脚本：精度 + 延迟 + 内存 全指标评测。

支持两种运行模式：
  1. 随机数据模式（默认）：测延迟 + 内存，无需数据集/checkpoint（本地演示用）
  2. 真实数据模式：需 --dataset-path + --checkpoint，测精度 + 延迟 + 内存（服务器用）

支持两个场景：
  --scene modelnet40  （场景1，40类分类）
  --scene sevd         （场景2，6类检测/分类）

用法（从 module_edge_perception/ 目录运行）：
    # 随机数据模式（本地演示）
    py -3.11 benchmarks/benchmark_full.py
    py -3.11 benchmarks/benchmark_full.py --model-name vit_tiny_patch16_224

    # 真实数据模式（服务器，场景1）
    py -3.11 benchmarks/benchmark_full.py --scene modelnet40 \
        --dataset-path ../data/modelnet40v2png_ori4 --checkpoint models/mv_vit_best.pth

    # 真实数据模式（服务器，场景2）
    py -3.11 benchmarks/benchmark_full.py --scene sevd \
        --dataset-path ../data/SEVD --checkpoint models/mv_vit_sevd.pth

输出：
    benchmarks/results/full_benchmark_<model>_<scene>.csv
    benchmarks/results/full_benchmark_<model>_<scene>.png
"""
import argparse
import csv
import gc
import math
import os
import time
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


# ============================================================
# 计时 / 内存 测量工具函数
# ============================================================

def measure_latency(model, fn, warmup=8, repeat=30):
    """测延迟，返回中位数/P95/均值（ms）。"""
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn()
        times = []
        for _ in range(repeat):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = fn()
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = fn()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "mean_ms": sum(times) / len(times),
    }


def measure_memory(model, fn, weight_mb):
    """测内存峰值，返回总占用 GB。"""
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(5):
                _ = fn()
        peak_bytes = torch.cuda.max_memory_allocated()
    else:
        if not HAS_PSUTIL:
            return {"total_gb": 0.0, "inference_mb": 0.0}
        gc.collect()
        proc = psutil.Process(os.getpid())
        rss_before = proc.memory_info().rss
        peak = rss_before
        with torch.no_grad():
            for _ in range(5):
                _ = fn()
                rss = proc.memory_info().rss
                if rss > peak:
                    peak = rss
        peak_bytes = peak - rss_before

    total_bytes = peak_bytes + int(weight_mb * 1024 ** 2)
    return {
        "total_gb": total_bytes / (1024 ** 3),
        "inference_mb": peak_bytes / (1024 ** 2),
    }


# ============================================================
# 数据加载
# ============================================================

def build_dataloader(args):
    """根据场景构建 DataLoader。返回 (loader, num_classes) 或 (None, num_classes)。"""
    if not args.dataset_path:
        return None, args.num_classes

    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.scene == "modelnet40":
        # 场景1：ModelNet40 四视角分类
        try:
            from drift_dataset import MultiViewDataset
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
            from drift_dataset import MultiViewDataset
        dataset = MultiViewDataset(
            root_dir=args.dataset_path, split="test", transform=transform
        )
        num_classes = 40
        collate = None
    elif args.scene == "sevd":
        # 场景2：SEVD 四路交通摄像头
        from sevd_dataset import SEVD_CLASSES, SEVDMultiView
        dataset = SEVDMultiView(
            root_dir=args.dataset_path, split="test", transform=transform,
            scenes=tuple(args.scenes) if args.scenes else None,
        )
        num_classes = len(SEVD_CLASSES)
        collate = dataset.collate_fn
    else:
        raise ValueError(f"未知场景: {args.scene}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )
    return loader, num_classes


def get_batch(loader, device, num_views):
    """从 loader 取一个 batch，返回 images [B, V, 3, 224, 224]。"""
    batch = next(iter(loader))
    if isinstance(batch, (list, tuple)):
        images = batch[0]  # (views, targets, metadata) 或 (images, labels)
    else:
        images = batch
    if images.dim() == 5 and images.shape[1] != num_views:
        images = images.permute(0, 1, 2, 3, 4)  # 确保是 [B, V, C, H, W]
    return images.to(device)


# ============================================================
# 精度测量
# ============================================================

def measure_accuracy(model, loader, device, keep_ratio, num_classes, max_batches=None):
    """测分类精度（Top-1）。keep_ratio=1.0 走原 forward，否则走硬剪枝。"""
    if loader is None:
        return None

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                labels = batch[1] if len(batch) > 1 else None
            else:
                images = batch
                labels = None

            # SEVD 的 batch 是 (views, targets, metadata)，无分类标签
            if labels is None or not torch.is_tensor(labels):
                # 无分类标签，跳过精度测量
                return None

            images = images.to(device)
            labels = labels.to(device)

            if keep_ratio >= 1.0:
                out = model(images)
            else:
                out = model.forward_hard_prune(images, keep_ratio=keep_ratio,
                                               token_score_mode="importance")
            pred = out.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return correct / max(total, 1) if total > 0 else None


# ============================================================
# 主流程
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="一体化评测：精度+延迟+内存")
    p.add_argument("--model-name", default="vit_small_patch16_224",
                   choices=["vit_small_patch16_224", "vit_tiny_patch16_224"])
    p.add_argument("--scene", default="modelnet40", choices=["modelnet40", "sevd"])
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--keep-ratios", default="1.0,0.4,0.2,0.1")
    p.add_argument("--dataset-path", default=None, help="数据集路径（不填则用随机数据）")
    p.add_argument("--checkpoint", default=None, help="模型权重路径（不填则随机权重）")
    p.add_argument("--scenes", nargs="*", default=None, help="SEVD 场景列表")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-accuracy-batches", type=int, default=None,
                   help="精度测量最大 batch 数（不填则全量）")
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keep_ratios = [float(x) for x in args.keep_ratios.split(",")]

    print("=" * 70)
    print("一体化评测：精度 + 延迟 + 内存")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"模型: {args.model_name}")
    print(f"场景: {args.scene}")
    print(f"batch: {args.batch_size}  warmup: {args.warmup}  repeat: {args.repeat}")
    print(f"keep_ratios: {keep_ratios}")
    print(f"数据集: {args.dataset_path or '随机数据（无精度）'}")
    print(f"权重: {args.checkpoint or '随机权重'}")
    print()

    # 1. 构建数据集（可选）
    loader, num_classes = build_dataloader(args)
    if loader is not None:
        print(f"数据集加载完成: {len(loader.dataset)} 样本, {num_classes} 类")
    else:
        print("无数据集，仅测延迟+内存（精度标 N/A）")

    # 2. 构建模型
    print(f"\n构建模型 {args.model_name} ...")
    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=args.num_views,
        num_classes=num_classes,
        pretrained=False,
    ).to(device).eval()

    # 加载 checkpoint（如有）
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)
        print(f"权重已加载: {args.checkpoint}")
    else:
        print("使用随机权重")

    n_params = sum(p.numel() for p in model.parameters())
    weight_mb = n_params * 4 / (1024 ** 2)
    print(f"参数量: {n_params:,} ({n_params/1e6:.1f}M)  权重: {weight_mb:.1f} MB")

    # 3. 准备输入
    if loader is not None:
        images = get_batch(loader, device, args.num_views)
    else:
        images = torch.randn(args.batch_size, args.num_views, 3, 224, 224, device=device)
    print(f"输入形状: {tuple(images.shape)}")

    # 4. 评测各 keep_ratio
    rows = []
    base_seq_len = 1 + args.num_views * 196
    base_median = None

    for kr in keep_ratios:
        keep_count = max(1, int(math.ceil(kr * 196)))
        seq_len = 1 + args.num_views * keep_count if kr < 1.0 else base_seq_len
        mode = "baseline" if kr >= 1.0 else "hard_prune"

        print(f"\n[{mode} keep={kr}] seq_len={seq_len}")

        # 延迟
        if kr >= 1.0:
            fn = lambda: model(images)
        else:
            fn = lambda: model.forward_hard_prune(images, keep_ratio=kr,
                                                  token_score_mode="importance")
        lat = measure_latency(model, fn, args.warmup, args.repeat)
        print(f"  延迟: median={lat['median_ms']:.1f}ms  p95={lat['p95_ms']:.1f}ms")

        if kr >= 1.0:
            base_median = lat["median_ms"]
            ttft_drop = 0.0
        else:
            ttft_drop = (base_median - lat["median_ms"]) / base_median * 100
            print(f"  TTFT降幅: {ttft_drop:.1f}%{' ✅' if ttft_drop >= 75 else ''}")

        # 内存
        mem = measure_memory(model, fn, weight_mb)
        mem_ok = mem["total_gb"] <= 1.5
        print(f"  内存: 总占用={mem['total_gb']:.3f}GB  推理增量={mem['inference_mb']:.1f}MB"
              f"  {'✅' if mem_ok else '❌'}")

        # 精度
        acc = measure_accuracy(model, loader, device, kr, num_classes,
                               args.max_accuracy_batches)
        if acc is not None:
            print(f"  精度: Top-1={acc*100:.2f}%")
        else:
            print(f"  精度: N/A（无分类标签或无数据集）")

        rows.append({
            "mode": mode,
            "keep_ratio": kr,
            "seq_len": seq_len,
            "median_ms": round(lat["median_ms"], 2),
            "p95_ms": round(lat["p95_ms"], 2),
            "ttft_drop_pct": round(ttft_drop, 1),
            "memory_gb": round(mem["total_gb"], 4),
            "memory_inference_mb": round(mem["inference_mb"], 1),
            "memory_ok": mem_ok,
            "accuracy": round(acc * 100, 2) if acc is not None else "N/A",
            "params_m": round(n_params / 1e6, 1),
        })

    # 5. 写 CSV
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model_name.replace('_', '')}_{args.scene}"
    csv_path = out_dir / f"full_benchmark_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "keep_ratio", "seq_len", "median_ms", "p95_ms",
            "ttft_drop_pct", "memory_gb", "memory_inference_mb", "memory_ok",
            "accuracy", "params_m"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV 已保存: {csv_path}")

    # 6. 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 延迟图
        ax = axes[0]
        xs = [r["keep_ratio"] for r in rows]
        ys = [r["median_ms"] for r in rows]
        ax.plot(xs, ys, "bo-")
        if base_median:
            ax.axhline(base_median, color="gray", linestyle=":", label="baseline")
            target = base_median * 0.25
            ax.axhline(target, color="green", linestyle="--", label="75% target")
        ax.set_xlabel("keep_ratio")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title("Latency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 内存图
        ax = axes[1]
        ys = [r["memory_gb"] for r in rows]
        ax.plot(xs, ys, "rs-")
        ax.axhline(1.5, color="red", linestyle="--", label="1.5GB limit")
        ax.set_xlabel("keep_ratio")
        ax.set_ylabel("Memory (GB)")
        ax.set_title("Memory")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 精度图（如有）
        ax = axes[2]
        accs = [r["accuracy"] for r in rows if r["accuracy"] != "N/A"]
        if accs:
            xs_a = [r["keep_ratio"] for r in rows if r["accuracy"] != "N/A"]
            ax.plot(xs_a, accs, "g^-")
            ax.set_ylabel("Top-1 Accuracy (%)")
            ax.set_title("Accuracy")
        else:
            ax.text(0.5, 0.5, "N/A\n(no dataset)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_title("Accuracy (N/A)")
        ax.set_xlabel("keep_ratio")
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"{args.model_name} | {args.scene} | batch={args.batch_size} | {device}",
                     fontsize=12)
        fig.tight_layout()
        png_path = out_dir / f"full_benchmark_{tag}.png"
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        print(f"图表已保存: {png_path}")
    except Exception as e:
        print(f"画图跳过: {e}")

    # 7. 汇总
    print("\n" + "=" * 70)
    print(f"汇总：{args.model_name} @ {args.scene} (batch={args.batch_size}, {device})")
    print("=" * 70)
    print(f"{'mode':<12}{'keep':<8}{'seq':<8}{'延迟ms':<10}{'TTFT降':<10}"
          f"{'内存GB':<10}{'精度':<10}{'达标':<10}")
    for r in rows:
        ttft = f"{r['ttft_drop_pct']:.1f}%" if r["mode"] != "baseline" else "-"
        mem = f"{r['memory_gb']:.3f}"
        acc = f"{r['accuracy']}%" if r["accuracy"] != "N/A" else "N/A"
        ok_ttft = "✅" if r["ttft_drop_pct"] >= 75 else ""
        ok_mem = "✅" if r["memory_ok"] else ""
        print(f"{r['mode']:<12}{r['keep_ratio']:<8}{r['seq_len']:<8}"
              f"{r['median_ms']:<10.1f}{ttft:<10}{mem:<10}{acc:<10}"
              f"{ok_ttft}{ok_mem:>6}")


if __name__ == "__main__":
    main()
