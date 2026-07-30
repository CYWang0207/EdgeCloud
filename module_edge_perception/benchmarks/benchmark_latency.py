"""
延迟测量脚本：对比无剪枝 vs 硬剪枝的 TTFT / 单帧延迟。

赛题口径：硬剪枝相对无剪枝基线的 TTFT 降幅（目标 ≥75%）。
软剪枝为可选参考（--with-soft-prune），用于佐证"软剪枝不降延迟"。

用法（从 module_edge_perception/ 目录运行）：
    py -3.11 benchmarks/benchmark_latency.py
    py -3.11 benchmarks/benchmark_latency.py --model-name vit_tiny_patch16_224
    py -3.11 benchmarks/benchmark_latency.py --batch-size 1 --repeat 100
    py -3.11 benchmarks/benchmark_latency.py --with-soft-prune   # 附带软剪枝对照

输出：
    benchmarks/results/latency_results.csv
    benchmarks/results/latency_comparison.png
"""
import argparse
import csv
import time
from pathlib import Path

import torch

# import 路径：本脚本位于 module_edge_perception/benchmarks/，model.py 在上级目录
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import EarlyFusionMultiViewViT


def parse_args():
    p = argparse.ArgumentParser(description="TTFT / 延迟测量：无剪枝 vs 硬剪枝")
    p.add_argument("--model-name", default="vit_small_patch16_224",
                   choices=["vit_small_patch16_224", "vit_tiny_patch16_224"])
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2, help="测量用 batch size")
    p.add_argument("--warmup", type=int, default=10, help="warmup 次数")
    p.add_argument("--repeat", type=int, default=50, help="测量次数（取中位数）")
    p.add_argument("--keep-ratios", default="1.0,0.8,0.6,0.4,0.2,0.1",
                   help="逗号分隔的保留率列表")
    p.add_argument("--with-soft-prune", action="store_true",
                   help="附带测软剪枝作为参考对照（默认不测）")
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    return p.parse_args()


def measure(model, fn, warmup, repeat):
    """通用计时函数，返回中位数/P95/均值（ms）。"""
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn(model)
        times = []
        for _ in range(repeat):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = fn(model)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = fn(model)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "mean_ms": sum(times) / len(times),
    }


def run_baseline(model, images):
    """无剪枝基线：直接走原 forward。"""
    def fn(_):
        return model(images)
    return fn


def run_soft_prune(model, images, keep_ratio, num_views):
    """软剪枝（参考）：用 mask token 替换，序列长度不变。"""
    keep = torch.full((images.shape[0], num_views), keep_ratio,
                      dtype=torch.float32, device=images.device)
    def fn(_):
        return model(images, keep_ratios=keep, token_score_mode="importance")
    return fn


def run_hard_prune(model, images, keep_ratio):
    """硬剪枝：真正删除 token，序列长度缩短。"""
    def fn(_):
        return model.forward_hard_prune(images, keep_ratio=keep_ratio,
                                        token_score_mode="importance")
    return fn


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  模型: {args.model_name}  batch: {args.batch_size}")
    print(f"warmup={args.warmup}  repeat={args.repeat}")
    print(f"对比口径：无剪枝 vs 硬剪枝"
          + ("（附带软剪枝参考）" if args.with_soft_prune else ""))

    # 构建模型（随机权重，benchmark 用途）
    print(f"\n构建模型 {args.model_name} ...")
    model = EarlyFusionMultiViewViT(
        model_name=args.model_name,
        num_views=args.num_views,
        num_classes=args.num_classes,
        pretrained=False,
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,} ({n_params/1e6:.1f}M)")

    # 随机输入
    images = torch.randn(args.batch_size, args.num_views, 3, 224, 224, device=device)

    keep_ratios = [float(x) for x in args.keep_ratios.split(",")]
    rows = []
    base_seq_len = 1 + args.num_views * 196

    # 1. 基线（无剪枝）
    print(f"\n[基线] 无剪枝  seq_len={base_seq_len} ...")
    m = measure(model, run_baseline(model, images), args.warmup, args.repeat)
    print(f"  median={m['median_ms']:.1f}ms  p95={m['p95_ms']:.1f}ms")
    rows.append({
        "mode": "baseline", "keep_ratio": 1.0, "seq_len": base_seq_len,
        **m
    })
    base_median = m["median_ms"]

    # 2. 各 keep_ratio 下硬剪枝（主对比）
    for kr in keep_ratios:
        if kr >= 1.0:
            continue
        keep_count = max(1, int(torch.ceil(torch.tensor(kr * 196)).item()))
        seq_len = 1 + args.num_views * keep_count

        print(f"\n[硬剪枝 keep={kr}] seq_len={seq_len} (vs 基线 {base_seq_len})")
        m_hard = measure(model, run_hard_prune(model, images, kr),
                         args.warmup, args.repeat)
        drop_pct = (base_median - m_hard["median_ms"]) / base_median * 100
        flag = " ✅" if drop_pct >= 75 else ""
        print(f"  median={m_hard['median_ms']:.1f}ms  "
              f"TTFT降幅={drop_pct:.1f}%{flag}")
        rows.append({
            "mode": "hard_prune", "keep_ratio": kr, "seq_len": seq_len,
            **m_hard
        })

        # 软剪枝（可选参考）
        if args.with_soft_prune:
            m_soft = measure(model, run_soft_prune(model, images, kr, args.num_views),
                             args.warmup, args.repeat)
            soft_chg = (m_soft["median_ms"] - base_median) / base_median * 100
            print(f"  [参考] 软剪枝: median={m_soft['median_ms']:.1f}ms  变化={soft_chg:+.1f}%")
            rows.append({
                "mode": "soft_prune", "keep_ratio": kr,
                "seq_len": base_seq_len,  # 软剪枝序列长度不变
                **m_soft
            })

    # 写 CSV
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "latency_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "keep_ratio", "seq_len", "median_ms", "p95_ms", "mean_ms"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV 已保存: {csv_path}")

    # 画图：无剪枝 vs 硬剪枝
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        # 基线水平线
        base_row = next(r for r in rows if r["mode"] == "baseline")
        ax.axhline(base_row["median_ms"], color="gray", linestyle=":",
                   label=f"baseline (no prune) = {base_row['median_ms']:.0f}ms")
        # 硬剪枝曲线
        xs = [r["keep_ratio"] for r in rows if r["mode"] == "hard_prune"]
        ys = [r["median_ms"] for r in rows if r["mode"] == "hard_prune"]
        ax.plot(xs, ys, "bo-", label="hard_prune")
        # 软剪枝（可选）
        if args.with_soft_prune:
            xs_s = [r["keep_ratio"] for r in rows if r["mode"] == "soft_prune"]
            ys_s = [r["median_ms"] for r in rows if r["mode"] == "soft_prune"]
            ax.plot(xs_s, ys_s, "rs--", label="soft_prune (ref)")
        # 75% 目标线
        target = base_row["median_ms"] * (1 - 0.75)
        ax.axhline(target, color="green", linestyle="--",
                   label=f"75% reduction target = {target:.0f}ms")
        ax.set_xlabel("keep_ratio")
        ax.set_ylabel("TTFT / median latency (ms)")
        ax.set_title(f"{args.model_name}  batch={args.batch_size}  device={device}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        png_path = out_dir / "latency_comparison.png"
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        print(f"图表已保存: {png_path}")
    except Exception as e:
        print(f"画图跳过: {e}")

    # 汇总：无剪枝 vs 硬剪枝
    print("\n" + "=" * 60)
    print(f"汇总：无剪枝 vs 硬剪枝（基线 median = {base_median:.1f}ms）")
    print("=" * 60)
    print(f"{'keep_ratio':<12}{'seq_len':<10}{'硬剪枝ms':<12}{'TTFT降幅':<12}{'达标':<8}")
    for r in rows:
        if r["mode"] != "hard_prune":
            continue
        drop = (base_median - r["median_ms"]) / base_median * 100
        ok = "✅" if drop >= 75 else ""
        print(f"{r['keep_ratio']:<12}{r['seq_len']:<10}{r['median_ms']:<12.1f}"
              f"{drop:>10.1f}%{ok:>8}")


if __name__ == "__main__":
    main()
