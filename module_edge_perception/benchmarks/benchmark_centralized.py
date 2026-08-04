"""
集中式基准：对比三种推理架构的端到端延迟与带宽开销。

三种架构：
  1. centralized  集中式：边缘把全部视角原图上云，云端完整模型推理
                  延迟 = 上传传输 + 云端推理 + 下传结果
                  带宽 = 4 视角 × 原图大小
  2. edge_local   边缘本地：边缘硬剪枝推理，不上云
                  延迟 = 边缘推理
                  带宽 = 0
  3. edge_cloud   端边协同：边缘硬剪枝推理，仅在置信度不足时上云求助
                  延迟 = 边缘推理 + 按需(p%)×(上传+云端推理)
                  带宽 = p% × 4 视角 × 原图大小

用途：作为"端边云协同系统"的对照基准，证明协同架构在延迟/带宽上的优势。
赛题要求端到端延迟 ≤0.2s，这里量化各架构的延迟构成。

用法（从 module_edge_perception/ 目录运行）：
    py -3.11 benchmarks/benchmark_centralized.py
    py -3.11 benchmarks/benchmark_centralized.py --bandwidth-mbps 10 --edge-cloud-ratio 0.2
    py -3.11 benchmarks/benchmark_centralized.py --adapter-r 32

输出：
    benchmarks/results/centralized_benchmark.csv
    benchmarks/results/centralized_benchmark.png
"""
import argparse
import csv
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import EarlyFusionMultiViewViT


def measure(fn, warmup=5, repeat=20):
    """测延迟，返回中位数（ms）。"""
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn()
        times = []
        for _ in range(repeat):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = fn()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            else:
                t0 = time.perf_counter()
                _ = fn()
                times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def parse_args():
    p = argparse.ArgumentParser(description="集中式基准：三种推理架构对比")
    p.add_argument("--model-name", default="vit_small_patch16_224",
                   choices=["vit_small_patch16_224", "vit_tiny_patch16_224"])
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--keep-ratio", type=float, default=0.2,
                   help="边缘硬剪枝保留率（默认 0.2）")
    # 带宽与传输建模
    p.add_argument("--bandwidth-mbps", type=float, default=20.0,
                   help="边到云上行带宽（Mbps），默认 20")
    p.add_argument("--image-size-kb", type=float, default=150.0,
                   help="单视角原图编码大小（KB），默认 150（JPEG）")
    p.add_argument("--edge-cloud-ratio", type=float, default=0.2,
                   help="端边协同中需要上云求助的比例（默认 20%）")
    p.add_argument("--adapter-r", type=int, default=None,
                   help="挂载 AdaptFormer（可选）")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("集中式基准：三种推理架构对比")
    print("=" * 70)
    print(f"设备: {device}  模型: {args.model_name}")
    print(f"边缘硬剪枝 keep_ratio={args.keep_ratio}")
    print(f"带宽: {args.bandwidth_mbps} Mbps  单视角原图: {args.image_size_kb} KB")
    print(f"端边协同上云比例: {args.edge_cloud_ratio*100:.0f}%")
    print()

    # 构建模型（边缘 = 硬剪枝版；云端 = 完整无剪枝版）
    print("构建边缘模型（硬剪枝）...")
    edge_model = EarlyFusionMultiViewViT(
        model_name=args.model_name, num_views=args.num_views,
        num_classes=args.num_classes, pretrained=False,
    ).to(device).eval()

    if args.adapter_r is not None:
        from adaptformer import attach_adaptformer, set_adapter_enabled
        attach_adaptformer(edge_model, r=args.adapter_r)
        set_adapter_enabled(edge_model, True)

    print("构建云端模型（完整无剪枝）...")
    cloud_model = EarlyFusionMultiViewViT(
        model_name=args.model_name, num_views=args.num_views,
        num_classes=args.num_classes, pretrained=False,
    ).to(device).eval()
    if args.adapter_r is not None:
        from adaptformer import attach_adaptformer as _af
        _af(cloud_model, r=args.adapter_r)

    images = torch.randn(args.batch_size, args.num_views, 3, 224, 224, device=device)

    # ---- 测各组件延迟 ----
    print("\n测量各组件延迟...")
    edge_lat = measure(
        lambda: edge_model.forward_hard_prune(images, keep_ratio=args.keep_ratio,
                                              token_score_mode="importance"),
        args.warmup, args.repeat)
    cloud_lat = measure(lambda: cloud_model(images), args.warmup, args.repeat)
    print(f"  边缘硬剪枝推理: {edge_lat:.1f} ms")
    print(f"  云端完整推理:   {cloud_lat:.1f} ms")

    # ---- 传输延迟建模 ----
    # 上传数据量 = batch × num_views × image_size_kb
    upload_kb = args.batch_size * args.num_views * args.image_size_kb
    upload_bits = upload_kb * 8 * 1024  # bit
    upload_ms = upload_bits / (args.bandwidth_mbps * 1e6) * 1000
    # 下行结果（logits）很小，固定 1KB 量级
    download_ms = (1 * 8 * 1024) / (args.bandwidth_mbps * 1e6) * 1000
    print(f"  上传传输: {upload_ms:.1f} ms ({upload_kb:.0f} KB @ {args.bandwidth_mbps} Mbps)")
    print(f"  下传结果: {download_ms:.2f} ms")

    # ---- 三种架构端到端延迟 ----
    rows = []

    # 1. 集中式：上传 + 云端推理 + 下传
    centralized_total = upload_ms + cloud_lat + download_ms
    rows.append({
        "architecture": "centralized",
        "edge_ms": 0.0,
        "upload_ms": round(upload_ms, 1),
        "cloud_ms": round(cloud_lat, 1),
        "download_ms": round(download_ms, 2),
        "e2e_ms": round(centralized_total, 1),
        "bandwidth_kb": round(upload_kb, 0),
        "meets_0_2s": centralized_total <= 200,
    })

    # 2. 边缘本地：仅边缘推理
    edge_local_total = edge_lat
    rows.append({
        "architecture": "edge_local",
        "edge_ms": round(edge_lat, 1),
        "upload_ms": 0.0,
        "cloud_ms": 0.0,
        "download_ms": 0.0,
        "e2e_ms": round(edge_local_total, 1),
        "bandwidth_kb": 0.0,
        "meets_0_2s": edge_local_total <= 200,
    })

    # 3. 端边协同：边缘推理 + 按需(p%)×(上传+云端+下传)
    p = args.edge_cloud_ratio
    ec_total = edge_lat + p * (upload_ms + cloud_lat + download_ms)
    rows.append({
        "architecture": "edge_cloud",
        "edge_ms": round(edge_lat, 1),
        "upload_ms": round(p * upload_ms, 1),
        "cloud_ms": round(p * cloud_lat, 1),
        "download_ms": round(p * download_ms, 2),
        "e2e_ms": round(ec_total, 1),
        "bandwidth_kb": round(p * upload_kb, 0),
        "meets_0_2s": ec_total <= 200,
    })

    # ---- 写 CSV ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "centralized_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "architecture", "edge_ms", "upload_ms", "cloud_ms", "download_ms",
            "e2e_ms", "bandwidth_kb", "meets_0_2s"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV 已保存: {csv_path}")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("汇总：三种架构端到端延迟对比")
    print("=" * 70)
    print(f"{'架构':<14}{'边缘ms':<9}{'上传ms':<9}{'云端ms':<9}{'下传ms':<9}"
          f"{'E2E ms':<10}{'带宽KB':<10}{'≤0.2s':<7}")
    for r in rows:
        ok = "✅" if r["meets_0_2s"] else "❌"
        print(f"{r['architecture']:<14}{r['edge_ms']:<9.1f}{r['upload_ms']:<9.1f}"
              f"{r['cloud_ms']:<9.1f}{r['download_ms']:<9.2f}{r['e2e_ms']:<10.1f}"
              f"{r['bandwidth_kb']:<10.0f}{ok:<7}")

    # ---- 画图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 延迟堆叠柱状图
        ax = axes[0]
        archs = [r["architecture"] for r in rows]
        edge_v = [r["edge_ms"] for r in rows]
        up_v = [r["upload_ms"] for r in rows]
        cloud_v = [r["cloud_ms"] for r in rows]
        dl_v = [r["download_ms"] for r in rows]
        x = range(len(archs))
        ax.bar(x, edge_v, label="edge", color="#4CAF50")
        ax.bar(x, up_v, bottom=edge_v, label="upload", color="#FF9800")
        ax.bar(x, cloud_v, bottom=[e+u for e, u in zip(edge_v, up_v)],
               label="cloud", color="#2196F3")
        ax.bar(x, dl_v, bottom=[e+u+c for e, u, c in zip(edge_v, up_v, cloud_v)],
               label="download", color="#9C27B0")
        ax.axhline(200, color="red", linestyle="--", label="0.2s limit")
        ax.set_xticks(list(x))
        ax.set_xticklabels(archs)
        ax.set_ylabel("E2E latency (ms)")
        ax.set_title("End-to-End Latency Breakdown")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 带宽对比
        ax = axes[1]
        bw = [r["bandwidth_kb"] for r in rows]
        ax.bar(x, bw, color=["#FF9800", "#4CAF50", "#2196F3"])
        ax.set_xticks(list(x))
        ax.set_xticklabels(archs)
        ax.set_ylabel("Bandwidth per inference (KB)")
        ax.set_title("Bandwidth Overhead")
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"{args.model_name} | bw={args.bandwidth_mbps}Mbps | "
                     f"keep={args.keep_ratio} | ec_ratio={args.edge_cloud_ratio}",
                     fontsize=12)
        fig.tight_layout()
        png_path = out_dir / "centralized_benchmark.png"
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        print(f"图表已保存: {png_path}")
    except Exception as e:
        print(f"画图跳过: {e}")

    # 结论
    print("\n结论：")
    cen = rows[0]["e2e_ms"]
    loc = rows[1]["e2e_ms"]
    ec = rows[2]["e2e_ms"]
    print(f"  集中式 {cen}ms vs 端边协同 {ec}ms：协同快 {cen-ec:.1f}ms "
          f"({(cen-ec)/cen*100:.1f}% 提升)")
    print(f"  带宽：集中式 {rows[0]['bandwidth_kb']:.0f}KB vs 协同 {rows[2]['bandwidth_kb']:.0f}KB "
          f"(省 {rows[0]['bandwidth_kb']-rows[2]['bandwidth_kb']:.0f}KB)")


if __name__ == "__main__":
    main()
