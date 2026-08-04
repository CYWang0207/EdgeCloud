"""
端到端时延骨架：u=0/u=1/u=2 三种协同模式。

公式（组长定义）：
  T_e2e = T_edge + T_comm + T_cloud

三种协同模式：
  u=0  本地推理      T_comm=0,            T_cloud=0
  u=1  adapter 同步   T_comm = 8*(S_adapter+S_query)/R*1000 + RTT,  T_cloud=云端推理
  u=2  重训          T_comm = 8*S_adapter/R*1000 + RTT,            T_cloud=重训时间

参数：
  S_adapter = 1.2 MB（adapter 下发体积，张晨权重实测）
  S_query   = 边缘上送 query 数据体积（默认 0.6 MB，4视角缩略图）
  R         = 边云带宽（Mbps）
  RTT       = 往返时延（ms）
  T_cloud   = 云端推理时间（实测 cloud_model forward）
  T_retrain = adapter 重训时间（实测或建模，默认按小 epoch 估计）

T_edge 为实测：MV-ViT 硬剪枝前向（含 adapter）。

用法（从 module_edge_perception/ 目录运行）：
    py -3.11 benchmarks/benchmark_e2e.py
    py -3.11 benchmarks/benchmark_e2e.py --keep-ratio 0.1 --adapter-r 32 \
        --adapter-checkpoint "D:/.../adapter_best.pth"
    py -3.11 benchmarks/benchmark_e2e.py --bandwidth-mbps 20 --rtt-ms 30

输出：
    benchmarks/results/e2e_modes.csv
    benchmarks/results/e2e_modes.png
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
    p = argparse.ArgumentParser(description="端到端时延骨架：u=0/u=1/u=2 三模式")
    p.add_argument("--model-name", default="vit_small_patch16_224",
                   choices=["vit_small_patch16_224", "vit_tiny_patch16_224"])
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--keep-ratio", type=float, default=0.2,
                   help="边缘硬剪枝保留率")
    # adapter
    p.add_argument("--adapter-r", type=int, default=32)
    p.add_argument("--adapter-scale", type=float, default=1.0)
    p.add_argument("--adapter-checkpoint", default=None,
                   help="adapter 权重路径（加载后 T_edge 含真实 adapter 开销）")
    # 通信参数
    p.add_argument("--bandwidth-mbps", type=float, default=20.0,
                   help="边云带宽 R（Mbps）")
    p.add_argument("--rtt-ms", type=float, default=30.0,
                   help="往返时延 RTT（ms）")
    p.add_argument("--s-adapter-mb", type=float, default=1.2,
                   help="adapter 下发体积 S_adapter（MB），张晨权重实测 1.2MB")
    p.add_argument("--s-query-mb", type=float, default=0.6,
                   help="边缘上送 query 体积 S_query（MB），默认 0.6（4视角缩略图）")
    # 重训参数
    p.add_argument("--retrain-ms", type=float, default=5000.0,
                   help="adapter 重训时间 T_retrain（ms），默认 5000（小 epoch 估计）")
    # 测量
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("端到端时延骨架：u=0 / u=1 / u=2 三模式")
    print("=" * 70)
    print(f"设备: {device}  模型: {args.model_name}")
    print(f"边缘硬剪枝 keep={args.keep_ratio}  adapter r={args.adapter_r}")
    print(f"带宽 R={args.bandwidth_mbps} Mbps  RTT={args.rtt_ms} ms")
    print(f"S_adapter={args.s_adapter_mb} MB  S_query={args.s_query_mb} MB")
    print(f"T_retrain={args.retrain_ms} ms")
    print()

    # 构建边缘模型（硬剪枝 + adapter）
    print("构建边缘模型...")
    edge_model = EarlyFusionMultiViewViT(
        model_name=args.model_name, num_views=args.num_views,
        num_classes=args.num_classes, pretrained=False,
    ).to(device).eval()

    from adaptformer import (
        attach_adaptformer, set_adapter_enabled,
        count_adapter_parameters, load_adapter_checkpoint,
    )
    attach_adaptformer(edge_model, r=args.adapter_r, scale=args.adapter_scale)
    n_adapter = count_adapter_parameters(edge_model)
    s_adapter_real_mb = n_adapter * 4 / (1024 ** 2)
    print(f"adapter 参数: {n_adapter:,}  实际下发体积: {s_adapter_real_mb:.3f} MB")

    if args.adapter_checkpoint:
        import os
        if os.path.exists(args.adapter_checkpoint):
            load_adapter_checkpoint(edge_model, args.adapter_checkpoint,
                                    device=str(device))
            print(f"adapter 权重已加载: {args.adapter_checkpoint}")
        else:
            print(f"⚠️ 权重不存在: {args.adapter_checkpoint}，用 identity 初始化")
    set_adapter_enabled(edge_model, True)

    # 构建云端模型（完整无剪枝 + adapter）
    print("构建云端模型...")
    cloud_model = EarlyFusionMultiViewViT(
        model_name=args.model_name, num_views=args.num_views,
        num_classes=args.num_classes, pretrained=False,
    ).to(device).eval()
    attach_adaptformer(cloud_model, r=args.adapter_r, scale=args.adapter_scale)
    if args.adapter_checkpoint and os.path.exists(args.adapter_checkpoint):
        load_adapter_checkpoint(cloud_model, args.adapter_checkpoint,
                                device=str(device))
    set_adapter_enabled(cloud_model, True)

    images = torch.randn(args.batch_size, args.num_views, 3, 224, 224, device=device)

    # ---- 实测各组件延迟 ----
    print("\n测量组件延迟...")
    t_edge = measure(
        lambda: edge_model.forward_hard_prune(
            images, keep_ratio=args.keep_ratio, token_score_mode="importance"),
        args.warmup, args.repeat)
    t_cloud = measure(lambda: cloud_model(images), args.warmup, args.repeat)
    print(f"  T_edge (硬剪枝+adapter, keep={args.keep_ratio}): {t_edge:.1f} ms")
    print(f"  T_cloud (完整+adapter): {t_cloud:.1f} ms")

    # ---- 通信延迟建模 ----
    # T_comm = 8 * (bytes) / (R * 1e6) * 1000  [ms]，bytes = MB * 1024 * 1024
    R = args.bandwidth_mbps
    RTT = args.rtt_ms
    S_adp = args.s_adapter_mb
    S_qry = args.s_query_mb

    def comm_ms(up_mb, down_mb):
        """上下行传输延迟（ms）。"""
        bits = (up_mb + down_mb) * 1024 * 1024 * 8
        return bits / (R * 1e6) * 1000 + RTT

    # ---- 三种模式 ----
    rows = []

    # u=0 本地推理
    t_comm_u0 = 0.0
    t_cloud_u0 = 0.0
    t_e2e_u0 = t_edge + t_comm_u0 + t_cloud_u0
    rows.append({
        "mode": "u0_local",
        "u": 0,
        "T_edge_ms": round(t_edge, 1),
        "T_comm_ms": round(t_comm_u0, 1),
        "T_cloud_ms": round(t_cloud_u0, 1),
        "T_e2e_ms": round(t_e2e_u0, 1),
        "comm_up_mb": 0.0,
        "comm_down_mb": 0.0,
        "meets_0_2s": t_e2e_u0 <= 200,
        "desc": "纯边缘本地推理，不上云",
    })

    # u=1 adapter 同步：上行 query + 下行 adapter + 云端推理
    t_comm_u1 = comm_ms(S_qry, S_adp)  # 上行 query，下行 adapter
    t_cloud_u1 = t_cloud
    t_e2e_u1 = t_edge + t_comm_u1 + t_cloud_u1
    rows.append({
        "mode": "u1_adapter_sync",
        "u": 1,
        "T_edge_ms": round(t_edge, 1),
        "T_comm_ms": round(t_comm_u1, 1),
        "T_cloud_ms": round(t_cloud_u1, 1),
        "T_e2e_ms": round(t_e2e_u1, 1),
        "comm_up_mb": S_qry,
        "comm_down_mb": S_adp,
        "meets_0_2s": t_e2e_u1 <= 200,
        "desc": "边缘上送query→云端推理→下发adapter",
    })

    # u=2 重训：云端重训 adapter + 下发
    t_comm_u2 = comm_ms(0.0, S_adp)  # 仅下行新 adapter
    t_cloud_u2 = args.retrain_ms
    t_e2e_u2 = t_edge + t_comm_u2 + t_cloud_u2
    rows.append({
        "mode": "u2_retrain",
        "u": 2,
        "T_edge_ms": round(t_edge, 1),
        "T_comm_ms": round(t_comm_u2, 1),
        "T_cloud_ms": round(t_cloud_u2, 1),
        "T_e2e_ms": round(t_e2e_u2, 1),
        "comm_up_mb": 0.0,
        "comm_down_mb": S_adp,
        "meets_0_2s": t_e2e_u2 <= 200,
        "desc": "云端重训adapter→下发（漂移恢复）",
    })

    # ---- 写 CSV ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "e2e_modes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "u", "T_edge_ms", "T_comm_ms", "T_cloud_ms", "T_e2e_ms",
            "comm_up_mb", "comm_down_mb", "meets_0_2s", "desc"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV 已保存: {csv_path}")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print(f"端到端时延：u=0/u=1/u=2 三模式  (R={R}Mbps, RTT={RTT}ms)")
    print("=" * 70)
    print(f"{'模式':<18}{'u':<4}{'T_edge':<10}{'T_comm':<10}{'T_cloud':<12}"
          f"{'T_e2e':<10}{'≤0.2s':<7}")
    for r in rows:
        ok = "✅" if r["meets_0_2s"] else "❌"
        print(f"{r['mode']:<18}{r['u']:<4}{r['T_edge_ms']:<10.1f}"
              f"{r['T_comm_ms']:<10.1f}{r['T_cloud_ms']:<12.1f}"
              f"{r['T_e2e_ms']:<10.1f}{ok:<7}")
    print()
    for r in rows:
        print(f"  u={r['u']} {r['desc']}")

    # ---- 画图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        modes = [r["mode"] for r in rows]
        edge_v = [r["T_edge_ms"] for r in rows]
        comm_v = [r["T_comm_ms"] for r in rows]
        cloud_v = [r["T_cloud_ms"] for r in rows]
        x = range(len(modes))
        ax.bar(x, edge_v, label="T_edge", color="#4CAF50")
        ax.bar(x, comm_v, bottom=edge_v, label="T_comm", color="#FF9800")
        ax.bar(x, cloud_v, bottom=[e+c for e, c in zip(edge_v, comm_v)],
               label="T_cloud", color="#2196F3")
        ax.axhline(200, color="red", linestyle="--", label="0.2s limit")
        ax.set_xticks(list(x))
        ax.set_xticklabels(modes, rotation=15)
        ax.set_ylabel("T_e2e (ms)")
        ax.set_title(f"E2E Latency by Mode  |  R={R}Mbps RTT={RTT}ms "
                     f"keep={args.keep_ratio} adapter={args.adapter_r}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        # 标注总数
        for i, r in enumerate(rows):
            ax.text(i, r["T_e2e_ms"] + 5, f"{r['T_e2e_ms']:.0f}ms",
                    ha="center", fontsize=11, fontweight="bold")
        fig.tight_layout()
        png_path = out_dir / "e2e_modes.png"
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        print(f"图表已保存: {png_path}")
    except Exception as e:
        print(f"画图跳过: {e}")

    # 结论
    print("\n结论：")
    print(f"  u=0 本地: {t_e2e_u0:.1f}ms — 最快，但无云端兜底")
    print(f"  u=1 同步: {t_e2e_u1:.1f}ms — {'达标 ✅' if t_e2e_u1<=200 else '超标 ❌'}"
          f"（通信 {t_comm_u1:.0f}ms 是主要开销）")
    print(f"  u=2 重训: {t_e2e_u2:.0f}ms — 漂移恢复用，非实时决策路径")


if __name__ == "__main__":
    main()
