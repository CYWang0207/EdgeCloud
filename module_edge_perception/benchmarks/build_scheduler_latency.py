"""
接口1产出：边缘感知 -> 调度器 的推理延迟字段。

从 latency_results.csv 提取硬剪枝的 median_ms，转为接口1要求的格式：
    model_name, keep_ratio, seq_len, inference_latency_ms, mode

供 A/D 的调度器（main_edge_cloud_new.py）作为节点调度代价参数使用。

用法（从 module_edge_perception/ 目录运行）：
    py -3.11 benchmarks/build_scheduler_latency.py
"""
import csv
from pathlib import Path


def build_interface1_output():
    results_dir = Path(__file__).resolve().parent / "results"
    out_rows = []

    # ViT-Small
    small_csv = results_dir / "latency_results.csv"
    if small_csv.exists():
        with small_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["mode"] in ("baseline", "hard_prune"):
                    out_rows.append({
                        "model_name": "vit_small_patch16_224",
                        "keep_ratio": row["keep_ratio"],
                        "seq_len": row["seq_len"],
                        "inference_latency_ms": round(float(row["median_ms"]), 2),
                        "mode": row["mode"],
                    })

    # ViT-Tiny
    tiny_csv = results_dir / "tiny" / "latency_results.csv"
    if tiny_csv.exists():
        with tiny_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["mode"] in ("baseline", "hard_prune"):
                    out_rows.append({
                        "model_name": "vit_tiny_patch16_224",
                        "keep_ratio": row["keep_ratio"],
                        "seq_len": row["seq_len"],
                        "inference_latency_ms": round(float(row["median_ms"]), 2),
                        "mode": row["mode"],
                    })

    out_path = results_dir / "latency_for_scheduler.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_name", "keep_ratio", "seq_len", "inference_latency_ms", "mode"
        ])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"接口1延迟数据已保存: {out_path}")
    print(f"共 {len(out_rows)} 条记录")


if __name__ == "__main__":
    build_interface1_output()
