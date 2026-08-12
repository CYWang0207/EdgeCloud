#!/usr/bin/env python3
"""Reproducible acceptance tests for the network-resilience scheduler.

This is deliberately an evaluation wrapper: it does not alter policy actions,
and keeps the former 20260804 results intact.  It produces the five deliverables
requested for the current network test: main four-mode runs, quality calibration,
foreground/background split, realtime communication probes, and u=2/Q_net study. Main runs keep MB-level S_query/Adapter/SCL traffic in Y_bw/Q_net instead of foreground T_comm; probes force payload into T_comm for stress checks only.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from network_sim import NetworkSimulator


ROOT = Path(__file__).resolve().parent
MODES = ("static", "jitter", "jitter_outage", "markov")


def probe_payloads(options):
    foreground = float(options.foreground_query_size_mb)
    adapter = float(options.adapter_size_mb)
    query = float(options.query_size_mb)
    scl = float(options.u2_update_size_mb)
    u1_foreground = foreground + adapter if options.include_adapter_in_foreground else foreground
    return (
        (u1_foreground, "u1_foreground_realtime"),
        (adapter, "adapter_payload_reference"),
        (query + adapter, "S_query+S_adapter_stress"),
        (query + scl, "S_query+SCL_weights_stress"),
    )


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxcars-input", type=Path, required=True)
    parser.add_argument("--modelnet-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Markov seeds; static/jitter/outage use the first one.")
    parser.add_argument("--disconnect-prob", type=float, default=0.10)
    parser.add_argument("--outage-period", type=int, default=50)
    parser.add_argument("--outage-duration", type=int, default=5)
    parser.add_argument("--adapter-size-mb", type=float, default=1.2,
                        help="S_adapter background Adapter payload size in MB.")
    parser.add_argument("--query-size-mb", type=float, default=5.0,
                        help="S_query long-term communication accounting size in MB; not foreground T_comm.")
    parser.add_argument("--foreground-query-size-mb", type=float, default=0.05,
                        help="Small foreground query/control payload size in MB, counted in T_comm.")
    parser.add_argument("--include-adapter-in-foreground", action="store_true",
                        help="Count u=1 adapter payload in foreground T_comm instead of async Q_net.")
    parser.add_argument("--u2-update-size-mb", type=float, default=50.0,
                        help="SCL_weights background update payload size in MB.")
    parser.add_argument("--q-net-max-mb", type=float, default=200.0,
                        help="Q_net physical queue capacity in MB; default 200; override this value for queue-size sensitivity tests.")
    return parser


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite(values):
    return [float(value) for value in values if value not in ("", None) and math.isfinite(float(value))]


def rate(rows, key):
    return float(np.mean([float(row[key]) for row in rows])) if rows else math.nan


def mean(values):
    values = finite(values)
    return float(np.mean(values)) if values else math.nan


def percentile(values, p):
    values = finite(values)
    return float(np.percentile(values, p)) if values else math.nan


def proxy_distribution(rows):
    values = finite(row["proxy_acc"] for row in rows)
    return {"proxy_acc_min": min(values), "proxy_acc_p5": percentile(values, 5),
            "proxy_acc_p50": percentile(values, 50), "proxy_acc_p95": percentile(values, 95),
            "proxy_acc_max": max(values)}


def completed_latency(rows, prefix, metric):
    count = sum(int(float(row[f"{prefix}_completed_event_count"])) for row in rows)
    total = sum(float(row[f"{prefix}_{metric}_sum"]) for row in rows)
    return total / count if count else math.nan


def summarize_run(rows):
    result = {
        "slots": len(rows),
        "business_continuity_rate": rate(rows, "business_available"),
        "avg_foreground_e2e_delay_ms": mean(row["e2e_delay_ms"] for row in rows),
        "disconnect_rate": rate(rows, "is_disconnected"),
        "deadline_met_rate": rate(rows, "deadline_met"),
        "adapter_task_latency_ms": completed_latency(rows, "adapter", "task_latency_ms"),
        "adapter_queue_latency_ms": completed_latency(rows, "adapter", "queue_latency_ms"),
        "scl_task_latency_ms": completed_latency(rows, "scl", "task_latency_ms"),
        "scl_queue_latency_ms": completed_latency(rows, "scl", "queue_latency_ms"),
        "adapter_completion_rate": float(rows[-1]["adapter_completion_rate"]),
        "scl_completion_rate": float(rows[-1]["u2_update_completion_rate"]),
        "drop_ratio": float(rows[-1]["drop_ratio"]),
        "ttl_expired_mb": sum(float(row["ttl_expired_mb"]) for row in rows),
        "cap_drop_mb": sum(float(row["cap_drop_mb"]) for row in rows),
        "u2_count": sum(int(float(row["u"])) == 2 for row in rows),
        "completed_events": sum(int(float(row["completed_event_count"])) for row in rows),
        "max_Q_net": max(float(row["Q_net"]) for row in rows),
        "Q_net_max": max(float(row["Q_net_max"]) for row in rows),
        "max_pending_comm": max(float(row["pending_comm"]) for row in rows),
        "total_served_comm": sum(float(row["served_comm"]) for row in rows),
    }
    result["u2_rate"] = result["u2_count"] / max(result["slots"], 1)
    result["pass"] = (result["business_continuity_rate"] >= .90 and
                      result["avg_foreground_e2e_delay_ms"] <= 200.0)
    result.update(proxy_distribution(rows))
    return result


def run_policy(input_path, output_path, mode, seed, options):
    command = [sys.executable, str(ROOT / "main_edge_cloud_new.py"),
               "--input", str(input_path), "--output", str(output_path),
               "--network-mode", mode, "--seed", str(seed),
               "--business-min-active-views", "3",
               "--disconnect-prob", str(options.disconnect_prob),
               "--outage-period", str(options.outage_period),
               "--outage-duration", str(options.outage_duration),
               "--adapter-size-mb", str(options.adapter_size_mb),
               "--query-size-mb", str(options.query_size_mb),
               "--foreground-query-size-mb", str(options.foreground_query_size_mb),
               "--u2-update-size-mb", str(options.u2_update_size_mb)]
    if options.q_net_max_mb is not None:
        command.extend(["--q-net-max-mb", str(options.q_net_max_mb)])
    if options.include_adapter_in_foreground:
        command.append("--include-adapter-in-foreground")
    with output_path.with_suffix(".console.txt").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quality_ratio(rows, static_rows):
    ratios = []
    for row, baseline in zip(rows, static_rows):
        denominator = float(baseline["proxy_acc"])
        if denominator > 0:
            ratios.append(float(row["proxy_acc"]) / denominator)
    return float(np.mean(np.asarray(ratios) >= .8)) if ratios else math.nan


def run_probes(output_dir, options):
    records = []
    seeds = options.seeds
    for mode in MODES:
        for seed in ([seeds[0]] if mode != "markov" else seeds):
            net = NetworkSimulator(
                mode=mode,
                seed=seed,
                disconnect_prob=options.disconnect_prob,
                outage_period=options.outage_period,
                outage_duration=options.outage_duration,
                adapter_size_mb=options.adapter_size_mb,
                query_size_mb=options.query_size_mb,
                foreground_query_size_mb=options.foreground_query_size_mb,
                include_adapter_in_foreground=options.include_adapter_in_foreground,
                u2_update_size_mb=options.u2_update_size_mb,
                q_net_max_mb=options.q_net_max_mb,
            )
            state = net.step()
            for payload_mb, payload_name in probe_payloads(options):
                info = net.compute_e2e(u=0, realtime_comm=payload_mb)
                records.append({"mode": mode, "seed": seed, "payload_mb": payload_mb,
                                "payload_name": payload_name, "network_state": state["network_state"],
                                "effective_bandwidth_mbps": state["effective_bandwidth_mbps"],
                                "loss_rate": state["loss_rate"], "disconnect_flag": int(state["disconnect_flag"]),
                                "comm_delay_ms": info["comm_delay_ms"], "e2e_delay_ms": info["e2e_delay_ms"],
                                "transmission_success": int(info["transmission_success"]),
                                "deadline_met": int(info["deadline_met"])})
    # Explicit optional synchronous-u2 path requested in the test specification.
    net = NetworkSimulator(
        mode="markov",
        seed=seeds[0],
        sync_u2=True,
        disconnect_prob=options.disconnect_prob,
        outage_period=options.outage_period,
        outage_duration=options.outage_duration,
        adapter_size_mb=options.adapter_size_mb,
        query_size_mb=options.query_size_mb,
        foreground_query_size_mb=options.foreground_query_size_mb,
        include_adapter_in_foreground=options.include_adapter_in_foreground,
        u2_update_size_mb=options.u2_update_size_mb,
        q_net_max_mb=options.q_net_max_mb,
    )
    state = net.step()
    sync_payload_mb = options.query_size_mb + options.u2_update_size_mb
    info = net.compute_e2e(u=2, realtime_comm=sync_payload_mb)
    records.append({"mode": "markov_sync_u2", "seed": seeds[0], "payload_mb": sync_payload_mb,
                    "payload_name": "S_query+SCL_weights_sync_stress", "network_state": state["network_state"],
                    "effective_bandwidth_mbps": state["effective_bandwidth_mbps"], "loss_rate": state["loss_rate"],
                    "disconnect_flag": int(state["disconnect_flag"]), "comm_delay_ms": info["comm_delay_ms"],
                    "e2e_delay_ms": info["e2e_delay_ms"], "transmission_success": int(info["transmission_success"]),
                    "deadline_met": int(info["deadline_met"])})
    write_csv(output_dir / "communication_probes.csv", records)

def make_high_struct_drift(input_path, output_path):
    """Create the specified extra trajectory without changing original measured rows."""
    rows = read_rows(input_path)
    fields = list(rows[0])
    if "struct_drift" not in fields:
        fields.append("struct_drift")
    for row in rows:
        row["struct_drift"] = "0.95"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    options = args_parser().parse_args()
    for path in (options.boxcars_input, options.modelnet_input):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = options.output_dir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result directory: {output_dir}")
    output_dir.mkdir(parents=True)

    all_results, logs = [], {}
    datasets = {"boxcars": options.boxcars_input, "modelnet40": options.modelnet_input}
    for dataset, input_path in datasets.items():
        for mode in MODES:
            run_seeds = options.seeds if mode == "markov" else [options.seeds[0]]
            for seed in run_seeds:
                output_path = output_dir / f"{dataset}_{mode}_seed{seed}.csv"
                run_policy(input_path, output_path, mode, seed, options)
                rows = read_rows(output_path)
                logs[(dataset, mode, seed)] = rows
                all_results.append({"dataset": dataset, "network_mode": mode, "seed": seed,
                                    **summarize_run(rows)})

    # All quality ratios use the same static seed and the same per-slot input row.
    for row in all_results:
        baseline = logs[(row["dataset"], "static", options.seeds[0])]
        row["quality_ratio_pass_rate"] = quality_ratio(logs[(row["dataset"], row["network_mode"], row["seed"])], baseline)
    write_csv(output_dir / "per_seed_summary.csv", all_results)

    grouped = []
    for dataset in datasets:
        for mode in MODES:
            candidates = [row for row in all_results if row["dataset"] == dataset and row["network_mode"] == mode]
            aggregate = {"dataset": dataset, "network_mode": mode, "seeds": ",".join(str(row["seed"]) for row in candidates)}
            for key in candidates[0]:
                if key not in {"dataset", "network_mode", "seed", "pass"} and isinstance(candidates[0][key], (int, float, bool)):
                    aggregate[key] = float(np.mean([row[key] for row in candidates]))
            aggregate["pass"] = int(aggregate["business_continuity_rate"] >= .90 and aggregate["avg_foreground_e2e_delay_ms"] <= 200.0)
            grouped.append(aggregate)
    write_csv(output_dir / "main_experiment_summary.csv", grouped)

    run_probes(output_dir, options)
    high_input = output_dir / "high_struct_drift_trajectory.csv"
    make_high_struct_drift(options.boxcars_input, high_input)
    high_output = output_dir / f"boxcars_high_struct_drift_markov_seed{options.seeds[0]}.csv"
    run_policy(high_input, high_output, "markov", options.seeds[0], options)
    high_rows = read_rows(high_output)
    high_summary = summarize_run(high_rows)
    high_summary["high_struct_drift_slots"] = len(high_rows)
    if high_summary["u2_count"] == 0:
        high_summary["u2_not_selected_reason_fields"] = "G_raw_u0..u2,G_effective_u0..u2 in per-slot CSV"
    (output_dir / "u2_qnet_summary.json").write_text(json.dumps(high_summary, indent=2, allow_nan=True), encoding="utf-8")

    (output_dir / "README.md").write_text(
        "# 网络韧性验收（前台/后台分离口径）\n\n"
        "正式验收仅使用 `business_continuity_rate >= 90%` 与 "
        "`avg_foreground_e2e_delay_ms <= 200ms`。主实验的前台 e2e 统计当前业务必须阻塞等待的实时通信，"
        "其中轻量前台请求由 foreground_query_size_mb 计入；"
        f"通信量参数：S_query={options.query_size_mb} MB，"
        f"S_adapter={options.adapter_size_mb} MB，SCL_weights={options.u2_update_size_mb} MB，"
        f"include_adapter_in_foreground={options.include_adapter_in_foreground}，"
        f"Q_net_max={options.q_net_max_mb}。"
        "S_query 与 SCL 的 MB 级流量保留在 Y_bw/Q_net 的长期/后台通信口径中；"
        "Adapter 默认在后台队列，开启 include_adapter_in_foreground 时改为进入前台 T_comm。\n\n"
        "`main_experiment_summary.csv` 是交付主表；`per_seed_summary.csv` 保留 Markov 三 seed；"
        "各原始 CSV 含前台、后台和 G 诊断字段。\n\n"
        "`communication_probes.csv` 是强制将 payload 放入当前时隙 T_comm 的压力探针，"
        "不代表主实验每时隙都实时传输该 payload。"
        "`u2_qnet_summary.json` 与 high_struct_drift CSV 是 u=2/Q_net 验证。\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
