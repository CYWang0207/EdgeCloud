"""单节点云边协同 RL 调度主循环（已接 NetworkSimulator 网络韧性层）。

原 Actor-Critic + Lyapunov + 注水算法逻辑不变，在主循环外按 CLAUDE.md 第五节
套一层"时变网络 + 时延 + 业务可用性"：

  每时隙 t:
    1. net.step()              采样 R_t / B_t / net_state / is_disconnected
    2. Actor 生成 (v, u) 候选
    3. net.filter_candidates   断联→强制 u=0；--strict-bandwidth→硬过滤超带宽候选
    4. Critic(注水) 评估剩余候选，net.apply_network_penalty 对超带宽软罚 G
    5. 选最优 (v*, u*, k*)
    6. net.compute_e2e         算 T_comm + T_cloud → T_e2e（u=0 特判 T_comm=0）
    7. net.is_business_available 业务可用四条件（防"断联切本地=100%可用"虚高）
    8. net.update_queues        更新 Y_bw（Lyapunov 虚拟队列）+ Q_net（物理积压，带上限/TTL）

理论故事锁死：Q_net 是物理传输积压队列，不进效用目标 G 的罚项；长期平均带宽
约束仍由 Y_bw 单一虚拟队列保证，Lyapunov 证明不变。
"""
import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

from network_sim import DEFAULT_ACC_FLOOR, NetworkSimulator
from actor_memory import CollaborativeMemoryDNN
from critic_water_filling import WaterFilling_Critic


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run edge-cloud RL scheduling on measured trajectory data (with network resilience)."
    )
    parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "real_trajectory_data.csv")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "rl_decision_log.csv")
    parser.add_argument("--b-avg", type=float, default=3.0, help="Average bandwidth budget per slot (MB/slot).")
    parser.add_argument("--v-lya", type=float, default=20.0, help="Lyapunov utility weight.")
    parser.add_argument("--eps-mask", type=float, default=0.05, help="Mask views whose w_i is below this value.")
    parser.add_argument("--min-active-views", type=float, default=3.0,
                        help="Minimum active views per slot. Use 3 to guarantee average >= 2.5.")
    parser.add_argument("--m-candidates", type=int, default=10, help="Actor candidate count before u cross product.")
    parser.add_argument("--scl-weights", type=float, default=50.0, help="Bandwidth cost of retrained model weights.")
    parser.add_argument("--alpha-env", type=float, default=0.4, help="Local accuracy penalty weight for E_drift.")
    parser.add_argument("--alpha-struct", type=float, default=0.3, help="Accuracy penalty weight for structural drift.")
    parser.add_argument("--retrain-bonus", type=float, default=0.2, help="Extra gain from retraining under severe structural drift.")
    parser.add_argument("--tau-retrain", type=float, default=0.5, help="Structural drift threshold for retraining gain.")
    parser.add_argument("--seed", type=int, default=42)

    # --- 网络韧性参数（CLAUDE.md 第五节）---
    parser.add_argument("--network-mode", default="static",
                        choices=("static", "jitter", "jitter_outage", "markov"),
                        help="static=基线对照；jitter=带宽抖动；jitter_outage=抖动+断联；markov=GOOD/WEAK/DOWN")
    parser.add_argument("--slot-duration", type=float, default=0.2, help="单时隙时长(s)，B_t=R_t*slot/8")
    parser.add_argument("--edge-delay-ms", type=float, default=80.0, help="T_edge 边缘推理延迟")
    parser.add_argument("--rtt-ms", type=float, default=10.0, help="往返时延，T_comm 计入")
    parser.add_argument("--deadline-ms", type=float, default=200.0, help="端到端时延硬指标上限(≤0.2s)")
    parser.add_argument(
        "--acc-floor",
        type=float,
        default=DEFAULT_ACC_FLOOR,
        help=f"业务可用 Critic quality proxy_acc 下限，默认 {DEFAULT_ACC_FLOOR}",
    )
    parser.add_argument("--business-min-active-views", type=int, default=3,
                        help="业务可用最小激活视角数（默认与 Actor 的 min_active_views=3 对齐）")
    parser.add_argument("--bw-min-mbps", type=float, default=20.0)
    parser.add_argument("--bw-max-mbps", type=float, default=120.0)
    parser.add_argument("--disconnect-prob", type=float, default=0.0, help="jitter_outage 随机断联概率")
    parser.add_argument("--outage-period", type=int, default=0, help="周期断联周期(时隙)，0=不周期断联")
    parser.add_argument("--outage-duration", type=int, default=0, help="周期断联持续(时隙)")
    parser.add_argument("--strict-bandwidth", action="store_true", help="超带宽直接过滤候选（默认仅软罚）")
    parser.add_argument("--sync-u2", action="store_true", help="u=2 实时同步（默认异步后台入 Q_net）")
    parser.add_argument("--adapter-size-mb", type=float, default=1.2, help="u=1 adapter 下发体积(MB)")
    parser.add_argument("--u2-update-size-mb", type=float, default=50.0, help="u=2 重训权重体积(MB)")
    return parser.parse_args()


def load_trajectory(path, num_views):
    if not path.exists():
        raise FileNotFoundError(f"找不到轨迹文件: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"轨迹文件为空: {path}")

    required = ["E_drift", *[f"w_{i + 1}" for i in range(num_views)]]
    missing = [col for col in required if col not in rows[0]]
    if missing:
        raise ValueError(f"轨迹文件缺少必要列: {missing}")

    return rows


def float_or_empty(row, key):
    value = row.get(key, "")
    return "" if value == "" else float(value)


def int_or_empty(row, key):
    value = row.get(key, "")
    return "" if value == "" else int(float(value))


def build_log_row(row, t, y_bw, w_t, struct_drift, v_opt, u_opt, k_opt,
                 c_comm, acc, cost, best_g, net_state, e2e_info,
                 business_available, q_net, queue_result, candidate_scores):
    log_row = {
        "t": t,
        "sample_id": int_or_empty(row, "sample_id"),
        "label": int_or_empty(row, "label"),
        "pred": int_or_empty(row, "pred"),
        "confidence": float_or_empty(row, "confidence"),
        "E_drift": float(row["E_drift"]),
        "drift_type": row.get("drift_type", ""),
        "severity": float_or_empty(row, "severity"),
        "struct_drift": float(struct_drift),
        "u": int(u_opt),
        "c_comm": float(c_comm),
        "proxy_acc": float(acc),
        "cost": float(cost),
        "G": float(best_g),
        "Y_bw": float(y_bw),
        "Q_net": float(q_net),
        "active_views": int(np.sum(v_opt)),
        "active_token_ratio_sum": float(np.sum(k_opt)),
        "network_state": net_state.get("network_state", ""),
        "bandwidth_mbps": float(net_state.get("bandwidth_mbps", 0.0)),
        "effective_bandwidth_mbps": float(net_state.get("effective_bandwidth_mbps", 0.0)),
        "B_t": float(net_state.get("B_t", 0.0)),
        "is_disconnected": int(bool(net_state.get("is_disconnected", False))),
        "comm_delay_ms": float(e2e_info.get("comm_delay_ms", 0.0)),
        "e2e_delay_ms": float(e2e_info.get("e2e_delay_ms", 0.0)),
        "deadline_met": int(bool(e2e_info.get("deadline_met", False))),
        "transmission_success": int(bool(e2e_info.get("transmission_success", True))),
        "business_available": int(bool(business_available)),
    }

    # 后台更新链路按时隙落盘；completed_events 的延迟在这里按任务类型汇总，
    # 以便测试脚本无需重放模拟器状态即可统计完整的队列指标。
    for key in (
        "pending_comm", "served_comm", "background_comm", "realtime_comm",
        "ttl_expired_mb", "cap_drop_mb", "drop_ratio", "adapter_drop_ratio",
        "adapter_completion_rate", "u2_update_completion_rate",
    ):
        log_row[key] = float(queue_result.get(key, 0.0))
    log_row["completed_event_count"] = int(queue_result.get("completed_event_count", 0))
    for task_type, prefix in (("adapter", "adapter"), ("scl_weights", "scl")):
        events = [event for event in queue_result.get("completed_events", [])
                  if event.get("task_type") == task_type]
        log_row[f"{prefix}_completed_event_count"] = len(events)
        for metric in ("task_latency_ms", "queue_latency_ms"):
            log_row[f"{prefix}_{metric}_sum"] = float(sum(event[metric] for event in events))

    # 高结构漂移诊断：每种 u 记录该槽所有候选中的最高原始/网络惩罚后评分。
    for u in (0, 1, 2):
        score = candidate_scores.get(u)
        log_row[f"G_raw_u{u}"] = "" if score is None else float(score["G_raw"])
        log_row[f"G_effective_u{u}"] = "" if score is None else float(score["G_effective"])

    for i, value in enumerate(w_t, start=1):
        log_row[f"w_{i}"] = float(value)
    for i, value in enumerate(v_opt, start=1):
        log_row[f"v_{i}"] = int(value)
    for i, value in enumerate(k_opt, start=1):
        log_row[f"k_{i}"] = float(value)

    return log_row


def save_decision_log(path, rows, num_views):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "t",
        "sample_id",
        "label",
        "pred",
        "confidence",
        "E_drift",
        "drift_type",
        "severity",
        "struct_drift",
        *[f"w_{i + 1}" for i in range(num_views)],
        *[f"v_{i + 1}" for i in range(num_views)],
        "u",
        *[f"k_{i + 1}" for i in range(num_views)],
        "c_comm",
        "proxy_acc",
        "cost",
        "G",
        "Y_bw",
        "Q_net",
        "active_views",
        "active_token_ratio_sum",
        "network_state",
        "bandwidth_mbps",
        "effective_bandwidth_mbps",
        "B_t",
        "is_disconnected",
        "comm_delay_ms",
        "e2e_delay_ms",
        "deadline_met",
        "transmission_success",
        "business_available",
        "pending_comm",
        "served_comm",
        "background_comm",
        "realtime_comm",
        "ttl_expired_mb",
        "cap_drop_mb",
        "drop_ratio",
        "adapter_drop_ratio",
        "adapter_completion_rate",
        "u2_update_completion_rate",
        "completed_event_count",
        "adapter_completed_event_count",
        "adapter_task_latency_ms_sum",
        "adapter_queue_latency_ms_sum",
        "scl_completed_event_count",
        "scl_task_latency_ms_sum",
        "scl_queue_latency_ms_sum",
        "G_raw_u0",
        "G_effective_u0",
        "G_raw_u1",
        "G_effective_u1",
        "G_raw_u2",
        "G_effective_u2",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(log_rows, y_bw):
    u_counter = Counter(row["u"] for row in log_rows)
    total = len(log_rows)
    avg_g = np.mean([row["G"] for row in log_rows])
    avg_comm = np.mean([row["c_comm"] for row in log_rows])
    avg_active_views = np.mean([row["active_views"] for row in log_rows])
    avg_token_sum = np.mean([row["active_token_ratio_sum"] for row in log_rows])
    business_rate = np.mean([row["business_available"] for row in log_rows])
    deadline_rate = np.mean([row["deadline_met"] for row in log_rows])
    disconnect_rate = np.mean([row["is_disconnected"] for row in log_rows])
    e2e_values = [row["e2e_delay_ms"] for row in log_rows
                  if math.isfinite(row["e2e_delay_ms"])]
    avg_e2e = float(np.mean(e2e_values)) if e2e_values else float("nan")
    p95_e2e = float(np.percentile(e2e_values, 95)) if e2e_values else float("nan")

    print("-" * 72)
    print("仿真结束")
    print(f"总时隙数: {total}")
    print(f"最终带宽队列 Y_bw: {y_bw[-1]:.4f}")
    print(f"平均单步目标 G: {avg_g:.4f}")
    print(f"平均通信开销: {avg_comm:.4f} MB/slot")
    print(f"平均激活视角数: {avg_active_views:.4f}")
    print(f"平均 Token 保留率总和: {avg_token_sum:.4f}")
    print(f"业务保持率: {business_rate:.2%}  (硬指标 ≥90%)")
    print(f"端到端时延达标率: {deadline_rate:.2%}  平均={avg_e2e:.1f}ms  P95={p95_e2e:.1f}ms  (≤{200}ms)")
    print(f"断联率: {disconnect_rate:.2%}")
    for u in [0, 1, 2]:
        count = u_counter.get(u, 0)
        print(f"u={u} 次数: {count} ({count / total:.2%})")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # --- 1. 系统参数初始化 ---
    num_views = 4
    sys_params = {
        "N": 196,
        "eta": 5e-5,
        "gamma": 10.0,
        "k_min": 0.1,
        "beta_0": 0.2,
        "S_adapter": args.adapter_size_mb,  # adapter 参数下发带宽消耗 (MB)，u=1 通信开销口径
        "SCL_weights": args.u2_update_size_mb,
        "S_query": 5.0,
        "alpha_env": args.alpha_env,
        "alpha_struct": args.alpha_struct,
        "retrain_bonus": args.retrain_bonus,
        "tau_retrain": args.tau_retrain,
    }
    # 每种 u 的通信开销查表（供 NetworkSimulator.filter_candidates 使用）
    c_comm_map = {
        0: 0.0,
        1: sys_params["S_adapter"] + sys_params["S_query"],
        2: sys_params["SCL_weights"] + sys_params["S_query"],
    }

    # --- 2. 加载真实轨迹数据 ---
    trajectory_rows = load_trajectory(args.input, num_views)
    total_slots = len(trajectory_rows)

    # --- 3. 初始化 Actor-Critic 记忆网络与网络韧性仿真器 ---
    mem = CollaborativeMemoryDNN(
        V=num_views,
        state_dim=3,
        eps_mask=args.eps_mask,
        min_active_views=args.min_active_views,
    )
    net = NetworkSimulator(
        mode=args.network_mode,
        slot_duration=args.slot_duration,
        b_avg=args.b_avg,
        bandwidth_min_mbps=args.bw_min_mbps,
        bandwidth_max_mbps=args.bw_max_mbps,
        disconnect_prob=args.disconnect_prob,
        outage_period=args.outage_period,
        outage_duration=args.outage_duration,
        rtt_ms=args.rtt_ms,
        edge_delay_ms=args.edge_delay_ms,
        deadline_ms=args.deadline_ms,
        acc_floor=args.acc_floor,
        business_min_active_views=args.business_min_active_views,
        adapter_size_mb=args.adapter_size_mb,
        u2_update_size_mb=args.u2_update_size_mb,
        strict_bandwidth=args.strict_bandwidth,
        sync_u2=args.sync_u2,
        seed=args.seed,
    )
    y_bw = np.zeros(total_slots)
    decision_log = []

    print(
        f"开始真实轨迹调度: V={num_views}, T={total_slots}, "
        f"B_avg={args.b_avg}, V_lya={args.v_lya}, eps_mask={args.eps_mask}, "
        f"min_active_views={args.min_active_views}, "
        f"S_adapter={args.adapter_size_mb}, SCL_weights={args.u2_update_size_mb}"
    )
    print(f"网络模式: {args.network_mode}, slot={args.slot_duration}s, "
          f"deadline={args.deadline_ms}ms, strict_bw={args.strict_bandwidth}, sync_u2={args.sync_u2}")
    print(f"输入轨迹: {args.input}")
    print("-" * 72)

    # --- 4. 主循环 ---
    for t, row in enumerate(trajectory_rows):
        # 4.1 采样网络状态（R_t, B_t, net_state, is_disconnected）
        net_state = net.step()
        is_disc = bool(net_state["is_disconnected"])

        e_drift = float(row["E_drift"])
        struct_drift = float(row.get("struct_drift", 0.0) or 0.0)
        w_t = np.array([float(row[f"w_{i + 1}"]) for i in range(num_views)], dtype=float)

        # 时隙开始的 Y_bw（net.step 不改 Y_bw，net.update_queues 才改）
        y_bw[t] = net.y_bw

        # State = [环境漂移, 结构性漂移, 归一化虚拟带宽队列]
        state_t = np.array([e_drift, struct_drift, y_bw[t] / 100.0], dtype=float)

        # 4.2 Actor 生成候选
        candidates = mem.decode_and_quantize(state_t, w_t, M_t=args.m_candidates)

        # 4.3 网络过滤：断联→强制 u=0；--strict-bandwidth→硬过滤超带宽
        filtered_result = net.filter_candidates(candidates, c_comm_map)
        feasible = filtered_result["candidates"]

        # 4.4 Critic 评估 + 超带宽软罚 G
        best_g = -np.inf
        best_action = None
        best_details = None
        candidate_scores = {}
        if not feasible:
            # 极端情况（断联且 Actor 未生成 u=0 候选）兜底：本地自治
            v_fallback = np.zeros(num_views, dtype=int)
            v_fallback[0] = 1
            feasible = [(v_fallback, 0)]

        for v_cand, u_cand in feasible:
            g_raw, k_t, c_comm, acc, cost = WaterFilling_Critic(
                v_cand, u_cand, w_t, e_drift, struct_drift, y_bw[t], args.v_lya, sys_params
            )
            # 超带宽软罚：G_effective = G_raw - overflow_penalty * (comm_overflow / B_t)
            # realtime_comm 对 u=1/u=2-async 为 0（异步后台），仅 u=2 --sync-u2 时非 0
            realtime_comm = net.realtime_comm_mb(u_cand, c_comm)
            g_effective = net.apply_network_penalty(g_raw, realtime_comm)["G_effective"]

            score = candidate_scores.get(int(u_cand))
            if score is None or g_effective > score["G_effective"]:
                candidate_scores[int(u_cand)] = {
                    "G_raw": float(g_raw), "G_effective": float(g_effective)
                }

            if g_effective > best_g:
                best_g = g_effective
                best_action = (v_cand, u_cand, k_t)
                best_details = (c_comm, acc, cost)

        v_opt, u_opt, k_opt = best_action
        c_comm_opt, acc_opt, cost_opt = best_details

        # 4.5 端到端时延（u=0 特判 T_comm=0 不加 RTT；由 compute_e2e 内部处理）
        e2e_info = net.compute_e2e(
            u_opt, net.realtime_comm_mb(u_opt, c_comm_opt), t_edge=args.edge_delay_ms
        )

        # 4.6 业务可用四条件（防"断联切本地=100%可用"虚高）
        decision_success = (u_opt == 0) or (not is_disc)
        business_available = net.is_business_available(
            decision_success=decision_success,
            e2e_ms=e2e_info["e2e_delay_ms"],
            active_views=int(np.sum(v_opt)),
            proxy_acc=acc_opt,
            transmission_success=e2e_info["transmission_success"],
        )

        # 4.7 更新 Y_bw（Lyapunov 虚拟队列）+ Q_net（物理积压，带上限/TTL）
        queue_result = net.update_queues(c_comm=c_comm_opt, b_avg=args.b_avg, u=u_opt)
        q_net = queue_result["Q_net"]

        mem.encode(state_t, v_opt, u_opt)

        decision_log.append(
            build_log_row(
                row, t, y_bw[t], w_t, struct_drift, v_opt, u_opt, k_opt,
                c_comm_opt, acc_opt, cost_opt, best_g, net_state, e2e_info,
                business_available, q_net, queue_result, candidate_scores,
            )
        )

        if t % 100 == 0 or t == total_slots - 1:
            k_str = "[" + ", ".join(f"{k:.2f}" for k in k_opt) + "]"
            print(
                f"时隙 {t:>4}: Y={y_bw[t]:>7.2f} Q={q_net:>6.2f} | v={v_opt} | u={u_opt} | "
                f"k={k_str} | net={net_state['network_state']:>7} "
                f"e2e={e2e_info['e2e_delay_ms']:>6.1f}ms | biz={int(business_available)}"
            )

    save_decision_log(args.output, decision_log, num_views)
    print_summary(decision_log, y_bw)
    print(f"决策日志已保存: {args.output}")


if __name__ == "__main__":
    main()
