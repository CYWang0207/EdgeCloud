import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run edge-cloud RL scheduling on measured trajectory data."
    )
    parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "real_trajectory_data.csv")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "rl_decision_log.csv")
    parser.add_argument("--b-avg", type=float, default=3.0, help="Average bandwidth budget per slot.")
    parser.add_argument("--v-lya", type=float, default=20.0, help="Lyapunov utility weight.")
    parser.add_argument("--eps-mask", type=float, default=0.05, help="Mask views whose w_i is below this value.")
    parser.add_argument("--min-active-views", type=float, default=3.0, help="Minimum active views per slot. Use 3 to guarantee average >= 2.5.")
    parser.add_argument("--m-candidates", type=int, default=10, help="Actor candidate count before u cross product.")
    parser.add_argument("--scl-weights", type=float, default=50.0, help="Bandwidth cost of retrained model weights.")
    parser.add_argument("--alpha-env", type=float, default=0.4, help="Local accuracy penalty weight for E_drift.")
    parser.add_argument("--alpha-struct", type=float, default=0.3, help="Accuracy penalty weight for structural drift.")
    parser.add_argument("--retrain-bonus", type=float, default=0.2, help="Extra gain from retraining under severe structural drift.")
    parser.add_argument("--tau-retrain", type=float, default=0.5, help="Structural drift threshold for retraining gain.")
    parser.add_argument("--seed", type=int, default=42)
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


def build_log_row(row, t, y_bw, w_t, struct_drift, v_opt, u_opt, k_opt, c_comm, acc, cost, best_g):
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
        "active_views": int(np.sum(v_opt)),
        "active_token_ratio_sum": float(np.sum(k_opt)),
    }

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
        "active_views",
        "active_token_ratio_sum",
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

    print("-" * 72)
    print("仿真结束")
    print(f"总时隙数: {total}")
    print(f"最终带宽队列 Y: {y_bw[-1]:.4f}")
    print(f"平均单步目标 G: {avg_g:.4f}")
    print(f"平均通信开销: {avg_comm:.4f} MB/slot")
    print(f"平均激活视角数: {avg_active_views:.4f}")
    print(f"平均 Token 保留率总和: {avg_token_sum:.4f}")
    for u in [0, 1, 2]:
        count = u_counter.get(u, 0)
        print(f"u={u} 次数: {count} ({count / total:.2%})")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    from actor_memory import CollaborativeMemoryDNN
    from critic_water_filling import WaterFilling_Critic

    # --- 1. 系统参数初始化 ---
    num_views = 4
    sys_params = {
        "N": 196,
        "eta": 5e-5,
        "gamma": 10.0,
        "k_min": 0.1,
        "beta_0": 0.2,
        "S_adapter": 1.2,  # adapter 参数下发带宽消耗 (MB)，u=1 通信开销口径
        "SCL_weights": args.scl_weights,
        "S_query": 5.0,
        "alpha_env": args.alpha_env,
        "alpha_struct": args.alpha_struct,
        "retrain_bonus": args.retrain_bonus,
        "tau_retrain": args.tau_retrain,
    }

    # --- 2. 加载真实轨迹数据 ---
    trajectory_rows = load_trajectory(args.input, num_views)
    total_slots = len(trajectory_rows)

    # --- 3. 初始化 Actor-Critic 记忆网络与虚拟队列 ---
    mem = CollaborativeMemoryDNN(
        V=num_views,
        state_dim=3,
        eps_mask=args.eps_mask,
        min_active_views=args.min_active_views,
    )
    y_bw = np.zeros(total_slots)
    decision_log = []

    print(
        f"开始真实轨迹调度: V={num_views}, T={total_slots}, "
        f"B_avg={args.b_avg}, V_lya={args.v_lya}, eps_mask={args.eps_mask}, "
        f"min_active_views={args.min_active_views}, SCL_weights={args.scl_weights}"
    )
    print(f"输入轨迹: {args.input}")
    print("-" * 72)

    # --- 4. 主循环 ---
    for t, row in enumerate(trajectory_rows):
        e_drift = float(row["E_drift"])
        struct_drift = float(row.get("struct_drift", 0.0) or 0.0)
        w_t = np.array([float(row[f"w_{i + 1}"]) for i in range(num_views)], dtype=float)

        # State = [环境漂移, 结构性漂移, 归一化虚拟带宽队列]
        state_t = np.array([e_drift, struct_drift, y_bw[t] / 100.0], dtype=float)

        candidates = mem.decode_and_quantize(state_t, w_t, M_t=args.m_candidates)

        best_g = -np.inf
        best_action = None
        best_details = None

        for v_cand, u_cand in candidates:
            g_value, k_t, c_comm, acc, cost = WaterFilling_Critic(
                v_cand, u_cand, w_t, e_drift, struct_drift, y_bw[t], args.v_lya, sys_params
            )

            if g_value > best_g:
                best_g = g_value
                best_action = (v_cand, u_cand, k_t)
                best_details = (c_comm, acc, cost)

        v_opt, u_opt, k_opt = best_action
        c_comm_opt, acc_opt, cost_opt = best_details

        if t < total_slots - 1:
            y_bw[t + 1] = max(y_bw[t] + c_comm_opt - args.b_avg, 0.0)

        mem.encode(state_t, v_opt, u_opt)

        decision_log.append(
            build_log_row(
                row,
                t,
                y_bw[t],
                w_t,
                struct_drift,
                v_opt,
                u_opt,
                k_opt,
                c_comm_opt,
                acc_opt,
                cost_opt,
                best_g,
            )
        )

        if t % 100 == 0 or t == total_slots - 1:
            k_str = "[" + ", ".join(f"{k:.2f}" for k in k_opt) + "]"
            print(
                f"时隙 {t:>4}: Y={y_bw[t]:>7.2f} | v={v_opt} | u={u_opt} | "
                f"k={k_str} | E_drift={e_drift:.4f} | S_drift={struct_drift:.2f}"
            )

    save_decision_log(args.output, decision_log, num_views)
    print_summary(decision_log, y_bw)
    print(f"决策日志已保存: {args.output}")


if __name__ == "__main__":
    main()
