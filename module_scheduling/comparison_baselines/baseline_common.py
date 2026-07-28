import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


BASELINE_DIR = Path(__file__).resolve().parent
MV_VIT_DIR = BASELINE_DIR.parent
FORMAL_DIR = MV_VIT_DIR / "formal_experiments"

for path in [str(FORMAL_DIR), str(MV_VIT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)


def add_common_args(parser):
    parser.add_argument("--input", required=True, help="Trajectory CSV.")
    parser.add_argument("--output", required=True, help="Output policy log CSV.")
    parser.add_argument("--summary-output", default="", help="Output summary JSON.")
    parser.add_argument("--num-views", type=int, default=None)
    parser.add_argument("--b-avg", type=float, default=8.0)
    parser.add_argument("--v-lya", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tokens-per-view", type=float, default=196.0)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--k-min", type=float, default=0.10)
    parser.add_argument("--beta-0", type=float, default=1.0)
    parser.add_argument("--alpha-env", type=float, default=0.4)
    parser.add_argument("--alpha-struct", type=float, default=0.3)
    parser.add_argument("--retrain-bonus", type=float, default=0.2)
    parser.add_argument("--tau-retrain", type=float, default=0.5)
    parser.add_argument("--s-prompt", type=float, default=2.0)
    parser.add_argument("--s-query", type=float, default=0.1)
    parser.add_argument("--scl-weights", type=float, default=20.0)
    return parser


def infer_num_views_from_row(row):
    view_cols = [key for key in row.keys() if key.startswith("w_")]
    if not view_cols:
        raise ValueError("Trajectory row has no w_i columns.")
    return max(int(key.split("_")[1]) for key in view_cols)


def load_trajectory(path, num_views=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Trajectory file is empty: {path}")

    if num_views is None:
        num_views = infer_num_views_from_row(rows[0])

    required = ["E_drift", *[f"w_{idx + 1}" for idx in range(num_views)]]
    missing = [col for col in required if col not in rows[0]]
    if missing:
        raise ValueError(f"Trajectory file is missing required columns: {missing}")
    return rows, num_views


def float_or_empty(row, key):
    value = row.get(key, "")
    return "" if value == "" else float(value)


def int_or_empty(row, key):
    value = row.get(key, "")
    return "" if value == "" else int(float(value))


def make_sys_params(args):
    return {
        "N": args.tokens_per_view,
        "eta": args.eta,
        "gamma": args.gamma,
        "k_min": args.k_min,
        "beta_0": args.beta_0,
        "S_prompt": args.s_prompt,
        "S_query": args.s_query,
        "SCL_weights": args.scl_weights,
        "alpha_env": args.alpha_env,
        "alpha_struct": args.alpha_struct,
        "retrain_bonus": args.retrain_bonus,
        "tau_retrain": args.tau_retrain,
    }


def comm_cost(u_t, sys_params):
    if int(u_t) == 1:
        return sys_params["S_prompt"] + sys_params["S_query"]
    if int(u_t) == 2:
        return sys_params["SCL_weights"] + sys_params["S_query"]
    return 0.0


def beta_actual(u_t, e_drift, struct_drift, sys_params):
    beta_0 = sys_params["beta_0"]
    alpha_env = sys_params.get("alpha_env", 0.4)
    alpha_struct = sys_params.get("alpha_struct", 0.3)
    retrain_bonus = sys_params.get("retrain_bonus", 0.2)
    tau_retrain = sys_params.get("tau_retrain", 0.5)

    if int(u_t) == 0:
        beta = beta_0 - alpha_env * e_drift - alpha_struct * struct_drift
    elif int(u_t) == 1:
        beta = beta_0 - alpha_struct * struct_drift
    else:
        beta = beta_0 + retrain_bonus * max(struct_drift - tau_retrain, 0.0)
    return max(0.01, beta)


def evaluate_fixed_k_action(v_t, u_t, k_t, w_t, e_drift, struct_drift, y_bw, v_lya, sys_params):
    v_t = np.asarray(v_t, dtype=int)
    k_t = np.asarray(k_t, dtype=float)
    w_t = np.asarray(w_t, dtype=float)
    active = np.where(v_t == 1)[0]
    c_comm = comm_cost(u_t, sys_params)
    beta = beta_actual(u_t, e_drift, struct_drift, sys_params)

    if len(active) == 0:
        acc = beta
        cost = 0.0
    else:
        gamma = sys_params["gamma"]
        eta = sys_params["eta"]
        n_tokens = sys_params["N"]
        acc = beta + sum(w_t[idx] * np.log(1.0 + gamma * k_t[idx]) for idx in active)
        cost = eta * (sum(k_t[idx] * n_tokens for idx in active)) ** 2

    utility = acc - cost
    g_value = v_lya * utility - y_bw * c_comm
    return g_value, k_t, c_comm, acc, cost, utility


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(len(weights), 1.0 / max(len(weights), 1), dtype=float)
    return weights / total


def top_indices(weights, count):
    weights = np.asarray(weights, dtype=float)
    count = int(max(0, min(count, len(weights))))
    if count == 0:
        return np.asarray([], dtype=int)
    return np.argsort(-weights, kind="stable")[:count]


def proportional_keep(v_t, weights, keep_sum, min_keep=0.10, max_keep=1.0):
    v_t = np.asarray(v_t, dtype=int)
    weights = normalize_weights(weights)
    k_t = np.zeros(len(v_t), dtype=float)
    active = np.where(v_t == 1)[0]
    if len(active) == 0:
        return k_t

    keep_sum = float(np.clip(keep_sum, min_keep * len(active), max_keep * len(active)))
    active_weights = normalize_weights(weights[active])
    k_active = active_weights * keep_sum
    k_active = np.clip(k_active, min_keep, max_keep)

    # Redistribute residual budget after clipping. This keeps the baseline simple
    # while avoiding impossible token ratios.
    for _ in range(4):
        residual = keep_sum - float(np.sum(k_active))
        if abs(residual) < 1e-8:
            break
        if residual > 0:
            free = np.where(k_active < max_keep - 1e-8)[0]
        else:
            free = np.where(k_active > min_keep + 1e-8)[0]
        if len(free) == 0:
            break
        free_weights = normalize_weights(active_weights[free])
        k_active[free] += residual * free_weights
        k_active = np.clip(k_active, min_keep, max_keep)

    k_t[active] = k_active
    return k_t


def semantic_blind_keep(v_t, weights, keep_sum, min_keep=0.10, max_keep=1.0, weak_keep=0.45):
    v_t = np.asarray(v_t, dtype=int)
    weights = normalize_weights(weights)
    k_t = np.zeros(len(v_t), dtype=float)
    active = np.where(v_t == 1)[0]
    if len(active) == 0:
        return k_t

    weak_idx = active[int(np.argmax(weights[active]))]
    weak_keep = float(np.clip(weak_keep, min_keep, max_keep))
    keep_sum = float(np.clip(keep_sum, min_keep * len(active), max_keep * len(active)))
    keep_sum = min(keep_sum, weak_keep + max_keep * max(len(active) - 1, 0))
    k_t[weak_idx] = weak_keep

    rest = np.asarray([idx for idx in active if idx != weak_idx], dtype=int)
    if len(rest) == 0:
        return k_t

    rest_budget = max(0.0, keep_sum - weak_keep)
    rest_weights = normalize_weights(1.0 - weights[rest] + 1e-6)
    k_rest = np.clip(rest_budget * rest_weights, min_keep, max_keep)

    for _ in range(4):
        residual = rest_budget - float(np.sum(k_rest))
        if abs(residual) < 1e-8:
            break
        if residual > 0:
            free = np.where(k_rest < max_keep - 1e-8)[0]
        else:
            free = np.where(k_rest > min_keep + 1e-8)[0]
        if len(free) == 0:
            break
        free_weights = normalize_weights(rest_weights[free])
        k_rest[free] += residual * free_weights
        k_rest = np.clip(k_rest, min_keep, max_keep)

    k_t[rest] = k_rest
    return k_t


def row_weights(row, num_views):
    return normalize_weights([float(row[f"w_{idx + 1}"]) for idx in range(num_views)])


def row_struct_drift(row):
    return float(row.get("struct_drift", 0.0) or 0.0)


def drift_key(row):
    return row.get("stage_name") or row.get("drift_type") or "unknown"


def make_action(v_t, u_t, k_t, w_real, w_decision, e_drift, struct_drift, y_bw, args, sys_params):
    decision = evaluate_fixed_k_action(
        v_t,
        u_t,
        k_t,
        w_decision,
        e_drift,
        struct_drift,
        y_bw,
        args.v_lya,
        sys_params,
    )
    realized = evaluate_fixed_k_action(
        v_t,
        u_t,
        k_t,
        w_real,
        e_drift,
        struct_drift,
        y_bw,
        args.v_lya,
        sys_params,
    )
    return {
        "v": np.asarray(v_t, dtype=int),
        "u": int(u_t),
        "k": np.asarray(k_t, dtype=float),
        "decision_G": float(decision[0]),
        "G": float(realized[0]),
        "c_comm": float(realized[2]),
        "proxy_acc": float(realized[3]),
        "cost": float(realized[4]),
        "U": float(realized[5]),
    }


def build_log_row(row, t, y_bw, w_real, w_decision, action, method, struct_drift, tokens_per_view):
    num_views = len(w_real)
    log_row = {
        "method": method,
        "t": t,
        "sample_id": int_or_empty(row, "sample_id"),
        "label": int_or_empty(row, "label"),
        "pred": int_or_empty(row, "pred"),
        "confidence": float_or_empty(row, "confidence"),
        "E_drift": float(row["E_drift"]),
        "drift_type": row.get("drift_type", ""),
        "severity": float_or_empty(row, "severity"),
        "struct_drift": float(struct_drift),
        "u": int(action["u"]),
        "c_comm": float(action["c_comm"]),
        "proxy_acc": float(action["proxy_acc"]),
        "cost": float(action["cost"]),
        "U": float(action["U"]),
        "G": float(action["G"]),
        "decision_G": float(action["decision_G"]),
        "Y_bw": float(y_bw),
        "active_views": int(np.sum(action["v"])),
        "active_token_ratio_sum": float(np.sum(action["k"])),
        "token_count": float(np.sum(action["k"]) * tokens_per_view),
        "decision_time_ms": float(action.get("decision_time_ms", 0.0)),
    }
    for key in ["stage_id", "stage_name"]:
        if key in row:
            log_row[key] = row.get(key, "")
    for idx in range(num_views):
        log_row[f"w_{idx + 1}"] = float(w_real[idx])
        log_row[f"w_decision_{idx + 1}"] = float(w_decision[idx])
        log_row[f"v_{idx + 1}"] = int(action["v"][idx])
        log_row[f"k_{idx + 1}"] = float(action["k"][idx])
    return log_row


def write_decision_log(path, rows, num_views):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "t",
        "sample_id",
        "label",
        "pred",
        "confidence",
        "E_drift",
        "drift_type",
        "severity",
        "struct_drift",
        *[f"w_{idx + 1}" for idx in range(num_views)],
        *[f"w_decision_{idx + 1}" for idx in range(num_views)],
        *[f"v_{idx + 1}" for idx in range(num_views)],
        "u",
        *[f"k_{idx + 1}" for idx in range(num_views)],
        "c_comm",
        "proxy_acc",
        "cost",
        "U",
        "G",
        "decision_G",
        "Y_bw",
        "active_views",
        "active_token_ratio_sum",
        "token_count",
        "decision_time_ms",
    ]
    extra_fields = []
    for key in ["stage_id", "stage_name"]:
        if rows and key in rows[0]:
            extra_fields.append(key)
    fieldnames = fieldnames[:10] + extra_fields + fieldnames[10:]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_log(rows, final_y):
    u_counter = Counter(int(row["u"]) for row in rows)
    total = max(len(rows), 1)
    summary = {
        "total_slots": len(rows),
        "avg_U": float(np.mean([row["U"] for row in rows])),
        "avg_G": float(np.mean([row["G"] for row in rows])),
        "avg_proxy_acc": float(np.mean([row["proxy_acc"] for row in rows])),
        "avg_cost": float(np.mean([row["cost"] for row in rows])),
        "avg_comm": float(np.mean([row["c_comm"] for row in rows])),
        "avg_queue": float(np.mean([row["Y_bw"] for row in rows])),
        "final_queue": float(final_y),
        "avg_active_views": float(np.mean([row["active_views"] for row in rows])),
        "avg_token_sum": float(np.mean([row["active_token_ratio_sum"] for row in rows])),
        "avg_token_count": float(np.mean([row["token_count"] for row in rows])),
        "avg_decision_time_ms": float(np.mean([row.get("decision_time_ms", 0.0) for row in rows])),
    }
    for u_t in [0, 1, 2]:
        summary[f"u{u_t}_count"] = int(u_counter.get(u_t, 0))
        summary[f"u{u_t}_ratio"] = float(u_counter.get(u_t, 0) / total)
    return summary


def write_summary(path, summary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def simulate_baseline(rows, num_views, method, args, decision_fn):
    rng = np.random.default_rng(args.seed)
    sys_params = make_sys_params(args)
    y_bw = np.zeros(len(rows), dtype=float)
    decision_log = []
    state = {"rng": rng, "cache": set(), "history": Counter()}

    for t, row in enumerate(rows):
        e_drift = float(row["E_drift"])
        struct_drift = row_struct_drift(row)
        w_real = row_weights(row, num_views)
        start_time = time.perf_counter()
        action = decision_fn(
            t=t,
            row=row,
            w_real=w_real,
            e_drift=e_drift,
            struct_drift=struct_drift,
            y_bw=float(y_bw[t]),
            num_views=num_views,
            args=args,
            sys_params=sys_params,
            state=state,
        )
        action["decision_time_ms"] = (time.perf_counter() - start_time) * 1000.0

        if t < len(rows) - 1:
            y_bw[t + 1] = max(y_bw[t] + action["c_comm"] - args.b_avg, 0.0)

        decision_log.append(
            build_log_row(
                row,
                t,
                float(y_bw[t]),
                w_real,
                action.get("w_decision", w_real),
                action,
                method,
                struct_drift,
                args.tokens_per_view,
            )
        )

    summary = summarize_log(decision_log, y_bw[-1] if len(y_bw) else 0.0)
    summary["method"] = method
    summary["num_views"] = num_views
    summary["b_avg"] = args.b_avg
    summary["v_lya"] = args.v_lya
    return decision_log, summary


def run_baseline(method, decision_fn, extra_args_fn=None):
    parser = argparse.ArgumentParser(description=f"Run the {method} comparison baseline.")
    add_common_args(parser)
    if extra_args_fn is not None:
        extra_args_fn(parser)
    args = parser.parse_args()

    rows, num_views = load_trajectory(args.input, args.num_views)
    args.num_views = num_views
    decision_log, summary = simulate_baseline(rows, num_views, method, args, decision_fn)

    output = Path(args.output)
    summary_output = Path(args.summary_output) if args.summary_output else output.with_suffix(".summary.json")
    write_decision_log(output, decision_log, num_views)
    write_summary(summary_output, summary)

    print("-" * 72)
    print(f"{method} baseline finished: slots={summary['total_slots']}")
    print(f"Decision log: {output}")
    print(f"Summary: {summary_output}")
    print(f"avg_proxy_acc={summary['avg_proxy_acc']:.6f}")
    print(f"avg_comm={summary['avg_comm']:.4f} MB/slot, avg_queue={summary['avg_queue']:.4f}")
    print(f"avg_active_views={summary['avg_active_views']:.4f}/{num_views}")
    print(f"avg_token_count={summary['avg_token_count']:.2f}")
    print(f"avg_decision_time_ms={summary['avg_decision_time_ms']:.6f}")
    print(
        "u ratio: "
        f"u0={summary['u0_ratio']:.2%}, "
        f"u1={summary['u1_ratio']:.2%}, "
        f"u2={summary['u2_ratio']:.2%}"
    )
