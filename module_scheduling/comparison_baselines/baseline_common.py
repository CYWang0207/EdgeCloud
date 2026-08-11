import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


BASELINE_DIR = Path(__file__).resolve().parent
MV_VIT_DIR = BASELINE_DIR.parent
FORMAL_DIR = MV_VIT_DIR / "formal_experiments"
EDGE_CLOUD_RL_DIR = MV_VIT_DIR / "EdgeCloud_RL"

for path in [str(FORMAL_DIR), str(MV_VIT_DIR), str(EDGE_CLOUD_RL_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from network_sim import DEFAULT_ACC_FLOOR, NetworkSimulator  # noqa: E402


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
    parser.add_argument("--s-adapter", type=float, default=1.2,
                        help="adapter 参数下发带宽 (MB)，u=1 通信开销口径")
    parser.add_argument("--s-query", type=float, default=0.1)
    parser.add_argument("--scl-weights", type=float, default=20.0)

    # --- 网络韧性参数（与 main_edge_cloud_new.py 对齐，让基线跑四档出 e2e/业务保持率）---
    parser.add_argument(
        "--network-mode",
        default="static",
        choices=("static", "jitter", "jitter_outage", "markov", "all"),
        help="static=基线对照；jitter=带宽抖动；jitter_outage=抖动+断联；markov=GOOD/WEAK/DOWN；all=四档全跑出对比表",
    )
    parser.add_argument("--slot-duration", type=float, default=0.2, help="单时隙时长(s)，B_t=R_t*slot/8")
    parser.add_argument("--edge-delay-ms", type=float, default=80.0, help="T_edge 边缘推理延迟")
    parser.add_argument("--rtt-ms", type=float, default=10.0, help="往返时延，T_comm 计入")
    parser.add_argument("--deadline-ms", type=float, default=200.0, help="端到端时延硬指标上限(≤0.2s)")
    parser.add_argument(
        "--acc-floor",
        type=float,
        default=DEFAULT_ACC_FLOOR,
        help=f"业务可用 proxy_acc 下限，默认 {DEFAULT_ACC_FLOOR}（与主方法共享）",
    )
    parser.add_argument("--business-min-active-views", type=int, default=1, help="业务可用最小激活视角数")
    parser.add_argument("--bw-min-mbps", type=float, default=20.0)
    parser.add_argument("--bw-max-mbps", type=float, default=120.0)
    parser.add_argument("--disconnect-prob", type=float, default=0.0, help="jitter_outage 随机断联概率")
    parser.add_argument("--outage-period", type=int, default=0, help="周期断联周期(时隙)，0=不周期断联")
    parser.add_argument("--outage-duration", type=int, default=0, help="周期断联持续(时隙)")
    parser.add_argument("--strict-bandwidth", action="store_true", help="超带宽直接过滤候选（默认仅软罚）")
    parser.add_argument("--sync-u2", action="store_true", help="u=2 实时同步（默认异步后台入 Q_net）")
    parser.add_argument(
        "--force-local",
        action="store_true",
        help="启用'断联→强制 u=0'保护（这是主方法 filter_candidates 的网络感知能力，基线默认不具备）。"
        "默认关闭：基线在断联时仍按自身 u 决策（u=1/2 会触发 decision_success=False），"
        "业务保持率在 jitter_outage/markov 下显著下降，体现基线网络无感的弱点。加此旗标可做消融对照。",
    )
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
        "S_adapter": args.s_adapter,
        "S_query": args.s_query,
        "SCL_weights": args.scl_weights,
        "alpha_env": args.alpha_env,
        "alpha_struct": args.alpha_struct,
        "retrain_bonus": args.retrain_bonus,
        "tau_retrain": args.tau_retrain,
    }


def build_network_simulator(args, mode, sys_params):
    """按 args 构造 NetworkSimulator，通信体积口径与 baseline 自身 comm_cost 一致。

    baseline 的 comm_cost(u) 用 S_adapter/S_query/SCL_weights；这里把同一组值喂给
    NetworkSimulator 的 adapter_size_mb/query_size_mb/u2_update_size_mb，确保
    decision_fn 算的 c_comm 与 network_sim 内部 comm_cost_mb 同口径。
    """
    return NetworkSimulator(
        mode=mode,
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
        adapter_size_mb=args.s_adapter,
        query_size_mb=args.s_query,
        u2_update_size_mb=args.scl_weights,
        strict_bandwidth=args.strict_bandwidth,
        sync_u2=args.sync_u2,
        seed=args.seed,
    )


def comm_cost(u_t, sys_params):
    if int(u_t) == 1:
        return sys_params["S_adapter"] + sys_params["S_query"]
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


def build_log_row(
    row,
    t,
    y_bw,
    w_real,
    w_decision,
    action,
    method,
    struct_drift,
    tokens_per_view,
    net_state=None,
    e2e_info=None,
    business_available=None,
    q_net=None,
    forced_local=False,
    decision_success=None,
):
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
    # 网络韧性字段（接 network_sim 后填充；旧调用方不传则留空，保持向后兼容）
    if net_state is not None:
        log_row["network_state"] = net_state.get("network_state", "")
        log_row["bandwidth_mbps"] = float(net_state.get("bandwidth_mbps", 0.0))
        log_row["loss_rate"] = float(net_state.get("loss_rate", 0.0))
        log_row["B_t"] = float(net_state.get("B_t", 0.0))
        log_row["is_disconnected"] = int(bool(net_state.get("is_disconnected", False)))
    else:
        log_row["network_state"] = ""
        log_row["bandwidth_mbps"] = ""
        log_row["loss_rate"] = ""
        log_row["B_t"] = ""
        log_row["is_disconnected"] = ""
    if e2e_info is not None:
        log_row["realtime_comm"] = float(e2e_info.get("realtime_comm", 0.0))
        log_row["comm_delay_ms"] = float(e2e_info.get("comm_delay_ms", 0.0))
        log_row["cloud_delay_ms"] = float(e2e_info.get("cloud_delay_ms", 0.0))
        log_row["e2e_delay_ms"] = float(e2e_info.get("e2e_delay_ms", 0.0))
        log_row["deadline_ms"] = float(e2e_info.get("deadline_ms", 0.0))
        log_row["deadline_met"] = int(bool(e2e_info.get("deadline_met", False)))
        log_row["transmission_success"] = int(bool(e2e_info.get("transmission_success", False)))
    else:
        for key in ["realtime_comm", "comm_delay_ms", "cloud_delay_ms", "e2e_delay_ms", "deadline_ms"]:
            log_row[key] = ""
        log_row["deadline_met"] = ""
        log_row["transmission_success"] = ""
    log_row["decision_success"] = "" if decision_success is None else int(bool(decision_success))
    log_row["forced_local"] = int(bool(forced_local))
    log_row["Q_net"] = "" if q_net is None else float(q_net)
    log_row["business_available"] = "" if business_available is None else int(bool(business_available))
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
        "network_state",
        "bandwidth_mbps",
        "loss_rate",
        "B_t",
        "is_disconnected",
        "realtime_comm",
        "comm_delay_ms",
        "cloud_delay_ms",
        "e2e_delay_ms",
        "deadline_ms",
        "deadline_met",
        "transmission_success",
        "decision_success",
        "forced_local",
        "Q_net",
        "business_available",
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

    def _num(row, key):
        val = row.get(key, "")
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _mean(key):
        vals = [v for v in (_num(r, key) for r in rows) if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    def _p95(key):
        vals = [v for v in (_num(r, key) for r in rows) if v is not None]
        return float(np.percentile(vals, 95)) if vals else float("nan")

    def _rate(key):
        vals = []
        for r in rows:
            v = r.get(key, "")
            if v == "" or v is None:
                continue
            vals.append(int(bool(v)))
        return float(np.mean(vals)) if vals else float("nan")

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
        # 网络韧性指标（接 network_sim 后新增；旧日志无对应列则记 NaN）
        "business_continuity_rate": _rate("business_available"),
        "avg_e2e_delay_ms": _mean("e2e_delay_ms"),
        "p95_e2e_delay_ms": _p95("e2e_delay_ms"),
        "deadline_met_rate": _rate("deadline_met"),
        "transmission_success_rate": _rate("transmission_success"),
        "decision_success_rate": _rate("decision_success"),
        "disconnect_rate": _rate("is_disconnected"),
        "forced_local_count": int(sum(int(bool(r.get("forced_local", 0))) for r in rows)),
        "avg_Q_net": _mean("Q_net"),
        "avg_bandwidth_mbps": _mean("bandwidth_mbps"),
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


def simulate_baseline(rows, num_views, method, args, decision_fn, network_mode=None):
    """跑一遍基线调度。接 NetworkSimulator 后每时隙：
    step → decision_fn → (断联强制 u=0) → compute_e2e → is_business_available → update_queues。

    network_mode 可覆盖 args.network_mode（供 --network-mode all 四档循环复用）。
    """
    rng = np.random.default_rng(args.seed)
    sys_params = make_sys_params(args)
    mode = network_mode or getattr(args, "network_mode", "static")
    net = build_network_simulator(args, mode=mode, sys_params=sys_params)
    force_local_enabled = bool(getattr(args, "force_local", False))

    decision_log = []
    state = {"rng": rng, "cache": set(), "history": Counter()}

    for t, row in enumerate(rows):
        # 1. 采样网络状态（R_t, B_t, net_state, is_disconnected）
        net_state = net.step()
        is_disc = bool(net_state["is_disconnected"])

        e_drift = float(row["E_drift"])
        struct_drift = row_struct_drift(row)
        w_real = row_weights(row, num_views)

        # 时隙开始 Y_bw（net.step 不改 Y_bw，net.update_queues 才改）
        y_bw_t = float(net.y_bw)

        # 把网络状态暴露给 decision_fn（基线可选用，多数基线网络无感、忽略即可）
        state["net_state"] = net_state
        state["is_disconnected"] = is_disc

        # 2. 基线自身决策（v_t, u_t, k_t）
        start_time = time.perf_counter()
        action = decision_fn(
            t=t,
            row=row,
            w_real=w_real,
            e_drift=e_drift,
            struct_drift=struct_drift,
            y_bw=y_bw_t,
            num_views=num_views,
            args=args,
            sys_params=sys_params,
            state=state,
        )
        action["decision_time_ms"] = (time.perf_counter() - start_time) * 1000.0

        # 3. 断联→强制 u=0（系统级保护，对齐 main_edge_cloud_new.filter_candidates）
        forced_local = False
        if is_disc and int(action["u"]) != 0 and force_local_enabled:
            v_t = action["v"]
            k_t = action["k"]
            w_decision = action.get("w_decision", w_real)
            action = make_action(
                v_t,
                0,
                k_t,
                w_real,
                w_decision,
                e_drift,
                struct_drift,
                y_bw_t,
                args,
                sys_params,
            )
            action["w_decision"] = w_decision
            action["decision_time_ms"] = (time.perf_counter() - start_time) * 1000.0
            forced_local = True

        u_final = int(action["u"])
        c_comm = float(action["c_comm"])
        realtime_comm = net.realtime_comm_mb(u_final, c_comm)

        # 4. 端到端时延（u=0/异步 u=1,u=2 → realtime_comm=0 → T_comm=0）
        e2e_info = net.compute_e2e(u_final, realtime_comm, t_edge=args.edge_delay_ms)

        # 5. 业务可用四条件
        decision_success = (u_final == 0) or (not is_disc)
        business_available = net.is_business_available(
            decision_success=decision_success,
            e2e_ms=e2e_info["e2e_delay_ms"],
            active_views=int(np.sum(action["v"])),
            proxy_acc=float(action["proxy_acc"]),
            transmission_success=e2e_info["transmission_success"],
        )

        # 6. 更新 Y_bw（Lyapunov 虚拟队列）+ Q_net（物理积压）
        queue_result = net.update_queues(c_comm=c_comm, b_avg=args.b_avg, u=u_final)
        q_net = queue_result["Q_net"]

        decision_log.append(
            build_log_row(
                row,
                t,
                y_bw_t,
                w_real,
                action.get("w_decision", w_real),
                action,
                method,
                struct_drift,
                args.tokens_per_view,
                net_state=net_state,
                e2e_info=e2e_info,
                business_available=business_available,
                q_net=q_net,
                forced_local=forced_local,
                decision_success=decision_success,
            )
        )

    summary = summarize_log(decision_log, net.y_bw)
    summary["method"] = method
    summary["num_views"] = num_views
    summary["b_avg"] = args.b_avg
    summary["v_lya"] = args.v_lya
    summary["network_mode"] = mode
    return decision_log, summary


NETWORK_MODES = ("static", "jitter", "jitter_outage", "markov")


def _print_summary_block(method, summary, num_views, output, summary_output):
    print("-" * 72)
    print(f"{method} baseline finished: slots={summary['total_slots']}")
    print(f"Decision log: {output}")
    print(f"Summary: {summary_output}")
    print(f"avg_proxy_acc={summary['avg_proxy_acc']:.6f}")
    print(f"avg_comm={summary['avg_comm']:.4f} MB/slot, avg_queue={summary['avg_queue']:.4f}")
    print(f"avg_active_views={summary['avg_active_views']:.4f}/{num_views}")
    print(f"avg_token_count={summary['avg_token_count']:.2f}")
    print(f"avg_decision_time_ms={summary['avg_decision_time_ms']:.6f}")
    biz = summary.get("business_continuity_rate", float("nan"))
    e2e = summary.get("avg_e2e_delay_ms", float("nan"))
    p95 = summary.get("p95_e2e_delay_ms", float("nan"))
    print(
        f"business_continuity={biz:.2%}, avg_e2e={e2e:.1f}ms, p95_e2e={p95:.1f}ms, "
        f"disconnect={summary.get('disconnect_rate', float('nan')):.2%}"
    )
    print(
        "u ratio: "
        f"u0={summary['u0_ratio']:.2%}, "
        f"u1={summary['u1_ratio']:.2%}, "
        f"u2={summary['u2_ratio']:.2%}"
    )


def _fmt(v, spec=".4f"):
    """格式化数值，NaN/None → 'N/A'。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "N/A"
    if math.isnan(f):
        return "N/A"
    return format(f, spec)


def _print_comparison_table(method, modes_summary):
    """打印 method 在四档网络下的对比表（业务保持率/e2e/通信/token）。"""
    print("=" * 78)
    print(f"[{method}] 四档网络对比表")
    print("-" * 78)
    header = f"{'mode':<14}{'业务保持率':>12}{'avg_e2e(ms)':>14}{'p95_e2e(ms)':>14}{'通信(MB/slot)':>16}{'token_count':>14}"
    print(header)
    print("-" * 78)
    for mode in NETWORK_MODES:
        s = modes_summary.get(mode, {})
        row = (
            f"{mode:<14}"
            f"{_fmt(s.get('business_continuity_rate'), '.2%'):>12}"
            f"{_fmt(s.get('avg_e2e_delay_ms'), '.1f'):>14}"
            f"{_fmt(s.get('p95_e2e_delay_ms'), '.1f'):>14}"
            f"{_fmt(s.get('avg_comm'), '.4f'):>16}"
            f"{_fmt(s.get('avg_token_count'), '.1f'):>14}"
        )
        print(row)
    print("=" * 78)


def run_baseline(method, decision_fn, extra_args_fn=None):
    parser = argparse.ArgumentParser(description=f"Run the {method} comparison baseline.")
    add_common_args(parser)
    if extra_args_fn is not None:
        extra_args_fn(parser)
    args = parser.parse_args()

    rows, num_views = load_trajectory(args.input, args.num_views)
    args.num_views = num_views

    output = Path(args.output)
    summary_output = (
        Path(args.summary_output) if args.summary_output else output.with_suffix(".summary.json")
    )

    if args.network_mode != "all":
        decision_log, summary = simulate_baseline(rows, num_views, method, args, decision_fn)
        write_decision_log(output, decision_log, num_views)
        write_summary(summary_output, summary)
        _print_summary_block(method, summary, num_views, output, summary_output)
        return

    # --- --network-mode all：四档循环 + 汇总对比表 ---
    modes_summary = {}
    for mode in NETWORK_MODES:
        decision_log, summary = simulate_baseline(
            rows, num_views, method, args, decision_fn, network_mode=mode
        )
        modes_summary[mode] = summary

        mode_output = output.with_name(f"{output.stem}_{mode}{output.suffix}")
        mode_summary = summary_output.with_name(f"{summary_output.stem}_{mode}{summary_output.suffix}")
        write_decision_log(mode_output, decision_log, num_views)
        write_summary(mode_summary, summary)
        print(f"[{mode}] ", end="")
        _print_summary_block(method, summary, num_views, mode_output, mode_summary)

    combined = {
        "method": method,
        "num_views": num_views,
        "b_avg": args.b_avg,
        "v_lya": args.v_lya,
        "modes": modes_summary,
        "comparison_table": [
            {
                "mode": mode,
                "business_continuity_rate": modes_summary[mode].get("business_continuity_rate"),
                "avg_e2e_delay_ms": modes_summary[mode].get("avg_e2e_delay_ms"),
                "p95_e2e_delay_ms": modes_summary[mode].get("p95_e2e_delay_ms"),
                "avg_comm": modes_summary[mode].get("avg_comm"),
                "avg_token_count": modes_summary[mode].get("avg_token_count"),
                "avg_queue": modes_summary[mode].get("avg_queue"),
                "avg_decision_time_ms": modes_summary[mode].get("avg_decision_time_ms"),
                "u0_ratio": modes_summary[mode].get("u0_ratio"),
                "u1_ratio": modes_summary[mode].get("u1_ratio"),
                "u2_ratio": modes_summary[mode].get("u2_ratio"),
            }
            for mode in NETWORK_MODES
        ],
    }
    write_summary(summary_output, combined)
    print(f"\nCombined summary: {summary_output}")
    _print_comparison_table(method, modes_summary)
