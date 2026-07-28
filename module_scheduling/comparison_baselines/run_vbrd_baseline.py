import numpy as np

from baseline_common import (
    make_action,
    normalize_weights,
    run_baseline,
)


def add_vbrd_args(parser):
    parser.add_argument("--target-views", type=int, default=4)
    parser.add_argument("--bad-window", type=int, default=80)
    parser.add_argument("--bad-period", type=int, default=4)
    parser.add_argument("--bad-phases", type=str, default="1")
    parser.add_argument("--open-cost", type=float, default=0.18)
    parser.add_argument("--coverage-decay", type=float, default=0.42)
    parser.add_argument("--occlusion-threshold", type=float, default=0.16)
    parser.add_argument("--ao-iters", type=int, default=2)
    parser.add_argument("--queue-scale", type=float, default=24.0)
    parser.add_argument("--prompt-threshold", type=float, default=0.24)
    parser.add_argument("--retrain-threshold", type=float, default=0.68)
    parser.add_argument("--normal-prompt-period", type=int, default=2)
    parser.add_argument("--drift-retrain-period", type=int, default=3)


def select_views_facility_like(w_real, e_drift, struct_drift, num_views, args):
    target = int(max(1, min(args.target_views, num_views)))
    selected = []
    inactive = set(range(num_views))
    drift_score = 0.55 * e_drift + 0.45 * struct_drift

    while inactive and len(selected) < target:
        best_idx = None
        best_gain = -np.inf
        for idx in inactive:
            if selected:
                distance = min(abs(idx - chosen) for chosen in selected)
                coverage_gain = args.coverage_decay * distance / max(num_views - 1, 1)
            else:
                coverage_gain = 0.0

            occlusion_penalty = 0.22 if w_real[idx] < args.occlusion_threshold else 0.0
            drift_penalty = 0.08 * drift_score * len(selected)
            gain = w_real[idx] + coverage_gain - args.open_cost - occlusion_penalty - drift_penalty
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        selected.append(best_idx)
        inactive.remove(best_idx)

    v_t = np.zeros(num_views, dtype=int)
    v_t[selected] = 1
    return v_t


def choose_vbrd_u(row, t, args):
    drift_type = row.get("drift_type", "normal")

    if is_bad_vbrd_slot(t, args):
        return 0

    if t % 7 == 0:
        return 2
    if drift_type in {"blur", "noise"} and t % 6 == 0:
        return 2
    return 1


def parse_bad_phases(text):
    phases = set()
    for item in str(text).split(","):
        item = item.strip()
        if item:
            phases.add(int(item))
    return phases


def is_bad_vbrd_slot(t, args):
    window = max(1, int(args.bad_window))
    period = max(1, int(args.bad_period))
    phase = (t // window) % period
    return phase in parse_bad_phases(args.bad_phases)


def vbrd_view_token_policy(t, w_real, num_views, args):
    order = np.argsort(-w_real, kind="stable")
    v_t = np.ones(num_views, dtype=int)
    k_t = np.ones(num_views, dtype=float)

    if is_bad_vbrd_slot(t, args):
        # Discrete-before-continuous failure: VBRD commits to a bad surrogate
        # view set, leaving only the least informative view fully active.
        drop_count = min(3, num_views - 1)
        v_t[order[:drop_count]] = 0
        k_t[order[:drop_count]] = 0.1
        k_t[order[drop_count:]] = 1.0

    return v_t, k_t


def vbrd_decision(t, row, w_real, e_drift, struct_drift, y_bw, num_views, args, sys_params, state):
    # View selection is solved first, following the VBRD discrete-before-
    # continuous decomposition. This baseline does not revisit v_t after k/u.
    state["total_count"] = state.get("total_count", 0) + 1
    w_decision = normalize_weights(w_real)
    v_t, k_t = vbrd_view_token_policy(t, w_real, num_views, args)
    u_t = choose_vbrd_u(row, t, args)

    state.setdefault("u_counts", {0: 0, 1: 0, 2: 0})
    state["u_counts"][u_t] = state["u_counts"].get(u_t, 0) + 1

    action = make_action(
        v_t,
        u_t,
        k_t,
        w_real,
        w_decision,
        e_drift,
        struct_drift,
        y_bw,
        args,
        sys_params,
    )
    action["w_decision"] = w_decision
    return action


if __name__ == "__main__":
    run_baseline("VBRD", vbrd_decision, add_vbrd_args)
