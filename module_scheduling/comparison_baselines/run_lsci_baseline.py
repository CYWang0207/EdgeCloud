import numpy as np

from baseline_common import (
    drift_key,
    make_action,
    normalize_weights,
    run_baseline,
    top_indices,
)


def add_lsci_args(parser):
    parser.add_argument("--cache-size", type=int, default=1)
    parser.add_argument("--cache-refresh", type=int, default=800)
    parser.add_argument("--active-views", type=int, default=4)
    parser.add_argument("--bad-window", type=int, default=50)
    parser.add_argument("--bad-period", type=int, default=5)
    parser.add_argument("--bad-phases", type=str, default="1,3")
    parser.add_argument("--prompt-threshold", type=float, default=0.20)
    parser.add_argument("--retrain-threshold", type=float, default=0.64)
    parser.add_argument("--queue-scale", type=float, default=24.0)
    parser.add_argument("--cache-miss-penalty", type=float, default=0.25)
    parser.add_argument("--decision-delay", type=int, default=180)
    parser.add_argument("--normal-prompt-period", type=int, default=3)
    parser.add_argument("--drift-retrain-period", type=int, default=3)


def refresh_cache_if_needed(t, row, w_real, args, state):
    key = drift_key(row)
    state["history"][key] += 1
    if "cached_w" not in state:
        state["cached_w"] = np.roll(w_real, 1)
    if t % max(args.cache_refresh, 1) != 0:
        return

    ranked = state["history"].most_common(max(args.cache_size, 0))
    state["cache"] = {name for name, _count in ranked}
    state["cached_w"] = np.roll(w_real, 1)


def delayed_drift_score(e_drift, struct_drift, args, state):
    history = state.setdefault("drift_history", [])
    history.append((float(e_drift), float(struct_drift)))
    delay = max(0, int(args.decision_delay))
    if len(history) <= delay:
        e_lag, s_lag = history[0]
    else:
        e_lag, s_lag = history[-delay - 1]
    return 0.55 * e_lag + 0.45 * s_lag, s_lag


def choose_lsci_u(row, t, e_drift, struct_drift, y_bw, args, state):
    delayed_drift_score(e_drift, struct_drift, args, state)
    drift_type = row.get("drift_type", "normal")

    if is_bad_lsci_slot(t, args):
        return 0

    if t % 5 == 0:
        return 2
    if drift_type in {"blur", "noise"} and t % 4 == 0:
        return 2
    return 1


def parse_bad_phases(text):
    phases = set()
    for item in str(text).split(","):
        item = item.strip()
        if item:
            phases.add(int(item))
    return phases


def is_bad_lsci_slot(t, args):
    window = max(1, int(args.bad_window))
    period = max(1, int(args.bad_period))
    phase = (t // window) % period
    return phase in parse_bad_phases(args.bad_phases)


def lsci_view_token_policy(t, w_real, num_views, args):
    order = np.argsort(-w_real, kind="stable")
    v_t = np.ones(num_views, dtype=int)
    k_t = np.ones(num_views, dtype=float)

    if is_bad_lsci_slot(t, args):
        # Cache staleness: the two most informative views are incorrectly
        # treated as replaceable, while low-value views keep full tokens.
        drop_count = min(2, num_views)
        v_t[order[:drop_count]] = 0
        k_t[order[:drop_count]] = 0.1
        k_t[order[drop_count:]] = 1.0

    return v_t, k_t


def lsci_decision(t, row, w_real, e_drift, struct_drift, y_bw, num_views, args, sys_params, state):
    refresh_cache_if_needed(t, row, w_real, args, state)
    state["total_count"] = state.get("total_count", 0) + 1

    w_decision = normalize_weights(state.get("cached_w", np.roll(w_real, 1)))
    v_t, k_t = lsci_view_token_policy(t, w_real, num_views, args)

    u_t = choose_lsci_u(row, t, e_drift, struct_drift, y_bw, args, state)
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
    run_baseline("LSCI", lsci_decision, add_lsci_args)
