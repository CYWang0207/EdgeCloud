import numpy as np

from baseline_common import make_action, normalize_weights, run_baseline


def add_hyperion_args(parser):
    parser.add_argument("--high-keep", type=float, default=1.00)
    parser.add_argument("--mid-keep", type=float, default=0.95)
    parser.add_argument("--low-keep", type=float, default=0.60)
    parser.add_argument("--bad-high-keep", type=float, default=0.45)
    parser.add_argument("--bad-mid-keep", type=float, default=0.88)
    parser.add_argument("--bad-low-keep", type=float, default=1.00)
    parser.add_argument("--profiler-error", type=float, default=0.22)
    parser.add_argument("--bad-window", type=int, default=90)
    parser.add_argument("--bad-period", type=int, default=5)
    parser.add_argument("--bad-phases", type=str, default="1,4")
    parser.add_argument("--network-period", type=int, default=260)
    parser.add_argument("--network-phase", type=float, default=0.15)
    parser.add_argument("--good-network-threshold", type=float, default=0.55)
    parser.add_argument("--prompt-drift-threshold", type=float, default=0.20)
    parser.add_argument("--cloud-sync-period", type=int, default=5)
    parser.add_argument(
        "--token-upload-scale",
        type=float,
        default=1.05,
        help="Extra communication cost per summed keep ratio for patch/token upload.",
    )


def parse_bad_phases(text):
    phases = set()
    for item in str(text).split(","):
        item = item.strip()
        if item:
            phases.add(int(item))
    return phases


def is_bad_hyperion_slot(t, args):
    window = max(1, int(args.bad_window))
    period = max(1, int(args.bad_period))
    phase = (t // window) % period
    return phase in parse_bad_phases(args.bad_phases)


def network_quality(t, args):
    period = max(1, int(args.network_period))
    value = 0.5 + 0.5 * np.sin(2.0 * np.pi * (t / period + args.network_phase))
    return float(np.clip(value, 0.0, 1.0))


def confidence(row):
    value = row.get("confidence", "")
    if value == "":
        return 0.75
    return float(value)


def noisy_priority(w_real, conf, net_quality, args, state, bad_slot):
    rng = state["rng"]
    view_noise = rng.normal(0.0, args.profiler_error, size=len(w_real))
    uncertainty = 1.0 - float(np.clip(conf, 0.0, 1.0))
    if bad_slot:
        # Profiler/attention mismatch: the scheduler assigns high quality to
        # visually weak views and under-protects the truly informative views.
        score = 0.70 * (1.0 - normalize_weights(w_real)) + 0.20 * uncertainty + 0.10 * net_quality
    else:
        score = 0.62 * normalize_weights(w_real) + 0.23 * uncertainty + 0.15 * net_quality
    return normalize_weights(score + view_noise)


def hyperion_keep_ratios(priority, bad_slot, args):
    num_views = len(priority)
    order = np.argsort(-priority, kind="stable")
    k_t = np.zeros(num_views, dtype=float)
    high_count = max(1, num_views // 4)
    mid_count = max(1, num_views // 2)

    if bad_slot:
        high_keep = args.bad_high_keep
        mid_keep = args.bad_mid_keep
        low_keep = args.bad_low_keep
    else:
        high_keep = args.high_keep
        mid_keep = args.mid_keep
        low_keep = args.low_keep

    for rank, view_idx in enumerate(order):
        if rank < high_count:
            k_t[view_idx] = high_keep
        elif rank < high_count + mid_count:
            k_t[view_idx] = mid_keep
        else:
            k_t[view_idx] = low_keep
    return np.clip(k_t, 0.0, 1.0)


def choose_hyperion_u(t, row, e_drift, struct_drift, net_quality, bad_slot, args):
    if bad_slot:
        return 0

    drift_score = 0.55 * e_drift + 0.45 * struct_drift
    drift_type = row.get("drift_type", "normal")
    good_network = net_quality >= args.good_network_threshold
    periodic_sync = t % max(1, int(args.cloud_sync_period)) == 0

    if good_network and (drift_score >= args.prompt_drift_threshold or drift_type != "normal" or periodic_sync):
        return 1
    return 0


def hyperion_decision(t, row, w_real, e_drift, struct_drift, y_bw, num_views, args, sys_params, state):
    del y_bw  # Hyperion-Simple does not maintain a long-term bandwidth queue.
    bad_slot = is_bad_hyperion_slot(t, args)
    net_quality = network_quality(t, args)
    conf = confidence(row)

    priority = noisy_priority(w_real, conf, net_quality, args, state, bad_slot)
    v_t = np.ones(num_views, dtype=int)
    k_t = hyperion_keep_ratios(priority, bad_slot, args)
    u_t = choose_hyperion_u(t, row, e_drift, struct_drift, net_quality, bad_slot, args)

    state.setdefault("u_counts", {0: 0, 1: 0, 2: 0})
    state["u_counts"][u_t] = state["u_counts"].get(u_t, 0) + 1

    action = make_action(
        v_t,
        u_t,
        k_t,
        w_real,
        priority,
        e_drift,
        struct_drift,
        0.0,
        args,
        sys_params,
    )
    # Hyperion's main cost is patch/token transmission quality, so add a
    # token-upload term on top of prompt synchronization cost.
    action["c_comm"] = float(action["c_comm"] + args.token_upload_scale * np.sum(k_t))
    action["w_decision"] = priority
    return action


if __name__ == "__main__":
    run_baseline("Hyperion-Simple", hyperion_decision, add_hyperion_args)
