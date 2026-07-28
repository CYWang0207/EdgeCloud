import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


COLORS = ["#263b5e", "#0073bd", "#86a9c1", "#a6d9f5", "#4d6f8c", "#b7cee2"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--"]


def parse_named_path(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use label=path format.")
    label, path = text.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("Use non-empty label=path format.")
    return label, Path(path)


def parse_named_column(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use label=column format.")
    label, column = text.split("=", 1)
    label = label.strip()
    column = column.strip()
    if not label or not column:
        raise argparse.ArgumentTypeError("Use non-empty label=column format.")
    return label, column


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot line charts from real evaluation logs, using the tu.py visual style."
    )
    parser.add_argument("--log", action="append", type=parse_named_path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument(
        "--x-col",
        default="",
        help="Use a CSV column as x axis. When omitted, the script uses t/index and time-series smoothing.",
    )
    parser.add_argument(
        "--x-tick-col",
        default="",
        help="Optional second column shown under x tick labels, e.g. m_candidates.",
    )
    parser.add_argument(
        "--series",
        action="append",
        type=parse_named_column,
        default=[],
        help="Summary mode only: label=column, e.g. Mean=mean_latency_ms.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        default="block",
        choices=["raw", "moving", "block", "cumulative_average"],
    )
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--percent", action="store_true")
    parser.add_argument("--y-scale", type=float, default=1.0)
    parser.add_argument("--xlabel", default="Time (slot)")
    parser.add_argument("--ylabel", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--ylim-low", type=float, default=None)
    parser.add_argument("--ylim-high", type=float, default=None)
    parser.add_argument("--fig-width", type=float, default=10.0)
    parser.add_argument("--fig-height", type=float, default=7.0)
    parser.add_argument("--legend-cols", type=int, default=1)
    parser.add_argument("--marker-every", type=int, default=1)
    parser.add_argument("--line-width", type=float, default=3.6)
    parser.add_argument("--marker-size", type=float, default=8.0)
    parser.add_argument("--font-weight", default="bold")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_series(path, metric):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    if metric not in rows[0]:
        raise ValueError(f"Metric '{metric}' not found in {path}")

    x = np.asarray(
        [float(row.get("t", idx)) if row.get("t", "") != "" else float(idx) for idx, row in enumerate(rows)],
        dtype=float,
    )
    y = np.asarray([float(row[metric]) if row.get(metric, "") != "" else 0.0 for row in rows], dtype=float)
    return x, y


def read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def read_xy(path, x_col, y_col):
    rows = read_rows(path)
    missing = [col for col in [x_col, y_col] if col not in rows[0]]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    rows.sort(key=lambda row: float(row[x_col]))
    x = np.asarray([float(row[x_col]) for row in rows], dtype=float)
    y = np.asarray([float(row[y_col]) for row in rows], dtype=float)
    return x, y, rows


def make_xtick_labels(rows, x_col, x_tick_col):
    if not rows or not x_tick_col:
        return None, None
    if x_tick_col not in rows[0]:
        raise ValueError(f"x tick column '{x_tick_col}' not found.")
    positions = [float(row[x_col]) for row in rows]
    labels = []
    for row in rows:
        x_value = int(float(row[x_col])) if float(row[x_col]).is_integer() else row[x_col]
        extra_value = int(float(row[x_tick_col])) if float(row[x_tick_col]).is_integer() else row[x_tick_col]
        labels.append(f"{x_value}\nM={extra_value}")
    return positions, labels


def smooth_series(x, y, mode, window):
    if mode == "raw":
        return x, y

    window = max(1, int(window))
    if mode == "moving":
        if window <= 1:
            return x, y
        kernel = np.ones(window, dtype=float) / window
        y_smooth = np.convolve(y, kernel, mode="valid")
        return x[window - 1 :], y_smooth

    if mode == "block":
        xs = []
        ys = []
        for start in range(0, len(y), window):
            end = min(start + window, len(y))
            xs.append(float(np.mean(x[start:end])))
            ys.append(float(np.mean(y[start:end])))
        return np.asarray(xs), np.asarray(ys)

    if mode == "cumulative_average":
        denom = np.arange(1, len(y) + 1, dtype=float)
        return x, np.cumsum(y) / denom

    raise ValueError(f"Unknown mode: {mode}")


def default_ylabel(metric, percent):
    labels = {
        "policy_correct": "Accuracy",
        "full_correct": "Accuracy",
        "c_comm": "Communication Cost (MB/slot)",
        "real_token_count": "Token Count (tokens)",
        "token_count": "Token Count (tokens)",
        "decision_time_ms": "Algorithm Execution Time (ms)",
        "mean_latency_ms": "Decision Latency (ms)",
        "p50_latency_ms": "Decision Latency (ms)",
        "p90_latency_ms": "Decision Latency (ms)",
        "p99_latency_ms": "Decision Latency (ms)",
        "Y_bw": "Queue Length (MB)",
    }
    label = labels.get(metric, metric)
    if percent and "%" not in label:
        return f"{label} (%)"
    return label


def draw_line_chart(series, args):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["font.weight"] = args.font_weight
    plt.rcParams["axes.labelweight"] = args.font_weight
    plt.rcParams["axes.titleweight"] = args.font_weight

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    marker_every = max(1, int(args.marker_every))

    for idx, (label, x, y) in enumerate(series):
        ax.plot(
            x,
            y,
            label=label,
            color=COLORS[idx % len(COLORS)],
            linestyle=LINESTYLES[idx % len(LINESTYLES)],
            linewidth=args.line_width,
            marker=MARKERS[idx % len(MARKERS)],
            markersize=args.marker_size,
            markevery=marker_every,
            markerfacecolor=COLORS[idx % len(COLORS)],
            markeredgecolor="white",
            markeredgewidth=1.1,
            zorder=3,
        )

    ax.set_xlabel(args.xlabel, fontsize=22, fontweight=args.font_weight)
    ax.set_ylabel(args.ylabel or default_ylabel(args.metric, args.percent), fontsize=22, fontweight=args.font_weight)
    if args.title:
        ax.set_title(args.title, fontsize=22, fontweight=args.font_weight)

    ax.tick_params(axis="x", labelsize=17, width=1.6)
    ax.tick_params(axis="y", labelsize=17, width=1.6)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(args.font_weight)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    if args.ylim_low is not None or args.ylim_high is not None:
        ax.set_ylim(args.ylim_low, args.ylim_high)
    if hasattr(args, "xtick_positions") and args.xtick_positions is not None:
        ax.set_xticks(args.xtick_positions)
        ax.set_xticklabels(args.xtick_labels)
        for tick in ax.get_xticklabels():
            tick.set_fontweight(args.font_weight)

    legend = ax.legend(fontsize=17, ncols=max(1, args.legend_cols), frameon=True)
    for text in legend.get_texts():
        text.set_fontweight(args.font_weight)
    ax.grid(True, zorder=0, linestyle="-", linewidth=0.8, alpha=0.55)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=args.dpi)
    plt.close()
    print(f"Saved figure: {output}")


def main():
    args = parse_args()
    series = []

    if args.x_col:
        if args.series:
            if len(args.log) != 1:
                raise ValueError("When --series is used, pass exactly one --log label=path.")
            _log_label, path = args.log[0]
            first_rows = None
            for label, y_col in args.series:
                x_plot, y_plot, rows = read_xy(path, args.x_col, y_col)
                first_rows = rows if first_rows is None else first_rows
                y_plot = y_plot * args.y_scale
                if args.percent:
                    y_plot = y_plot * 100.0
                series.append((label, x_plot, y_plot))
            args.xtick_positions, args.xtick_labels = make_xtick_labels(first_rows, args.x_col, args.x_tick_col)
            if not args.ylabel:
                args.ylabel = default_ylabel(args.series[0][1], args.percent)
        else:
            first_rows = None
            for label, path in args.log:
                x_plot, y_plot, rows = read_xy(path, args.x_col, args.metric)
                first_rows = rows if first_rows is None else first_rows
                y_plot = y_plot * args.y_scale
                if args.percent:
                    y_plot = y_plot * 100.0
                series.append((label, x_plot, y_plot))
            args.xtick_positions, args.xtick_labels = make_xtick_labels(first_rows, args.x_col, args.x_tick_col)
    else:
        for label, path in args.log:
            x, y = read_series(path, args.metric)
            x_plot, y_plot = smooth_series(x, y, args.mode, args.window)
            y_plot = y_plot * args.y_scale
            if args.percent:
                y_plot = y_plot * 100.0
            series.append((label, x_plot, y_plot))

    draw_line_chart(series, args)


if __name__ == "__main__":
    main()
