"""Plot benchmark projection solver evolution curves.

Loads the .npy metric files saved by ``benchmark_projection.py --save`` and
plots objective, recovery error, and constraint metrics as a function of
iterations or wall time for each solver.

Example
-------
python plot/plot_benchmark_projection.py \\
    --log_dir logs/benchmark_projection/dim20_noise1e-01_rbf_seeds1-5 \\
    --x_axis iteration \\
    --metrics objectives,recovery_errors,max_off \\
    --n_runs 5

Use ``--no_tex`` on machines without a LaTeX installation.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times",
    "font.size": 15,
    "legend.fontsize": 10,
    "figure.autolayout": True,
})

SOLVERS = ["ipm", "activeset", "admm"]
COLORS = {"ipm": "C0", "activeset": "C1", "admm": "C2"}
MARKERS = {"ipm": "o", "activeset": "s", "admm": "^"}
LABELS = {"ipm": "IPM", "activeset": "Active Set", "admm": "ADMM"}

METRIC_LABELS = {
    "objectives": r"$\frac{1}{2}\Vert \hat{\mathbf{L}} - \bar{\mathbf{L}} \Vert_F^2$",
    "recovery_errors": r"$\Vert \hat{\mathbf{L}} - \mathbf{L} \Vert_F / \Vert \mathbf{L} \Vert_F$",
    "precision": r"Precision (\%)",
    "recall": r"Recall (\%)",
    "f1_score": r"F-score (\%)",
    "nae": r"NAE $= \Vert W - W_0 \Vert_F / \Vert W_0 \Vert_F$",
    "row_sum_max_abs": r"$\max_i \vert \sum_j L_{ij} \vert$",
    "trace_err": r"$\vert \mathrm{Tr}(L) - \mathrm{vol} \vert$",
    "max_off": r"$\max_{i \neq j} L_{ij}$",
    "sp": r"NNZ",
}


def load_evolution(log_dir: str, solver: str, run_idx: int):
    """Load one run's evolution arrays, if present."""
    evo_dir = os.path.join(log_dir, solver, f"run{run_idx}")
    if not os.path.isdir(evo_dir):
        return None
    evo = {}
    for key in [
        "iterations",
        "times",
        "objectives",
        "recovery_errors",
        "row_sum_max_abs",
        "trace_err",
        "max_off",
        "precision",
        "recall",
        "f1_score",
        "nae",
        "sp",
    ]:
        path = os.path.join(evo_dir, f"{key}.npy")
        if os.path.exists(path):
            evo[key] = np.load(path)
    return evo if evo else None


def average_evolutions(log_dir: str, solver: str, n_runs: int, x_axis: str,
                        n_grid: int = 200):
    """Average evolution arrays across runs by interpolating onto a common grid."""
    evos = []
    for run_idx in range(n_runs):
        evo = load_evolution(log_dir, solver, run_idx)
        if evo is not None:
            evos.append(evo)
    if not evos:
        return None

    x_key = "iterations" if x_axis == "iteration" else "times"
    x_min = min(float(e[x_key].min()) for e in evos)
    x_max = max(float(e[x_key].max()) for e in evos)
    if x_max <= x_min:
        return None
    x_grid = np.linspace(x_min, x_max, n_grid)

    averaged = {x_key: x_grid}
    metric_keys = [
        "objectives",
        "recovery_errors",
        "row_sum_max_abs",
        "trace_err",
        "max_off",
        "precision",
        "recall",
        "f1_score",
        "nae",
        "sp",
    ]
    for key in metric_keys:
        if key not in evos[0]:
            continue
        interpolated = []
        for evo in evos:
            x = evo[x_key]
            y = evo[key]
            # Ensure x is strictly increasing for np.interp.
            if len(x) == 1:
                interpolated.append(np.full_like(x_grid, y[0]))
                continue
            idx = np.argsort(x)
            x_sorted = x[idx]
            y_sorted = y[idx]
            interpolated.append(np.interp(x_grid, x_sorted, y_sorted))
        arr = np.vstack(interpolated)
        averaged[key] = arr.mean(axis=0)
        averaged[f"{key}_std"] = arr.std(axis=0)
    return averaged


def _is_device_token(tok: str) -> bool:
    return tok == "cpu" or tok == "cuda" or (tok.startswith("cuda")
                                               and tok[4:].isdigit())


def _resolve_device_dirs(log_dir: str):
    """Resolve {(device, log_dir)} for CPU and CUDA.

    The counterpart directory is derived from ``log_dir`` by swapping the
    device token (``cpu``/``cuda0``) in the folder name, as produced by
    ``benchmark_projection.py --save``.
    """
    norm = os.path.normpath(log_dir)
    tokens = os.path.basename(norm).split("_")
    idx = [i for i, t in enumerate(tokens) if _is_device_token(t)]
    if len(idx) != 1:
        raise SystemExit(
            f"Cannot determine device from log dir name '{os.path.basename(norm)}' "
            "(expected exactly one 'cpu' or 'cuda0' token)."
        )
    i = idx[0]
    given_dev = "cpu" if tokens[i] == "cpu" else "cuda"
    parent = os.path.dirname(norm)

    def build(dev_tok: str) -> str:
        t = list(tokens)
        t[i] = dev_tok
        return os.path.join(parent, "_".join(t))

    dirs = {given_dev: norm}
    dirs["cpu" if given_dev == "cuda" else "cuda"] = build(
        "cpu" if given_dev == "cuda" else "cuda0"
    )
    ordered = []
    for dev in ("cuda", "cpu"):
        if os.path.isdir(dirs[dev]):
            ordered.append((dev, dirs[dev]))
        else:
            print(f"Warning: {dev} log dir not found: {dirs[dev]}")
    if not ordered:
        raise SystemExit("No device log dirs found.")
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot projection solver evolution curves"
    )
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Path to the benchmark_projection log directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (also saved as .svg; default: <log_dir>/evolution_<x_axis>.png)")
    parser.add_argument("--x_axis", type=str, default="iteration",
                        choices=["iteration", "time"],
                        help="Plot against solver iteration or wall time")
    parser.add_argument("--metrics", type=str,
                        default="objectives,recovery_errors,f1_score",
                        help="Comma-separated evolution metric names to plot")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of runs to average over")
    parser.add_argument("--no_tex", action="store_true",
                        help="Disable LaTeX (usetex) rendering and use "
                             "matplotlib's built-in mathtext instead")
    parser.add_argument("--compare_devices", action="store_true",
                        help="Plot CPU (dashed) and CUDA (solid) curves in the "
                             "same figure; the counterpart log dir is derived "
                             "from --log_dir by swapping the device token")
    parser.add_argument("--compare_modes", action="store_true",
                        help="Plot optimized (solid), naive (dashed) and "
                             "minres (dotted) curves together; expects "
                             "<solver>_<mode>/ subdirs as saved by "
                             "benchmark_projection.py --modes "
                             "naive,optimized,minres")
    parser.add_argument("--max_x", type=float, default=None,
                        help="Cap the x-axis (iteration or time, per --x_axis) "
                             "at this value, e.g. to zoom in on the region "
                             "where fast solvers converge instead of being "
                             "squashed by a slow outlier's full range")
    args = parser.parse_args()

    mode_list = ["optimized", "naive", "minres"] if args.compare_modes else [None]

    if args.no_tex:
        plt.rcParams["text.usetex"] = False

    if args.compare_devices:
        device_dirs = _resolve_device_dirs(args.log_dir)
    else:
        device_dirs = [(None, args.log_dir)]

    if args.output is None:
        args.output = os.path.join(args.log_dir, f"evolution_{args.x_axis}.png")

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metric_names:
        raise ValueError("No metrics selected")

    n_metrics = len(metric_names)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(8, 3 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    x_label = "Iteration" if args.x_axis == "iteration" else "Time (s)"
    x_max_seen = 0.0

    for dev_idx, (dev, log_dir) in enumerate(device_dirs):
        for mode_idx, mode in enumerate(mode_list):
            if mode is not None:
                linestyle = {"optimized": "-", "naive": "--", "minres": ":"}[mode]
            else:
                linestyle = "--" if dev == "cpu" else "-"
            # When combining --compare_devices with --compare_modes,
            # linestyle is already spoken for by mode, so device is encoded
            # as line width instead (CUDA thick, CPU thin) rather than
            # silently overlapping CPU/CUDA with no visual distinction.
            if args.compare_devices and args.compare_modes:
                linewidth = 1.0 if dev == "cpu" else 2.2
            else:
                linewidth = 1.5
            for solver in SOLVERS:
                sub = f"{solver}_{mode}" if mode is not None else solver
                evo = average_evolutions(log_dir, sub, args.n_runs, args.x_axis)
                if evo is None:
                    print(f"Warning: no evolution data for {sub} in {log_dir}, skipping.")
                    continue

                x = evo["iterations"] if args.x_axis == "iteration" else evo["times"]
                if len(x):
                    x_max_seen = max(x_max_seen, float(np.max(x)))

                for ax, metric in zip(axes, metric_names):
                    if metric not in evo:
                        print(f"Warning: {metric} not found for {solver}, skipping.")
                        continue
                    y = evo[metric]
                    markevery = max(1, len(x) // 8)
                    # Label only the first (device, mode) pass so the method
                    # legend has one entry per solver; device/mode are decoded
                    # by line style.
                    label = LABELS[solver] if (dev_idx == 0 and mode_idx == 0) else None
                    ax.plot(
                        x, y,
                        label=label,
                        color=COLORS[solver],
                        marker=MARKERS[solver],
                        markevery=markevery,
                        linewidth=linewidth,
                        markersize=8,
                        linestyle=linestyle,
                    )
                    std_key = f"{metric}_std"
                    if std_key in evo:
                        ax.fill_between(
                            x,
                            y - evo[std_key],
                            y + evo[std_key],
                            alpha=0.2,
                            color=COLORS[solver],
                        )
                    ax.set_ylabel(METRIC_LABELS.get(metric, metric.replace("_", " ").title()))
                    ax.grid(True, linestyle="--", alpha=0.6)
                    if metric in ("f1_score", "precision", "recall"):
                        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, symbol=""))
                    if metric in ("objectives", "recovery_errors", "nae") and np.min(y) > 0:
                        ax.set_yscale("log")

    # Iteration/time always starts at 0; letting matplotlib autoscale the
    # x-axis produces a nonsensical margin when a few slow runs (e.g. a
    # solver that never converges and hits --max_time) stretch x_max far
    # beyond where most solvers actually finish. Pin the range explicitly
    # instead, with a plain 3% margin on each side.
    x_right = min(x_max_seen, args.max_x) if args.max_x is not None else x_max_seen
    x_pad = 0.03 * x_right if x_right > 0 else 1.0
    for ax in axes:
        ax.legend(loc="best", fontsize="small")
        ax.set_xlim(-x_pad, x_right + x_pad)
    if args.compare_modes and args.compare_devices:
        style_handles = [
            mlines.Line2D([], [], color="black", linestyle="-", label="Optimized"),
            mlines.Line2D([], [], color="black", linestyle="--", label="Naive"),
            mlines.Line2D([], [], color="black", linestyle=":", label="MINRES"),
            mlines.Line2D([], [], color="black", linewidth=2.2, label="CUDA"),
            mlines.Line2D([], [], color="black", linewidth=1.0, label="CPU"),
        ]
        fig.legend(handles=style_handles, loc="upper right", ncol=5,
                   fontsize="small", bbox_to_anchor=(0.99, 0.96))
    elif args.compare_modes:
        style_handles = [
            mlines.Line2D([], [], color="black", linestyle="-", label="Optimized"),
            mlines.Line2D([], [], color="black", linestyle="--", label="Naive"),
            mlines.Line2D([], [], color="black", linestyle=":", label="MINRES"),
        ]
        fig.legend(handles=style_handles, loc="upper right", ncol=3,
                   fontsize="small", bbox_to_anchor=(0.99, 0.96))
    elif args.compare_devices:
        style_handles = [
            mlines.Line2D([], [], color="black", linestyle="-", label="CUDA"),
            mlines.Line2D([], [], color="black", linestyle="--", label="CPU"),
        ]
        fig.legend(handles=style_handles, loc="upper right", ncol=2,
                   fontsize="small", bbox_to_anchor=(0.99, 0.96))
    axes[-1].set_xlabel(x_label)
    title = os.path.basename(os.path.normpath(args.log_dir))
    if args.compare_devices:
        title = "_".join("cpu+cuda" if _is_device_token(t) else t
                         for t in title.split("_"))
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(args.output, dpi=150)
    print(f"Saved evolution plot to {args.output}")
    svg_output = os.path.splitext(args.output)[0] + ".svg"
    plt.savefig(svg_output)
    print(f"Saved evolution plot to {svg_output}")


if __name__ == "__main__":
    main()
