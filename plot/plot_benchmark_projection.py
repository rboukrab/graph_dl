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
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
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
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.log_dir, f"evolution_{args.x_axis}.png")

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metric_names:
        raise ValueError("No metrics selected")

    n_metrics = len(metric_names)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(8, 3 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for solver in SOLVERS:
        evo = average_evolutions(args.log_dir, solver, args.n_runs, args.x_axis)
        if evo is None:
            print(f"Warning: no evolution data for {solver}, skipping.")
            continue

        if args.x_axis == "iteration":
            x = evo["iterations"]
            x_label = "Iteration"
        else:
            x = evo["times"]
            x_label = "Time (s)"

        for ax, metric in zip(axes, metric_names):
            if metric not in evo:
                print(f"Warning: {metric} not found for {solver}, skipping.")
                continue
            y = evo[metric]
            markevery = max(1, len(x) // 8)
            ax.plot(
                x, y,
                label=LABELS[solver],
                color=COLORS[solver],
                marker=MARKERS[solver],
                markevery=markevery,
                linewidth=1.5,
                markersize=8,
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

    for ax in axes:
        ax.legend(loc="best", fontsize="small")
    axes[-1].set_xlabel(x_label)
    fig.suptitle(os.path.basename(args.log_dir))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(args.output, dpi=150)
    print(f"Saved evolution plot to {args.output}")
    svg_output = os.path.splitext(args.output)[0] + ".svg"
    plt.savefig(svg_output)
    print(f"Saved evolution plot to {svg_output}")


if __name__ == "__main__":
    main()
