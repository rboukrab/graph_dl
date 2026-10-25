"""Plot projection-solver scaling across problem dimensions.

Reads the aggregated results saved by ``benchmark_projection.py --save``
(``logs/benchmark_projection/dim*/results.json``) and plots mean wall time,
the peak GPU allocation during the solve (CUDA logs only), and the recovery
error to the true Laplacian against the number of vertices ``D`` on log-log
axes. The error panel shows that converged solvers agree on the solution
regardless of speed.

Runs whose mean wall time hit the benchmark's ``--max_time`` cap are drawn
with open markers; their true cost is at least the plotted value.

Example
-------
python plot/plot_projection_scaling.py \\
    --log_root logs/benchmark_projection \\
    --device cpu

Use ``--no_tex`` on machines without a LaTeX installation.
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times",
    "font.size": 15,
    "legend.fontsize": 10,
    "figure.autolayout": True,
})

# One color per (solver, mode) series from the colorblind-safe Okabe-Ito
# palette: family color + linestyle alone made the four ADMM variants hard to
# tell apart. Linestyle still encodes the inner mode, the marker the family.
SERIES_COLORS = {
    "ipm": "#0072B2",
    "activeset": "#E69F00",
    "admm_naive": "#D55E00",
    "admm_minres": "#56B4E9",
    "admm_cudss": "#CC79A7",
    "admm_optimized": "#009E73",
}
MODE_LINESTYLES = {"optimized": "-", "naive": "--", "minres": ":", "cudss": "-."}
MARKERS = {"ipm": "o", "activeset": "s", "admm": "^"}
LABELS = {
    "ipm": "IPM",
    "activeset": "Active Set",
    "ipm_optimized": "IPM (optimized)",
    "ipm_naive": "IPM",
    "activeset_optimized": "Active Set (optimized)",
    "activeset_naive": "Active Set",
    "admm_optimized": "ADMM + ALP",
    "admm_naive": "ADMM + dense LU",
    "admm_minres": "ADMM + MINRES",
    "admm_cudss": "ADMM + cuDSS",
}

# Legacy series dropped from the plot: the dense-LU ADMM mode was removed
# from ProjADMM ('naive' now aliases to 'cudss'), so old admm_naive runs no
# longer represent a method that exists in the code.
SKIP_SERIES = {"admm_naive"}

DIM_RE = re.compile(r"dim(\d+)_")

METRIC_YLABELS = {
    "time": "Time (s)",
    "gpumem": "Peak GPU memory (MB)",
    "error": r"$\Vert \hat{\mathbf{L}} - \mathbf{L} \Vert_F / "
             r"\Vert \mathbf{L} \Vert_F$",
}


def collect_scaling(log_root: str, device: str):
    """Scan log_root for dim* dirs and gather per-solver arrays.

    Returns {run_name: {"dims": [...], "time_s": [...], "time_s_std": [...],
    "gpumem": [...], "capped": [...]}} sorted by dimension, where ``capped``
    flags runs whose mean wall time hit that run's --max_time cap.
    """
    series = {}
    for entry in sorted(os.listdir(log_root)):
        match = DIM_RE.match(entry)
        if match is None:
            continue
        if f"_{device}_" not in f"_{entry}_":
            continue
        results_path = os.path.join(log_root, entry, "results.json")
        if not os.path.exists(results_path):
            print(f"Warning: no results.json in {entry}, skipping.")
            continue
        with open(results_path) as f:
            payload = json.load(f)
        aggregated = payload["aggregated"]
        max_time = payload.get("args", {}).get("max_time")
        dim = int(match.group(1))
        for run_name, metrics in aggregated.items():
            if run_name in SKIP_SERIES:
                continue
            if "time_mean_ms_mean" not in metrics:
                continue
            entry_dict = series.setdefault(run_name, {
                "dims": [], "time_s": [], "time_s_std": [],
                "gpumem": [], "gpumem_std": [],
                "err": [], "err_std": [], "capped": [],
            })
            time_s = metrics["time_mean_ms_mean"] / 1000.0
            entry_dict["dims"].append(dim)
            entry_dict["time_s"].append(time_s)
            entry_dict["time_s_std"].append(
                metrics.get("time_mean_ms_std", 0.0) / 1000.0)
            # Peak GPU allocation during the solve; absent for CPU runs and
            # some legacy logs (kept as NaN, dropped per-panel).
            entry_dict["gpumem"].append(
                metrics.get("peak_gpu_mb_mean", np.nan))
            entry_dict["gpumem_std"].append(
                metrics.get("peak_gpu_mb_std", np.nan))
            entry_dict["err"].append(metrics.get("recovery_error_mean", np.nan))
            entry_dict["err_std"].append(
                metrics.get("recovery_error_std", 0.0))
            capped = max_time is not None and time_s >= 0.999 * max_time
            entry_dict["capped"].append(bool(capped))
    for entry_dict in series.values():
        order = np.argsort(entry_dict["dims"])
        for key in entry_dict:
            entry_dict[key] = [entry_dict[key][i] for i in order]
    return series


def style_for(run_name: str):
    solver, _, mode = run_name.partition("_")
    if not mode and solver in ("ipm", "activeset"):
        # Naive-only runs saved under the plain solver name.
        mode = "naive"
    color = SERIES_COLORS.get(run_name) \
        or SERIES_COLORS.get(f"{solver}_{mode}", "C3")
    linestyle = MODE_LINESTYLES.get(mode, "-")
    marker = MARKERS.get(solver, "d")
    label = LABELS.get(run_name, run_name.replace("_", " ").title())
    return color, linestyle, marker, label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot projection-solver time/GPU-memory scaling vs "
                    "problem dimension"
    )
    parser.add_argument("--log_root", type=str, default="logs/benchmark_projection")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device token in the dim* directory names "
                             "(cpu, cuda0); default cpu")
    parser.add_argument("--metrics", type=str, default="time,gpumem",
                        help="Comma-separated panels: time, gpumem, error "
                             "(gpumem = peak GPU allocation during the "
                             "solve, CUDA logs only; error = recovery error "
                             "to the true Laplacian)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (also saved as .svg; default: "
                             "<log_root>/scaling_<device>.png)")
    parser.add_argument("--no_tex", action="store_true",
                        help="Disable LaTeX text rendering")
    args = parser.parse_args()

    if args.no_tex:
        plt.rcParams["text.usetex"] = False
    if args.output is None:
        args.output = os.path.join(args.log_root, f"scaling_{args.device}.png")

    series = collect_scaling(args.log_root, args.device)
    if not series:
        raise SystemExit(
            f"No results found under {args.log_root} for device '{args.device}'. "
            "Run benchmark_projection.py --save first."
        )

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    fig, axes = plt.subplots(1, len(metric_names),
                             figsize=(6 * len(metric_names), 5), squeeze=False)
    axes = axes[0]

    # Pin the x-axis to the measured dim range: log-scale autoscale rounds
    # the limits outward (starting near 10^0 although the smallest
    # measurement is D = 20), wasting a decade of empty axis. Add a small
    # constant margin in log space (i.e. multiplicative on D) so the first
    # and last points don't sit on the frame.
    d_lo = min(min(e["dims"]) for e in series.values())
    d_hi = max(max(e["dims"]) for e in series.values())
    log_pad = 0.03 * (np.log10(d_hi) - np.log10(d_lo))
    x_lo = 10 ** (np.log10(d_lo) - log_pad)
    x_hi = 10 ** (np.log10(d_hi) + log_pad)

    for run_name in sorted(series):
        entry_dict = series[run_name]
        dims = entry_dict["dims"]
        color, linestyle, marker, base_label = style_for(run_name)
        for ax, metric in zip(axes, metric_names):
            if metric == "time":
                values = entry_dict["time_s"]
                # Capped points are plotted separately with open markers.
                keep = [not c for c in entry_dict["capped"]]
                fit_dims = [d for d, k in zip(dims, keep) if k]
                fit_vals = [v for v, k in zip(values, keep) if k]
                fit_stds = [s for s, k in zip(entry_dict["time_s_std"], keep) if k]
                open_dims = [d for d, k in zip(dims, keep) if not k]
                open_vals = [v for v, k in zip(values, keep) if not k]
            elif metric == "gpumem":
                # Peak GPU allocation during the solve (CUDA runs only).
                # Unaffected by the wall-time cap. Legacy runs without the
                # metric are dropped pointwise; series with no valid points
                # at all are skipped for this panel.
                pairs = [(d, v, s) for d, v, s in zip(
                    dims, entry_dict["gpumem"], entry_dict["gpumem_std"])
                    if np.isfinite(v)]
                if not pairs:
                    continue
                fit_dims = [p[0] for p in pairs]
                fit_vals = [p[1] for p in pairs]
                fit_stds = [p[2] for p in pairs]
                open_dims, open_vals = [], []
            elif metric == "error":
                # Recovery error to the true Laplacian. No power-law fit
                # (a quality panel, not a complexity one), and capped runs
                # keep their (worse) values: the point is that converged
                # solvers land on the same solution.
                values = entry_dict["err"]
                fit_dims, fit_vals = dims, values
                fit_stds = entry_dict["err_std"]
                open_dims, open_vals = [], []
            else:
                raise ValueError(f"Unknown metric: {metric}")

            label = base_label

            if fit_stds is not None:
                ax.errorbar(fit_dims, fit_vals, yerr=fit_stds, label=label,
                            color=color, linestyle=linestyle, marker=marker,
                            markersize=8, linewidth=1.5, capsize=3)
            else:
                ax.plot(fit_dims, fit_vals, label=label, color=color,
                        linestyle=linestyle, marker=marker, markersize=8,
                        linewidth=1.5)
            if open_dims:
                ax.plot(open_dims, open_vals, color=color, linestyle="none",
                        marker=marker, markersize=8, markerfacecolor="none")
            ax.set_xlabel("Number of vertices $D$")
            ax.set_ylabel(METRIC_YLABELS.get(metric, metric))
            ax.set_xscale("log")
            ax.set_xlim(x_lo, x_hi)
            ax.set_yscale("log")
            if metric == "gpumem":
                ax.set_ylim(bottom=1)
            ax.grid(True, which="both", linestyle="--", alpha=0.6)

    # Shared legend below the panels: an in-axes legend overlaps the data
    # region, and its line samples are easily mistaken for measurements
    # (e.g. the cuDSS sample floating above the dense-LU curve).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize="small",
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Projection solver scaling ({args.device})")
    plt.tight_layout(rect=[0, 0.09, 1, 0.96])
    plt.savefig(args.output, dpi=150)
    plt.savefig(os.path.splitext(args.output)[0] + ".svg")
    print(f"Saved scaling plot to {args.output} (.svg)")


if __name__ == "__main__":
    main()
