"""Plot projection-solver scaling across problem dimensions.

Reads the aggregated results saved by ``benchmark_projection.py --save``
(``logs/benchmark_projection/dim*/results.json``) and plots mean wall time
and solver memory against the number of vertices ``D`` on log-log axes.
A power-law exponent (slope of log-quantity vs log-D) is fitted per curve
to corroborate the complexity claims of Table ``tab:alp_complexity`` in the
paper (e.g. slope ~6 for dense solvers, ~2 for ALP/MINRES).

Runs whose mean wall time hit the benchmark's ``--max_time`` cap are drawn
with open markers, excluded from the time-slope fit, and marked with a `*`
in the legend; treat those slopes as lower bounds. Memory is unaffected by
the cap.

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

# Colors follow the solver family, linestyles the ADMM inner mode, matching
# plot_benchmark_projection.py --compare_modes.
SOLVER_COLORS = {"ipm": "C0", "activeset": "C1", "admm": "C2"}
MODE_LINESTYLES = {"optimized": "-", "naive": "--", "minres": ":"}
MARKERS = {"ipm": "o", "activeset": "s", "admm": "^"}
LABELS = {
    "ipm_optimized": "IPM (optimized)",
    "ipm_naive": "IPM (naive)",
    "activeset_optimized": "Active Set (optimized)",
    "activeset_naive": "Active Set (naive)",
    "admm_optimized": "ADMM + ALP",
    "admm_naive": "ADMM + dense LU",
    "admm_minres": "ADMM + MINRES",
}

DIM_RE = re.compile(r"dim(\d+)_")


def collect_scaling(log_root: str, device: str):
    """Scan log_root for dim* dirs and gather per-solver arrays.

    Returns {run_name: {"dims": [...], "time_s": [...], "time_s_std": [...],
    "mem_mb": [...], "capped": [...]}} sorted by dimension, where ``capped``
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
            if "time_mean_ms_mean" not in metrics:
                continue
            entry_dict = series.setdefault(run_name, {
                "dims": [], "time_s": [], "time_s_std": [], "mem_mb": [],
                "capped": [],
            })
            time_s = metrics["time_mean_ms_mean"] / 1000.0
            entry_dict["dims"].append(dim)
            entry_dict["time_s"].append(time_s)
            entry_dict["time_s_std"].append(
                metrics.get("time_mean_ms_std", 0.0) / 1000.0)
            entry_dict["mem_mb"].append(metrics.get("memory_mb_mean", np.nan))
            capped = max_time is not None and time_s >= 0.999 * max_time
            entry_dict["capped"].append(bool(capped))
    for entry_dict in series.values():
        order = np.argsort(entry_dict["dims"])
        for key in entry_dict:
            entry_dict[key] = [entry_dict[key][i] for i in order]
    return series


def fit_slope(dims, values):
    """Power-law exponent of values vs dims, or None if not fittable."""
    x = np.asarray(dims, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    return float(np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)[0])


def style_for(run_name: str):
    solver, _, mode = run_name.partition("_")
    color = SOLVER_COLORS.get(solver, "C3")
    linestyle = MODE_LINESTYLES.get(mode, "-")
    marker = MARKERS.get(solver, "d")
    label = LABELS.get(run_name, run_name.replace("_", " ").title())
    return color, linestyle, marker, label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot projection-solver time/memory scaling vs dimension"
    )
    parser.add_argument("--log_root", type=str, default="logs/benchmark_projection")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device token in the dim* directory names "
                             "(cpu, cuda0); default cpu")
    parser.add_argument("--metrics", type=str, default="time,memory",
                        help="Comma-separated: time, memory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (also saved as .svg; default: "
                             "<log_root>/scaling_<device>.png)")
    parser.add_argument("--no_fit", action="store_true",
                        help="Do not annotate power-law slope fits")
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

    for run_name in sorted(series):
        entry_dict = series[run_name]
        dims = entry_dict["dims"]
        color, linestyle, marker, base_label = style_for(run_name)
        for ax, metric in zip(axes, metric_names):
            if metric == "time":
                values = entry_dict["time_s"]
                # Capped points are shown open and excluded from the fit.
                keep = [not c for c in entry_dict["capped"]]
                fit_dims = [d for d, k in zip(dims, keep) if k]
                fit_vals = [v for v, k in zip(values, keep) if k]
                fit_stds = [s for s, k in zip(entry_dict["time_s_std"], keep) if k]
                open_dims = [d for d, k in zip(dims, keep) if not k]
                open_vals = [v for v, k in zip(values, keep) if not k]
            elif metric == "memory":
                values = entry_dict["mem_mb"]
                fit_dims, fit_vals = dims, values
                fit_stds, open_dims, open_vals = None, [], []
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if not args.no_fit:
                slope = fit_slope(fit_dims, fit_vals)
                if slope is None:
                    suffix = "[capped]" if not fit_dims else "[n/a]"
                elif open_dims:
                    suffix = f"[{slope:.1f}*]"
                else:
                    suffix = f"[{slope:.1f}]"
                label = f"{base_label} {suffix}"
            else:
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
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.6)
            ax.legend(loc="best", fontsize="small")

    fig.suptitle(f"Projection solver scaling ({args.device})")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output, dpi=150)
    plt.savefig(os.path.splitext(args.output)[0] + ".svg")
    print(f"Saved scaling plot to {args.output} (.svg)")


if __name__ == "__main__":
    main()
