"""Generate a LaTeX table of projection-benchmark results.

Reads the aggregated results saved by ``benchmark_projection.py --save``
(``logs/benchmark_projection/dim*/results.json``) and writes booktabs floats
with one table per selected metric (wall time, solver memory, recovery error
to the true Laplacian). Methods are rows, problem dimensions are columns,
and a missing (method, dim) pair is rendered as ``--`` -- e.g. dense solvers
above the DENSE_MAX_DIM=150 cap of run_dim_sweep.sh, which is precisely the
paper's complexity argument. Runs whose mean time hit the benchmark's
``--max_time`` cap are marked with $^{\\ast}$.

Example
-------
python plot/make_projection_table.py --device cuda0 \
    --dims 20,60,100,150,300,800 --output results_table.tex
"""

import argparse
import json
import os
import re

import numpy as np

DIM_RE = re.compile(r"dim(\d+)_")

METHOD_ORDER = [
    "ipm", "activeset", "admm_minres", "admm_cudss", "admm_optimized",
]
LABELS = {
    "ipm": "IPM",
    "activeset": "Active Set",
    "admm_minres": "ADMM + MINRES",
    "admm_cudss": "ADMM + cuDSS",
    "admm_optimized": "ALP",
}

CAPTIONS = {
    "time": "Mean wall time (ms) per solve across seeds and reps "
            "($^{\\ast}$\\,=\\,hit the {max_time} s per-solve cap).",
    "memory": "Peak GPU memory (MB) during the solve (mean across seeds).",
    "error": "Recovery error $\\Vert\\hat{\\mathbf{L}}-\\mathbf{L}\\Vert_F"
             "/\\Vert\\mathbf{L}\\Vert_F$ (mean across seeds).",
}
LABEL = {
    "time": "tab:proj_time",
    "memory": "tab:proj_memory",
    "error": "tab:proj_error",
}


def collect(log_root: str, device: str):
    """Return ({dim: {method: metrics_dict}}, max_time) for ``device``."""
    by_dim = {}
    max_time = None
    for entry in sorted(os.listdir(log_root)):
        match = DIM_RE.match(entry)
        if match is None or f"_{device}_" not in f"_{entry}_":
            continue
        results_path = os.path.join(log_root, entry, "results.json")
        if not os.path.exists(results_path):
            print(f"Warning: no results.json in {entry}, skipping.")
            continue
        with open(results_path) as f:
            payload = json.load(f)
        if max_time is None:
            max_time = payload.get("args", {}).get("max_time")
        cap = max_time
        dim = int(match.group(1))
        methods = {}
        for name, m in payload["aggregated"].items():
            if "time_mean_ms_mean" not in m:
                continue
            capped = cap is not None and \
                m["time_mean_ms_mean"] / 1000.0 >= 0.999 * cap
            methods[name] = {
                "time_ms": m["time_mean_ms_mean"],
                "time_ms_std": m.get("time_mean_ms_std", 0.0),
                "mem_mb": m.get("peak_gpu_mb_mean", np.nan),
                "err": m.get("recovery_error_mean", np.nan),
                "err_std": m.get("recovery_error_std", 0.0),
                "capped": bool(capped),
            }
        by_dim[dim] = methods
    return by_dim, max_time


def fmt(v, sig=3):
    """Format a float for a LaTeX table cell ('--' for missing)."""
    if v is None or not np.isfinite(v):
        return "--"
    s = f"{v:.{sig}g}"
    if "e" in s:
        mant, expo = s.split("e")
        return f"${mant} \\times 10^{{{int(expo)}}}$"
    return s


def make_table(metric, dims, by_dim, with_std):
    """Build one LaTeX table float for ``metric``."""
    def entry(method, dim):
        cell = by_dim.get(dim, {}).get(method)
        if cell is None:
            return "--"
        if metric == "time":
            v, s = cell["time_ms"], \
                (cell["time_ms_std"] if with_std else None)
        elif metric == "memory":
            v, s = cell["mem_mb"], None
        else:
            v, s = cell["err"], cell["err_std"] if with_std else None
        text = fmt(v)
        if s is not None and np.isfinite(s) and s > 0:
            text += f" $\\pm$ {fmt(s, sig=2)}"
        if metric == "time" and cell["capped"]:
            text += "$^{\\ast}$"
        return text

    col_spec = "l" + "c" * len(dims)
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{CAPTIONS[metric]}}}",
        f"  \\label{{{LABEL[metric]}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        "    Method & " + " & ".join(f"$D={d}$" for d in dims) + " \\\\",
        "    \\midrule",
    ]
    for method in METHOD_ORDER:
        if not any(method in by_dim.get(d, {}) for d in dims):
            continue
        cells = [entry(method, d) for d in dims]
        lines.append(f"    {LABELS[method]} & " + " & ".join(cells) + " \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LaTeX table of projection-benchmark results"
    )
    parser.add_argument("--log_root", type=str, default="logs/benchmark_projection")
    parser.add_argument("--device", type=str, default="cuda0")
    parser.add_argument("--dims", type=str, default=None,
                        help="Comma-separated dim subset (default: all found)")
    parser.add_argument("--metrics", type=str, default="time,memory,error")
    parser.add_argument("--with_std", action="store_true",
                        help="Append ± std to every entry")
    parser.add_argument("--output", type=str, default=None,
                        help=".tex output path (default: stdout)")
    args = parser.parse_args()

    by_dim, max_time = collect(args.log_root, args.device)
    if not by_dim:
        raise SystemExit(f"No results found under {args.log_root} "
                         f"for device '{args.device}'.")

    if args.dims is not None:
        wanted = [int(d) for d in args.dims.split(",")]
        dims = [d for d in wanted if d in by_dim]
        missing = [d for d in wanted if d not in by_dim]
        for d in missing:
            print(f"Warning: dim {d} not found on device {args.device}.")
    else:
        dims = sorted(by_dim)

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    cap_str = f"{max_time:.0f}" if max_time is not None else "30"
    tables = []
    for metric in metric_names:
        if metric not in CAPTIONS:
            raise SystemExit(f"Unknown metric: {metric}")
        CAPTIONS[metric] = CAPTIONS[metric].replace("{max_time}", cap_str)
        tables.append(make_table(metric, dims, by_dim, args.with_std))

    body = "\n\n".join(tables) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(body)
        print(f"Saved LaTeX table to {args.output}")
    else:
        print(body)


if __name__ == "__main__":
    main()
