"""Unified grid-search dispatcher for graph-learning algorithms.

Examples
--------
python grid_search_gl.py --algorithm glama --data_set stat
python grid_search_gl.py --algorithm smooth --data_set smooth
python grid_search_gl.py --algorithm smooth --data_set fmri
python grid_search_gl.py --algorithm poly --data_set stat
python grid_search_gl.py --algorithm pgl --data_set stat
"""

import argparse
import csv
import os
import subprocess
import sys
from typing import Union

import numpy as np

from core.config import ensure_log_dir, gl_experiment_path


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["glama", "smooth", "poly", "pgl"],
        required=True,
        help="Graph-learning algorithm to grid-search",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument(
        "--data_set",
        type=str,
        choices=["stat", "fmri", "smooth"],
        default="fmri",
        help="Dataset to use",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1_000_000,
        help="Number of samples (for synthetic datasets)",
    )
    parser.add_argument(
        "--dim", type=int, default=20, help="Graph dimension"
    )
    parser.add_argument(
        "--n_iters", type=int, default=1, help="Number of outer iterations"
    )
    parser.add_argument(
        "--sim_time",
        type=float,
        default=9999999.0,
        help="Maximum simulation time in seconds",
    )
    parser.add_argument(
        "--test_interval",
        type=int,
        default=10,
        help="Compute metrics every this many iterations",
    )
    parser.add_argument(
        "--metric_mode",
        type=str,
        choices=["binary", "weighted"],
        default="binary",
        help="How to compute stat-set precision/recall/F1",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="First random seed"
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None,
        help="Number of random seeds per grid cell (default: algorithm-specific)",
    )
    parser.add_argument(
        "--grid_summary",
        type=str,
        default=None,
        help="Save final per-cell metrics to this CSV file",
    )


def _add_glama_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min_gamma_exp", type=float, default=-2.5713793,
        help="log10(min gamma)"
    )
    parser.add_argument(
        "--max_gamma_exp", type=float, default=-2.5713793,
        help="log10(max gamma)"
    )
    parser.add_argument("--n_gamma", type=int, default=1, help="Number of gamma values")
    parser.add_argument(
        "--min_delta_exp", type=float, default=-1.4285241,
        help="log10(min delta)"
    )
    parser.add_argument(
        "--max_delta_exp", type=float, default=-1.4285241,
        help="log10(max delta)"
    )
    parser.add_argument("--n_delta", type=int, default=1, help="Number of delta values")


def _add_smooth_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--solver", type=str,
        choices=["ProjADMM", "ipm", "activeset"],
        default="activeset",
        help="Inner solver for the Laplacian projection",
    )
    parser.add_argument(
        "--min_alpha_exp", type=float, default=-2.9284862,
        help="log10(min alpha)"
    )
    parser.add_argument(
        "--max_alpha_exp", type=float, default=-2.9284862,
        help="log10(max alpha)"
    )
    parser.add_argument("--n_alpha", type=int, default=1, help="Number of alpha values")
    parser.add_argument(
        "--min_beta_exp", type=float, default=-1.6667543,
        help="log10(min beta)"
    )
    parser.add_argument(
        "--max_beta_exp", type=float, default=-1.6667543,
        help="log10(max beta)"
    )
    parser.add_argument("--n_beta", type=int, default=1, help="Number of beta values")
    parser.add_argument(
        "--k_nn", type=int, default=5,
        help="Number of nearest neighbours for --data_set smooth"
    )
    parser.add_argument(
        "--sigma2", type=float, default=0.0,
        help="Noise power for --data_set smooth"
    )
    parser.add_argument(
        "--inner_metrics", action="store_true",
        help="Compute metrics at every inner projection iteration"
    )
    parser.add_argument(
        "--inner_metric_interval", type=int, default=1,
        help="Compute inner metrics every N inner iterations"
    )


def _add_poly_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min_deg", type=int, default=5, help="Minimum polynomial degree")
    parser.add_argument("--max_deg", type=int, default=5, help="Maximum polynomial degree")
    parser.add_argument(
        "--min_alpha_exp", type=float, default=0.833338389,
        help="log10(min alpha)"
    )
    parser.add_argument(
        "--max_alpha_exp", type=float, default=0.833338389,
        help="log10(max alpha)"
    )
    parser.add_argument("--n_alpha", type=int, default=1, help="Number of alpha values")
    parser.add_argument(
        "--min_eta_scale_exp", type=float, default=0.0,
        help="log10(min eta_scale)"
    )
    parser.add_argument(
        "--max_eta_scale_exp", type=float, default=0.0,
        help="log10(max eta_scale)"
    )
    parser.add_argument("--n_eta_scale", type=int, default=1,
                        help="Number of eta_scale values")


def _add_pgl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min_rho_exp", type=float, default=-9, help="log10(min rho)"
    )
    parser.add_argument(
        "--max_rho_exp", type=float, default=5, help="log10(max rho)"
    )
    parser.add_argument("--n_rho", type=int, default=8, help="Number of rho values")
    parser.add_argument(
        "--min_eta_exp", type=float, default=-5.5, help="log10(min eta)"
    )
    parser.add_argument(
        "--max_eta_exp", type=float, default=-2, help="log10(max eta)"
    )
    parser.add_argument("--n_eta", type=int, default=8, help="Number of eta values")


_ALGO_ARG_ADDERS = {
    "glama": _add_glama_args,
    "smooth": _add_smooth_args,
    "poly": _add_poly_args,
    "pgl": _add_pgl_args,
}


def _build_parser() -> Union[argparse.ArgumentParser, argparse.Namespace]:
    # For --help, show all possible arguments (legacy behavior).
    if "--help" in sys.argv or "-h" in sys.argv:
        parser = argparse.ArgumentParser(
            description="Grid search for graph-learning algorithms",
            conflict_handler="resolve",
        )
        _add_common_args(parser)
        for add_args in _ALGO_ARG_ADDERS.values():
            add_args(parser)
        return parser

    # Two-pass parse: first resolve the algorithm and common args, then add only
    # the selected algorithm's grid-search arguments. This avoids default-value
    # collisions between algorithms that share option names (e.g. --min_alpha_exp
    # for smooth and poly).
    pre_parser = argparse.ArgumentParser(add_help=False)
    _add_common_args(pre_parser)
    known, remaining = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Grid search for graph-learning algorithms",
    )
    _add_common_args(parser)
    _ALGO_ARG_ADDERS[known.algorithm](parser)
    # Re-inject --algorithm so the full parser sees its required argument.
    full_argv = ["--algorithm", known.algorithm, *remaining]
    args = parser.parse_args(full_argv, namespace=known)
    return args


def main() -> None:
    parser_or_args = _build_parser()
    if isinstance(parser_or_args, argparse.ArgumentParser):
        parser_or_args.parse_args()
        return
    args = parser_or_args

    algorithm = args.algorithm
    base_cmd = [
        sys.executable,
        "run_gl.py",
        "--algorithm",
        algorithm,
        "--device",
        str(args.device),
        "--data_set",
        str(args.data_set),
        "--n_samples",
        str(args.n_samples),
        "--dim",
        str(args.dim),
        "--n_iters",
        str(args.n_iters),
        "--sim_time",
        str(args.sim_time),
        "--test_interval",
        str(args.test_interval),
        "--metric_mode",
        str(args.metric_mode),
    ]

    if args.n_seeds is not None:
        seeds = range(args.seed, args.seed + args.n_seeds)
    elif algorithm == "glama":
        seeds = range(1, 2)
    elif algorithm == "smooth":
        seeds = range(1, 2)
    elif algorithm == "poly":
        seeds = range(1, 301)
    elif algorithm == "pgl":
        seeds = range(1, 11)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    z_name = None
    zs = [None]
    if algorithm == "glama":
        xs = np.logspace(args.min_gamma_exp, args.max_gamma_exp, args.n_gamma)
        ys = np.logspace(args.min_delta_exp, args.max_delta_exp, args.n_delta)
        x_name, y_name = "--gamma", "--delta"
    elif algorithm == "smooth":
        xs = np.logspace(args.min_alpha_exp, args.max_alpha_exp, args.n_alpha)
        ys = np.logspace(args.min_beta_exp, args.max_beta_exp, args.n_beta)
        x_name, y_name = "--alpha", "--beta"
        base_cmd.extend([
            "--solver", str(args.solver),
            "--k_nn", str(args.k_nn),
            "--sigma2", str(args.sigma2),
        ])
        if args.inner_metrics:
            base_cmd.append("--inner_metrics")
            base_cmd.extend(["--inner_metric_interval", str(args.inner_metric_interval)])
    elif algorithm == "poly":
        xs = np.arange(args.min_deg, args.max_deg + 1)
        ys = np.logspace(args.min_alpha_exp, args.max_alpha_exp, args.n_alpha)
        zs = np.logspace(args.min_eta_scale_exp, args.max_eta_scale_exp,
                         args.n_eta_scale)
        x_name, y_name, z_name = "--deg", "--alpha", "--eta_scale"
    elif algorithm == "pgl":
        xs = np.logspace(args.min_rho_exp, args.max_rho_exp, args.n_rho)
        ys = np.logspace(args.min_eta_exp, args.max_eta_exp, args.n_eta)
        x_name, y_name = "--rho", "--eta"
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    param_x_name = x_name.lstrip("-")
    param_y_name = y_name.lstrip("-")
    param_z_name = z_name.lstrip("-") if z_name is not None else None
    metric_names = ["f1_score", "precision", "recall", "nae", "sp"]
    if algorithm == "poly":
        metric_names.append("filterr")
    fieldnames = [param_x_name, param_y_name]
    if param_z_name is not None:
        fieldnames.append(param_z_name)
    fieldnames.extend(["seed", *metric_names])
    grid_rows = []

    for x in xs:
        for y in ys:
            for z in zs:
                for seed in seeds:
                    x_val = f"{x:.3E}" if algorithm != "poly" else str(int(x))
                    y_val = f"{y:.3E}"
                    cmd = [
                        *base_cmd,
                        "--seed",
                        str(seed),
                        x_name,
                        x_val,
                        y_name,
                        y_val,
                    ]
                    if z_name is not None:
                        cmd.extend([z_name, f"{z:.3E}"])
                    subprocess.run(cmd, check=False)

                    sim_path = gl_experiment_path(
                        method=algorithm,
                        data_set=args.data_set,
                        dim=args.dim,
                        n_samples=args.n_samples,
                        param_x_name=param_x_name,
                        param_x=x,
                        param_y_name=param_y_name,
                        param_y=y,
                        seed=seed,
                        param_z_name=param_z_name,
                        param_z=z,
                    )
                    row = {param_x_name: x, param_y_name: y, "seed": seed}
                    if param_z_name is not None:
                        row[param_z_name] = z
                    for metric in metric_names:
                        try:
                            arr = np.load(os.path.join(sim_path, f"{metric}.npy"))
                            row[metric] = float(arr[-1])
                        except Exception:
                            row[metric] = float("nan")
                    grid_rows.append(row)

    if args.grid_summary:
        summary_dir = os.path.dirname(args.grid_summary)
        if summary_dir:
            ensure_log_dir(summary_dir)
        with open(args.grid_summary, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(grid_rows)
        print(f"Saved grid summary to {args.grid_summary}")


if __name__ == "__main__":
    main()
