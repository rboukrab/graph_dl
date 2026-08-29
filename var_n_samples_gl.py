"""Unified sample-size scalability sweep for graph-learning algorithms.

Runs each algorithm over a log-spaced range of sample sizes `N` and many
random seeds, then saves results under the standard log directory.

Examples
--------
python var_n_samples_gl.py --algorithm glama --data_set stat
python var_n_samples_gl.py --algorithm smooth --data_set stat
python var_n_samples_gl.py --algorithm poly --data_set stat
python var_n_samples_gl.py --algorithm pgl --data_set stat
"""

import argparse
import subprocess
import sys
from typing import Union

import numpy as np


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["glama", "smooth", "poly", "pgl"],
        required=True,
        help="Graph-learning algorithm to benchmark",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument(
        "--data_set",
        type=str,
        choices=["stat", "fmri"],
        default="stat",
        help="Dataset to use",
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
        "--min_n_exp", type=int, default=2, help="log10(min number of samples)"
    )
    parser.add_argument(
        "--max_n_exp", type=int, default=6, help="log10(max number of samples)"
    )
    parser.add_argument(
        "--n_n", type=int, default=9, help="Number of sample sizes"
    )


def _add_glama_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gamma", type=float, default=3.162e-3, help="GLAMA gamma")
    parser.add_argument("--delta", type=float, default=1e-2, help="GLAMA delta")


def _add_smooth_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--alpha", type=float, default=3.162e-4, help="Smoothness alpha")
    parser.add_argument("--beta", type=float, default=1.0, help="Smoothness beta")


def _add_poly_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--deg", type=int, default=1, help="Polynomial degree")
    parser.add_argument("--alpha", type=float, default=3.0, help="Polynomial alpha")


def _add_pgl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rho", type=float, default=0.1, help="PGL rho")
    parser.add_argument("--eta", type=float, default=1e-4, help="PGL eta")


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
            description="Sample-size scalability sweep for graph-learning algorithms",
            conflict_handler="resolve",
        )
        _add_common_args(parser)
        for add_args in _ALGO_ARG_ADDERS.values():
            add_args(parser)
        return parser

    # Two-pass parse: first resolve the algorithm and common args, then add only
    # the selected algorithm's hyper-parameters. This avoids default-value
    # collisions between algorithms that share option names (e.g. --alpha for
    # smooth and poly).
    pre_parser = argparse.ArgumentParser(add_help=False)
    _add_common_args(pre_parser)
    known, remaining = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Sample-size scalability sweep for graph-learning algorithms",
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
        "--n_iters",
        str(args.n_iters),
        "--sim_time",
        str(args.sim_time),
        "--test_interval",
        str(args.test_interval),
    ]

    if algorithm == "glama":
        param_args = ["--gamma", f"{args.gamma:.3E}", "--delta", f"{args.delta:.3E}"]
    elif algorithm == "smooth":
        param_args = ["--alpha", f"{args.alpha:.3E}", "--beta", f"{args.beta:.3E}"]
    elif algorithm == "poly":
        param_args = ["--deg", str(args.deg), "--alpha", f"{args.alpha:.3E}"]
    elif algorithm == "pgl":
        param_args = ["--rho", f"{args.rho:.3E}", "--eta", f"{args.eta:.3E}"]
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    ns = np.logspace(args.min_n_exp, args.max_n_exp, args.n_n).astype(int)

    for n_samples in ns:
        for seed in range(1, 101):
            cmd = [
                *base_cmd,
                "--n_samples",
                str(n_samples),
                "--seed",
                str(seed),
                *param_args,
            ]
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
