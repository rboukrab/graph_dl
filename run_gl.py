"""Unified dispatcher for graph-learning experiments.

Examples
--------
python run_gl.py --algorithm glama --data_set stat --dim 20 --n_iters 100
python run_gl.py --algorithm smooth --data_set fmri --alpha 1e-2 --beta 0.3
python run_gl.py --algorithm poly --deg 2 --alpha 4
python run_gl.py --algorithm pgl --rho 0.1 --eta 1e-1
"""

import argparse

import torch

from joint_ol_gl.runners import RUNNERS, load_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified graph-learning experiment runner",
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=list(RUNNERS.keys()),
        required=True,
        help="Graph-learning algorithm to run",
    )
    parser.add_argument("--device", type=str, default=None,
                        help="Device string or None for auto")
    parser.add_argument(
        "--data_set",
        type=str,
        choices=["stat", "fmri"],
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
        "--n_iters", type=int, default=10_000, help="Number of outer iterations"
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
    parser.add_argument("--dim", type=int, default=20, help="Graph dimension")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    # Collect all algorithm-specific arguments in one parser.
    for runner_cls in RUNNERS.values():
        runner_cls.add_args(parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    runner_cls = RUNNERS[args.algorithm]
    dataset = load_dataset(args)

    # Update dim from dataset if it was not known up front (fmri).
    if args.data_set == "fmri":
        args.dim = dataset.dim

    runner = runner_cls(args, dataset, device)
    runner.run()


if __name__ == "__main__":
    main()
