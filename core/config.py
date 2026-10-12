"""Shared configuration helpers for experiment naming and paths."""

import os
from typing import Any, Dict, Optional

LOG_DIR = "logs"


def ensure_log_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    if not os.path.isdir(path):
        os.makedirs(path)


def gl_experiment_path(
    method: str,
    data_set: str,
    dim: int,
    n_samples: int,
    param_x_name: str,
    param_x: Any,
    param_y_name: str,
    param_y: Any,
    seed: int,
    param_z_name: Optional[str] = None,
    param_z: Any = None,
) -> str:
    """Build the standard log directory path for graph-learning experiments."""
    z_suffix = (
        f"-{param_z_name}{param_z:.3E}" if param_z_name is not None else ""
    )
    if method == "poly":
        name = (
            f"gl_{method}-{data_set}-dim{dim}-n{n_samples}-"
            f"{param_x_name}{param_x}-{param_y_name}{param_y:.3E}"
            f"{z_suffix}-s{seed}"
        )
    else:
        name = (
            f"gl_{method}-{data_set}-dim{dim}-n{n_samples}-"
            f"{param_x_name}{param_x:.3E}-{param_y_name}{param_y:.3E}"
            f"{z_suffix}-s{seed}"
        )
    return os.path.join(LOG_DIR, name)


def joint_experiment_path(
    data_set: str,
    method: str,
    graph_learn: str,
    batch_size: int,
    sp: int,
    alpha: float,
    seed: int,
) -> str:
    """Build the standard log directory path for joint DL+GL experiments."""
    name = f"{data_set}-{method}-{graph_learn}-b{batch_size}-s{sp}-a{alpha:.3E}-r{seed}"
    return os.path.join(LOG_DIR, name)


def benchmark_experiment_path(
    dim: int, deg: int, alpha: float, seed: int
) -> str:
    """Build the standard log directory path for benchmark_l_update.py."""
    name = f"benchmark_l_update-dim{dim}-deg{deg}-alpha{alpha:.3E}-s{seed}"
    return os.path.join(LOG_DIR, name)


def script_cmd(script: str, args: Dict[str, Any]) -> list:
    """Build a command list for launching another script via subprocess."""
    import sys

    cmd = [sys.executable, script]
    for key, value in args.items():
        cmd.append(f"--{key}")
        if isinstance(value, bool):
            # argparse expects flags without values for store_true/store_false
            continue
        cmd.append(str(value))
    return cmd
