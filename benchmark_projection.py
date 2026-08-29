"""Benchmark Laplacian projection solvers on a realistic H_k.

The benchmark can build the projection matrix H_k from either the new smooth
graph-signal dataset (:class:`core.datasets.SmoothDataset`) or the classic
stationary dataset (:class:`core.datasets.StatDataset`). It then compares
LaplacianIPMSparse, LaplacianActiveSet, and ProjADMM2 in terms of wall time,
inner iterations, solution quality, and memory usage.

Example
-------
python benchmark_projection.py --data_set smooth --dim 20 --n_samples 100000 --alpha 1e-2 --beta 10 --n_reps 10
"""

import argparse
import gc
import json
import os
import time
import tracemalloc
import sys
from typing import Dict, List

import numpy as np
import torch

from core.config import ensure_log_dir
from core.datasets import SmoothDataset, StatDataset
from core.graph_learn import LaplacianActiveSet, LaplacianIPMSparse, ProjADMM2


# Solvers to benchmark, with constructor flags and forward-call wrappers to
# paper over their slightly different APIs.
SOLVERS = {
    "ipm": {
        "class": LaplacianIPMSparse,
        "kwargs": {"eps_in": 1e-3, "eps_out": 1e-3, "max_inner_iter": 100, "t0": 1.0},
        "call": lambda solver, H, tol, max_iter: solver.forward(H, tolerance=tol, max_iter=max_iter),
        "iters": lambda solver: int(getattr(solver, "iter", -1)),
    },
    "activeset": {
        "class": LaplacianActiveSet,
        "kwargs": {},
        "call": lambda solver, H, tol, max_iter: solver.forward(H, max_iter=max_iter, tolerance=tol),
        "iters": lambda solver: int(getattr(solver, "iter", -1)),
    },
    "admm": {
        "class": ProjADMM2,
        "kwargs": {"mode": "optimized"},
        "call": lambda solver, H, tol, max_iter: solver.forward(H, tolerance=tol, max_iter=max_iter),
        "iters": lambda solver: int(getattr(solver, "iter", -1)),
    },
}


def _capture_h_k(
    dim: int,
    n_samples: int,
    alpha: float,
    beta: float,
    device: torch.device,
    seed: int = 1,
    data_set: str = "smooth",
    k_nn: int = 5,
    sigma2: float = 0.0,
):
    """Build the projection input H_k directly from a dataset.

    We mimic the first iteration of :class:`core.runners.SmoothRunner`: the
    inner solver is initialized with the identity Laplacian, so the captured
    ``H_k = -alpha/beta * C`` is independent of the chosen projection solver.
    """
    if data_set == "smooth":
        dataset = SmoothDataset(seed, dim, n_samples, k=k_nn, sigma2=sigma2)
    elif data_set == "stat":
        dataset = StatDataset(seed, dim, n_samples)
    else:
        raise ValueError(f"Unsupported data_set for benchmark: {data_set}")

    X_train = dataset.X.to(device)
    graph_vol = float(dataset.graph.W.todense().sum())

    # Initial Laplacian used by SmoothRunner before the first projection.
    L0 = torch.eye(dim, device=device)
    Y_train = X_train @ torch.linalg.inv(torch.eye(dim, device=device) + alpha * L0)
    C = Y_train.T @ Y_train
    H_k = -alpha * C / beta

    return H_k, graph_vol, L0


def _constraint_metrics(L: torch.Tensor, vol: float) -> Dict[str, float]:
    """Compute Laplacian constraint violations and objective helpers."""
    n = L.shape[0]
    eye = torch.eye(n, device=L.device, dtype=L.dtype)
    off = L[eye == 0]
    diag = L[eye == 1]
    return {
        "row_sum_max_abs": float(torch.abs(L.sum(dim=1)).max().cpu()),
        "trace_err": float(abs(L.trace().item() - vol)),
        "max_off": float(off.max().cpu()),
        "min_off": float(off.min().cpu()),
        "mean_diag": float(diag.mean().cpu()),
        "n_edges": int((off.abs() > 1e-6).sum().cpu()) // 2,
    }


def _objective(L: torch.Tensor, H: torch.Tensor) -> float:
    return float((0.5 * torch.norm(L - H) ** 2).cpu())


def _benchmark_solver(
    name: str,
    solver_factory,
    H: torch.Tensor,
    vol: float,
    L0: torch.Tensor,
    tol: float,
    max_iter: int,
    n_reps: int,
    device: torch.device,
) -> Dict[str, float]:
    """Run warm-up + timed repetitions for one solver."""
    call = SOLVERS[name]["call"]
    iters_fn = SOLVERS[name]["iters"]

    # Warm-up (not timed).
    solver = solver_factory()
    call(solver, H, tol, max_iter)
    del solver
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    times_ms: List[float] = []
    mem_deltas_kb: List[float] = []
    objectives: List[float] = []

    for _ in range(n_reps):
        solver = solver_factory()

        # Memory snapshot before solve.
        tracemalloc.start()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        call(solver, H, tol, max_iter)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_deltas_kb.append((peak - current) / 1024.0)

        times_ms.append(elapsed_ms)
        objectives.append(_objective(solver.L, H))

        del solver
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # Final quality run to report constraint metrics and iteration count.
    solver = solver_factory()
    call(solver, H, tol, max_iter)
    metrics = _constraint_metrics(solver.L, vol)
    metrics["objective"] = _objective(solver.L, H)
    metrics["iters"] = iters_fn(solver)
    metrics["time_mean_ms"] = float(np.mean(times_ms))
    metrics["time_std_ms"] = float(np.std(times_ms))
    metrics["time_min_ms"] = float(np.min(times_ms))
    metrics["time_max_ms"] = float(np.max(times_ms))
    metrics["mem_delta_kb_mean"] = float(np.mean(mem_deltas_kb))
    metrics["objective_std"] = float(np.std(objectives))

    if device.type == "cuda":
        metrics["peak_gpu_mb"] = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)

    del solver
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Laplacian projection solvers on a real H_k"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data_set", type=str, default="smooth",
                        choices=["stat", "smooth"],
                        help="Dataset used to build H_k")
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=100_000)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--n_reps", type=int, default=3,
                        help="Repetitions per seed (for timing stability)")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of different seeds (cases) to average over")
    parser.add_argument("--solvers", type=str, default="ipm,activeset,admm")
    parser.add_argument("--seed", type=int, default=1,
                        help="First random seed")
    parser.add_argument("--k_nn", type=int, default=5,
                        help="Number of nearest neighbours for --data_set smooth")
    parser.add_argument("--sigma2", type=float, default=0.0,
                        help="Noise power for --data_set smooth")
    parser.add_argument("--save", action="store_true", help="Save results to logs/")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    solver_names = [n.strip() for n in args.solvers.split(",") if n.strip() in SOLVERS]
    if not solver_names:
        raise ValueError(f"No known solvers selected: {args.solvers}")

    # Per-run raw results: solver -> list of metric dicts.
    raw_results: Dict[str, List[Dict[str, float]]] = {name: [] for name in solver_names}

    for run_idx in range(args.n_runs):
        seed = args.seed + run_idx
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        print(f"\n=== Run {run_idx + 1}/{args.n_runs} (seed={seed}) ===")
        H_k, vol, L0 = _capture_h_k(
            dim=args.dim,
            n_samples=args.n_samples,
            alpha=args.alpha,
            beta=args.beta,
            device=device,
            seed=seed,
            data_set=args.data_set,
            k_nn=args.k_nn,
            sigma2=args.sigma2,
        )
        print(f"H_k norm: {torch.norm(H_k).item():.4f}, vol: {vol:.4f}")

        for name in solver_names:
            cfg = SOLVERS[name]
            solver_factory = lambda: cfg["class"](
                dim=args.dim,
                L=L0,
                vol=vol,
                device=str(device),
                **cfg["kwargs"],
            )

            print(f"  Benchmarking {name}...")
            metrics = _benchmark_solver(
                name=name,
                solver_factory=solver_factory,
                H=H_k,
                vol=vol,
                L0=L0,
                tol=args.tol,
                max_iter=args.max_iter,
                n_reps=args.n_reps,
                device=device,
            )
            raw_results[name].append(metrics)

    # Aggregate across runs.
    aggregated: Dict[str, Dict[str, float]] = {}
    for name, runs in raw_results.items():
        agg = {}
        for key in runs[0].keys():
            values = np.array([r[key] for r in runs])
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
        aggregated[name] = agg

    # Print summary table.
    print("\n" + "=" * 100)
    print(f"{'Solver':<12} {'time_ms':<20} {'iters':<16} {'obj':<18} {'row_err':<12} {'trace_err':<14} {'max_off':<12}")
    print("=" * 100)
    for name, m in aggregated.items():
        time_str = f"{m['time_mean_ms_mean']:.2f} ± {m['time_mean_ms_std']:.2f}"
        iter_str = f"{m['iters_mean']:.1f} ± {m['iters_std']:.1f}"
        obj_str = f"{m['objective_mean']:.4f} ± {m['objective_std']:.4f}"
        print(f"{name:<12} {time_str:<20} {iter_str:<16} {obj_str:<18} "
              f"{m['row_sum_max_abs_mean']:<12.2e} {m['trace_err_mean']:<14.2e} {m['max_off_mean']:<12.2e}")
    print("=" * 100)

    if args.save:
        sim_path = os.path.join(
            "logs", "benchmark_projection",
            f"dim{args.dim}_alpha{args.alpha:.0e}_beta{args.beta:.0e}_seeds{args.seed}-{args.seed + args.n_runs - 1}"
        )
        ensure_log_dir(sim_path)
        out_file = os.path.join(sim_path, "results.json")
        with open(out_file, "w") as f:
            json.dump({
                "args": vars(args),
                "aggregated": aggregated,
                "raw_results": raw_results,
            }, f, indent=2)
        print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
