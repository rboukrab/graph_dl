"""Benchmark Laplacian projection solvers on a noisy Laplacian.

The benchmark corrupts a valid combinatorial Laplacian with edge noise using
:class:`core.datasets.NoisyLaplacianDataset` and measures how well each
projection solver (ProjIPM, LaplacianActiveSet, ProjADMM) recovers
a valid Laplacian.  Metrics include wall time, inner iterations, constraint
violations, objective value, and recovery error relative to the ground-truth
Laplacian.

Example
-------
python benchmark_projection.py --dim 20 --noise_level 0.2 --n_reps 10
"""

import argparse
import gc
import json
import os
import time
import tracemalloc
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

from core.config import ensure_log_dir
from core.datasets import NoisyLaplacianDataset
from core.graph_learn import LaplacianActiveSet, ProjIPM, ProjADMM
from core.metrics import evaluate_laplacian_weighted


# Solvers to benchmark, with constructor flags and forward-call wrappers to
# paper over their slightly different APIs.
SOLVERS = {
    "ipm": {
        "class": ProjIPM,
        "kwargs": {"eps_in": 1e-3, "eps_out": 1e-3, "max_inner_iter": 100, "t0": 1.0},
        "call": lambda solver, H, tol, max_iter, max_time=None, callback=None: solver.forward(
            H, tolerance=tol, max_iter=max_iter, callback=callback, max_time=max_time
        ),
        "iters": lambda solver: int(getattr(solver, "iter", -1)),
    },
    "activeset": {
        "class": LaplacianActiveSet,
        "kwargs": {},
        "call": lambda solver, H, tol, max_iter, max_time=None, callback=None: solver.forward(
            H, max_iter=max_iter, tolerance=tol, callback=callback, max_time=max_time
        ),
        "iters": lambda solver: int(getattr(solver, "iter", -1)),
    },
    "admm": {
        "class": ProjADMM,
        "kwargs": {"mode": "optimized"},
        "call": lambda solver, H, tol, max_iter, max_time=None, callback=None: solver.forward(
            H, tolerance=tol, max_iter=max_iter, callback=callback, max_time=max_time
        ),
        "iters": lambda solver: int(getattr(solver, "iter", -1)),
    },
}


def _build_noisy_input(
    dim: int,
    noise_level: float,
    device: torch.device,
    seed: int = 1,
    graph_type: str = "rbf",
):
    """Build a noisy Laplacian projection input.

    Returns the ground-truth graph, Laplacian ``L``, the noisy matrix ``H``
    used as input to the projection solvers, and an identity initial
    Laplacian ``L0``.
    """
    dataset = NoisyLaplacianDataset(
        seed=seed, dim=dim, noise_level=noise_level, graph_type=graph_type
    )
    graph = dataset.graph
    H = dataset.H.to(device)
    L_true = dataset.L.to(device)
    graph_vol = dataset.vol
    L0 = torch.eye(dim, device=device)
    return H, L_true, graph, graph_vol, L0


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


def _recovery_error(L_hat: torch.Tensor, L_true: torch.Tensor) -> float:
    """Relative Frobenius recovery error."""
    return float(torch.norm(L_hat - L_true) / torch.norm(L_true))


class _MetricCallback:
    """Record solver metrics at regular iteration intervals."""

    def __init__(
        self,
        H: torch.Tensor,
        L_true: torch.Tensor,
        graph,
        vol: float,
        test_interval: int,
        L0: torch.Tensor,
    ):
        self.H = H
        self.L_true = L_true
        self.graph = graph
        self.vol = vol
        self.test_interval = test_interval
        self.iterations: List[int] = [0]
        self.times: List[float] = [0.0]
        self.objectives: List[float] = [_objective(L0, H)]
        self.recovery_errors: List[float] = [_recovery_error(L0, L_true)]
        cm0 = _constraint_metrics(L0, vol)
        self.row_sum_max_abs: List[float] = [cm0["row_sum_max_abs"]]
        self.trace_err: List[float] = [cm0["trace_err"]]
        self.max_off: List[float] = [cm0["max_off"]]
        gm0 = evaluate_laplacian_weighted(self.graph, L0)
        self.precision: List[float] = [gm0["precision"]]
        self.recall: List[float] = [gm0["recall"]]
        self.f1_score: List[float] = [gm0["f1_score"]]
        self.nae: List[float] = [gm0["nae"]]
        self.sp: List[float] = [gm0["sp"]]
        self._t0: Optional[float] = None

    def __call__(self, iteration: int, L: torch.Tensor) -> None:
        if self._t0 is None:
            self._t0 = time.perf_counter()
        if iteration % self.test_interval != 0:
            return
        self.iterations.append(iteration)
        self.times.append(time.perf_counter() - self._t0)
        self.objectives.append(_objective(L, self.H))
        self.recovery_errors.append(_recovery_error(L, self.L_true))
        cm = _constraint_metrics(L, self.vol)
        self.row_sum_max_abs.append(cm["row_sum_max_abs"])
        self.trace_err.append(cm["trace_err"])
        self.max_off.append(cm["max_off"])
        gm = evaluate_laplacian_weighted(self.graph, L)
        self.precision.append(gm["precision"])
        self.recall.append(gm["recall"])
        self.f1_score.append(gm["f1_score"])
        self.nae.append(gm["nae"])
        self.sp.append(gm["sp"])

    def to_dict(self) -> Dict[str, List]:
        return {
            "iterations": self.iterations,
            "times": self.times,
            "objectives": self.objectives,
            "recovery_errors": self.recovery_errors,
            "row_sum_max_abs": self.row_sum_max_abs,
            "trace_err": self.trace_err,
            "max_off": self.max_off,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "nae": self.nae,
            "sp": self.sp,
        }


def _benchmark_solver(
    name: str,
    solver_factory,
    H: torch.Tensor,
    L_true: torch.Tensor,
    graph,
    vol: float,
    L0: torch.Tensor,
    tol: float,
    max_iter: int,
    max_time: Optional[float],
    n_reps: int,
    test_interval: int,
    device: torch.device,
) -> Dict[str, float]:
    """Run warm-up + timed repetitions + metric-evolution run for one solver."""
    call = SOLVERS[name]["call"]
    iters_fn = SOLVERS[name]["iters"]

    # Warm-up (not timed).
    solver = solver_factory()
    call(solver, H, tol, max_iter, max_time=max_time)
    del solver
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    times_ms: List[float] = []
    mem_deltas_kb: List[float] = []
    objectives: List[float] = []
    recovery_errors: List[float] = []

    for _ in range(n_reps):
        solver = solver_factory()

        # Memory snapshot before solve.
        tracemalloc.start()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        call(solver, H, tol, max_iter, max_time=max_time)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_deltas_kb.append((peak - current) / 1024.0)

        times_ms.append(elapsed_ms)
        objectives.append(_objective(solver.L, H))
        recovery_errors.append(_recovery_error(solver.L, L_true))

        del solver
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # Metric-evolution run: record metrics at every test_interval iteration.
    solver = solver_factory()
    callback = _MetricCallback(H, L_true, graph, vol, test_interval, L0)
    call(solver, H, tol, max_iter, max_time=max_time, callback=callback)
    metrics = _constraint_metrics(solver.L, vol)
    metrics["objective"] = _objective(solver.L, H)
    metrics["recovery_error"] = _recovery_error(solver.L, L_true)
    metrics["input_recovery_error"] = _recovery_error(H, L_true)
    metrics["iters"] = iters_fn(solver)
    metrics["time_mean_ms"] = float(np.mean(times_ms))
    metrics["time_std_ms"] = float(np.std(times_ms))
    metrics["time_min_ms"] = float(np.min(times_ms))
    metrics["time_max_ms"] = float(np.max(times_ms))
    metrics["mem_delta_kb_mean"] = float(np.mean(mem_deltas_kb))
    metrics["objective_std"] = float(np.std(objectives))
    metrics["recovery_error_std"] = float(np.std(recovery_errors))
    metrics["evolution"] = callback.to_dict()

    if device.type == "cuda":
        metrics["peak_gpu_mb"] = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)

    del solver
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Laplacian projection solvers on a real H_k"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--noise_level", type=float, default=0.1,
                        help="Relative Frobenius-norm noise level on edges")
    parser.add_argument("--graph_type", type=str, default="rbf",
                        choices=["rbf", "rgg"],
                        help="Graph model used to build the ground-truth Laplacian")
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--max_time", type=float, default=None,
                        help="Maximum wall time per solver solve in seconds")
    parser.add_argument("--test_interval", type=int, default=1,
                        help="Record evolution metrics every N solver iterations")
    parser.add_argument("--n_reps", type=int, default=3,
                        help="Repetitions per seed (for timing stability)")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of different seeds (cases) to average over")
    parser.add_argument("--solvers", type=str, default="ipm,activeset,admm")
    parser.add_argument("--seed", type=int, default=1,
                        help="First random seed")
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
        H, L_true, graph, vol, L0 = _build_noisy_input(
            dim=args.dim,
            noise_level=args.noise_level,
            device=device,
            seed=seed,
            graph_type=args.graph_type,
        )
        print(f"H norm: {torch.norm(H).item():.4f}, "
              f"L_true norm: {torch.norm(L_true).item():.4f}, vol: {vol:.4f}")

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
                H=H,
                L_true=L_true,
                graph=graph,
                vol=vol,
                L0=L0,
                tol=args.tol,
                max_iter=args.max_iter,
                max_time=args.max_time,
                n_reps=args.n_reps,
                test_interval=args.test_interval,
                device=device,
            )
            raw_results[name].append(metrics)

    # Aggregate across runs.
    aggregated: Dict[str, Dict[str, float]] = {}
    for name, runs in raw_results.items():
        agg = {}
        for key in runs[0].keys():
            if key == "evolution":
                continue
            values = np.array([r[key] for r in runs])
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
        aggregated[name] = agg

    # Print summary table.
    print("\n" + "=" * 120)
    print(f"{'Solver':<12} {'time_ms':<20} {'iters':<16} {'obj':<18} "
          f"{'rec_err':<18} {'row_err':<12} {'trace_err':<14} {'max_off':<12}")
    print("=" * 120)
    for name, m in aggregated.items():
        time_str = f"{m['time_mean_ms_mean']:.2f} ± {m['time_mean_ms_std']:.2f}"
        iter_str = f"{m['iters_mean']:.1f} ± {m['iters_std']:.1f}"
        obj_str = f"{m['objective_mean']:.4f} ± {m['objective_std']:.4f}"
        rec_str = f"{m['recovery_error_mean']:.4f} ± {m['recovery_error_std']:.4f}"
        print(f"{name:<12} {time_str:<20} {iter_str:<16} {obj_str:<18} {rec_str:<18} "
              f"{m['row_sum_max_abs_mean']:<12.2e} {m['trace_err_mean']:<14.2e} {m['max_off_mean']:<12.2e}")
    print("=" * 120)

    if args.save:
        sim_path = os.path.join(
            "logs", "benchmark_projection",
            f"dim{args.dim}_noise{args.noise_level:.0e}_{args.graph_type}_seeds{args.seed}-{args.seed + args.n_runs - 1}"
        )
        ensure_log_dir(sim_path)
        out_file = os.path.join(sim_path, "results.json")

        # Save evolution arrays per solver/run on disk; keep only scalar
        # metrics in the JSON summary.
        raw_for_json: Dict[str, List[Dict[str, float]]] = {name: [] for name in solver_names}
        for name, runs in raw_results.items():
            for run_idx, run in enumerate(runs):
                run_copy = {k: v for k, v in run.items() if k != "evolution"}
                raw_for_json[name].append(run_copy)
                evo = run["evolution"]
                evo_dir = os.path.join(sim_path, name, f"run{run_idx}")
                ensure_log_dir(evo_dir)
                for arr_name, arr in evo.items():
                    np.save(os.path.join(evo_dir, f"{arr_name}.npy"), np.asarray(arr))

        with open(out_file, "w") as f:
            json.dump({
                "args": vars(args),
                "aggregated": aggregated,
                "raw_results": raw_for_json,
            }, f, indent=2)
        print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
