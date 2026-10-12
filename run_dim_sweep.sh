#!/usr/bin/env bash
# Projection-solver comparison across dims, on GPU.
#
# Solver/mode grid (per dim):
#   ipm, activeset : naive mode only (their optimized/minres variants are
#                    not part of the comparison)
#   admm           : naive, optimized, minres AND cudss (the paper's
#                    ADMM+solver rows vs ALP)
# Because benchmark_projection.py applies --modes to every --solver
# (cross-product), each dim needs TWO invocations into the SAME
# dim<d>_..._seeds1-5 dir (the dir name carries no solver/mode tag):
#   1) --solvers ipm,activeset --modes naive
#   2) --solvers admm         --modes naive,optimized,minres,cudss
# benchmark_projection.py --save MERGES into an existing results.json, so
# the combined file ends up with ipm, activeset, admm_naive, admm_optimized,
# admm_minres and admm_cudss (single-mode invocations keep the plain solver
# names). Both invocations calibrate --target_f1 from the same seed, so they
# reuse the identical noise level and timings stay comparable across groups.
#
# DIMS is log-spaced from 20 to 1000: a log-log slope fit needs support along
# the D axis more than tight error bars at each point. Above DENSE_MAX_DIM
# (=150) the dense-solver modes (ipm/activeset naive, admm naive) are NOT
# run: their O(D^4) dense KKT needs 1.6 GB already at dim 200 (8.3 GB at
# 300, 63 GB at 500, 1 TB at 1000), and dim>=200 is crash territory on some
# builds -- so 150 is the hard cap for anything dense, and dims 200-1000
# run only the O(D^2)-memory admm modes (optimized=ALP, minres, cudss
# sparse-direct). The scaling plot shows the dense series simply not having
# points above 150 -- the paper's complexity argument is that they cannot
# afford to be there at all.
#
# Do NOT run the dim>150 entries ON THIS MACHINE: the custom PyTorch build
# segfaults intermittently at dim>=200 in sparse COO ops (pre-existing,
# affects all modes, output is lost unflushed when it hits). Standard torch
# builds (the server) are fine -- smoke-test one dim=200 run there first:
#   python benchmark_projection.py --device cuda:0 --dim 200 --target_f1 0.8 \
#       --n_runs 1 --n_reps 1 --max_time 120 --solvers admm \
#       --modes optimized,minres,cudss
# Server extra needed for the cudss mode: pip install torch-sla[cudss]
#
# GPU-only sweep, and not by choice: this machine's custom PyTorch build
# lacks CPU LAPACK, so torch.linalg.solve (naive mode dense KKT solve, and
# ipm naive/active-set refactorizations) crashes on CPU with
# "requires compiling PyTorch with LAPACK". On CUDA the same calls route to
# cuSOLVER and work. CPU timings would be preferable for the paper, so if
# torch is ever rebuilt with LAPACK, switch DEVICES back to (cpu) (GPU adds
# nothing here anyway: measured CPU-vs-CUDA wall time is flat, 42-53ms at
# dim=100, both admm modes -- the ops are kernel-launch-latency bound).
#
# Per-point repetition is trimmed to n_runs=5 seeds x n_reps=2 timing reps =
# 10 timings per (solver, mode, dim) point, down from 30. That is still
# enough for a stable median per point, and the slope fit across dims
# averages out the residual noise. Expected runtime: dims <=150 take ~2-5
# min each; 200-300 ~5-15 min (naive LU dominates); 500 ~10-20 min; 800/1000
# ~15-45 min (minres Krylov cost and cuDSS re-factorization per ADMM
# iteration dominate -- torch-sla 0.3.3 has no cached cuDSS factorization).
# Solves that hit --max_time are capped runs: excluded from the slope fit,
# marked as such in the scaling plot (expected for ipm everywhere).
#
# Uses --target_f1 so recovery difficulty is held constant across dims
# (see conversation: comparing solvers at fixed noise_level confounds
# problem-size scaling with problem-difficulty scaling).

set -uo pipefail

DIMS=(20 30 40 60 80 100 125 150 200 300 500 800 1000)
DENSE_MAX_DIM=150
DEVICES=(cuda:0)
TARGET_F1=0.8
MAX_TIME=30
MAX_ITER=5000
N_REPS=2
N_RUNS=5
# Per-invocation safety cap: each call loops over n_runs seeds x its
# solver/mode combos x (warmup + n_reps + 1 evolution) passes, each up to
# MAX_TIME. Worst case per dim: 5 combos x 5 seeds x 4 passes x 30s = 50
# min (dims <=150), 30 min (dims >=200, 3 combos). Realistic runtime is well
# below that (see header); 3.5h is kept for margin, so 26 invocations x 3.5h
# = 91h is an absolute ceiling that should never be approached. If an
# invocation is killed (OOM, crash), the already-saved results from prior
# invocations (including the merged results.json) are untouched -- only that
# one invocation's combos are lost.
PER_INVOCATION_TIMEOUT=3.5h

run_benchmark() {
  local device="$1" dim="$2" solvers="$3" modes="$4"
  echo "=== device=${device} dim=${dim} solvers=${solvers} modes=${modes} (started $(date)) ==="
  timeout "${PER_INVOCATION_TIMEOUT}" python benchmark_projection.py \
    --device "${device}" \
    --dim "${dim}" \
    --target_f1 "${TARGET_F1}" \
    --max_time "${MAX_TIME}" \
    --max_iter "${MAX_ITER}" \
    --n_reps "${N_REPS}" \
    --n_runs "${N_RUNS}" \
    --solvers "${solvers}" \
    --modes "${modes}" \
    --save \
    || echo "!!! FAILED or TIMED OUT: device=${device} dim=${dim} solvers=${solvers} modes=${modes} (continuing, finished $(date))"
}

for device in "${DEVICES[@]}"; do
  for dim in "${DIMS[@]}"; do
    if (( dim <= DENSE_MAX_DIM )); then
      run_benchmark "${device}" "${dim}" ipm,activeset naive
      run_benchmark "${device}" "${dim}" admm naive,optimized,minres,cudss
    else
      # Dense-solver modes are capped at DENSE_MAX_DIM (=150) to stay clear
      # of the memory-crash territory (dense KKT is 1.6 GB already at dim
      # 200, and dim>=200 has shown intermittent crashes on some builds).
      # Only the O(D^2)-memory admm modes run here; no invocation 1 at all,
      # so results.json for these dims holds admm_optimized/admm_minres/
      # admm_cudss only -- the scaling plot handles missing series per dim.
      run_benchmark "${device}" "${dim}" admm optimized,minres,cudss
    fi
  done
done

echo "=== sweep finished $(date) ==="
