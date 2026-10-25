#!/usr/bin/env bash
# Re-measure all ADMM modes (ALP, MINRES, cuDSS) across all dims, on GPU.
#
# Purpose: the ProjADMM 'optimized' (ALP) loop was rewritten, so the existing
# admm_optimized rows in logs/benchmark_projection/dim*/results.json are
# stale. The admm_minres/admm_cudss code is unchanged, but the original
# sweep's GPU model was never recorded -- if this machine's GPU differs from
# the original server's, ALP rows measured here would not be comparable with
# the preserved MINRES/cuDSS rows. Re-running all three ADMM modes together
# keeps the ADMM-vs-ADMM comparison same-GPU no matter what the old GPU was.
#
# benchmark_projection.py --save MERGES into the existing results.json and
# now records a gpu_name field with a WARNING on cross-GPU merges. The
# ipm/activeset rows (dims <=150 only) are time-capped at every dim, so
# their absolute numbers carry no timing information anyway; they are NOT
# re-run here.
#
# Flags mirror the original sweep (run_dim_sweep.sh) so the refreshed rows
# line up with the preserved ones.
#
# Expected runtime: dims <=150 ~2-5 min each; 200-300 ~5-15 min; 500 ~10-20
# min; 800/1000 ~15-45 min (cuDSS re-factorizes per ADMM iteration; torch-sla
# has no cached cuDSS factorization). Total ~1.5-3 h.
#
# Do NOT run dim>150 ON THIS MACHINE: the custom PyTorch build segfaults
# intermittently at dim>=200 (pre-existing). Standard torch builds (the
# server) are fine.

set -uo pipefail

DIMS=(20 30 40 60 80 100 125 150 200 300 500 800 1000)
DEVICES=(cuda:0)
TARGET_F1=0.8
MAX_TIME=30
MAX_ITER=5000
N_REPS=2
N_RUNS=5
# Safety cap per invocation; 3 combos x 5 seeds x 4 passes x 30s = 30 min
# worst case, so this should never fire. Overridable from the environment;
# PER_INVOCATION_TIMEOUT=9999h effectively disables the cap.
PER_INVOCATION_TIMEOUT=9999h
echo "Invocation timeout: ${PER_INVOCATION_TIMEOUT}"

run_benchmark() {
  local device="$1" dim="$2"
  echo "=== device=${device} dim=${dim} admm optimized,minres,cudss (started $(date)) ==="
  timeout "${PER_INVOCATION_TIMEOUT}" python benchmark_projection.py \
    --device "${device}" \
    --dim "${dim}" \
    --target_f1 "${TARGET_F1}" \
    --max_time "${MAX_TIME}" \
    --max_iter "${MAX_ITER}" \
    --n_reps "${N_REPS}" \
    --n_runs "${N_RUNS}" \
    --solvers admm \
    --modes optimized,minres,cudss \
    --save \
    || echo "!!! FAILED or TIMED OUT: device=${device} dim=${dim} (continuing, finished $(date))"
}

for device in "${DEVICES[@]}"; do
  for dim in "${DIMS[@]}"; do
    run_benchmark "${device}" "${dim}"
  done
done

echo "=== ADMM sweep finished $(date) ==="
