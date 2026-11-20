#!/usr/bin/env bash
# The two-stage arm refitted under the unbounded Normal prior.
#
# `cohort_comparison.yaml` still carries `beta_prior: uniform`, which reaches this arm
# and not the hierarchical one -- the hierarchical model's globals are Normal by
# construction and it does not take the setting. So the comparator, and only the
# comparator, is fitted under a prior whose reported width is a delta-method
# linearisation on a transformed axis. Measured in S1 Table: 0.1525 against the wide
# Normal's 0.1711, an 11% narrowing that leaves the point estimate untouched.
#
# The direction matters. It makes the arm the hierarchy has to beat look MORE precise
# than it is, on the axis the paper compares.
#
# Waits for the machine rather than competing with it: the power grid owns the GPU and
# the kernel sweep the CPU, and this is 528 CPU fits that nothing is blocked on.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
PHOTOSVI_RUNS=${PHOTOSVI_RUNS:-runs}
OUT=${OUT:-$PHOTOSVI_RUNS/runs_cohort_ts_normal}
JOBS=${JOBS:-16}
SEEDS=${SEEDS:-19}
LOADMAX=${LOADMAX:-12}          # one-minute load average to wait below

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG="logs/cohort_ts_normal_$(date +%Y%m%d-%H%M).log"
mkdir -p logs "$OUT"
{
  echo "[queued] $(date -Is)  waiting for load below $LOADMAX"
  while true; do
    load=$(awk '{print int($1)}' /proc/loadavg)
    [ "$load" -lt "$LOADMAX" ] && break
    sleep 300
  done
  echo "[start ] $(date -Is)  load $(awk '{print $1}' /proc/loadavg)  jobs=$JOBS"
  echo "[prior ] normal (overriding the config's uniform for this arm only)"
  seq 0 "$SEEDS" | xargs -P "$JOBS" -n 1 sh -c \
    "OUT=$OUT DEVICE=cpu BETA_PRIOR=normal \"$PY\" scripts/cohort_twostage.py 0.00 \$0"
  echo "[done  ] $(date -Is)  sessions=$(ls "$OUT"/ts_*.json 2>/dev/null | wc -l)"
} 2>&1 | tee "$LOG"
