#!/usr/bin/env bash
# The two-stage arm, refitted with the guide the config now names.
#
# WHY THIS EXISTS. Until 2026-09-14 `cohort_twostage.py` built its SingleTraceModel
# without passing `beta_guide`, so all 519 per-session fits of the study took the
# constructor's mean-field default. Measured against a known truth at this trace
# length, that family reports intervals 3.91x narrower than the estimator's actual
# spread (90% coverage: 22%); the fullrank block reports 1.50x (coverage 67%) with an
# unchanged point estimate. The two-stage arm is the comparator the hierarchical model
# has to beat, so it has to be beaten at its best, not at its default.
#
# SCOPE. The null cell only, seeds 0..21 -- the cohorts the baseline arm actually
# covers. The comparison is between two analyses of identical recordings, so refitting
# a seed the baseline never reached would compare the arms on different data. The
# baseline is also incomplete within those seeds (11 of 22 cohorts have all 24
# sessions; the rest are one or two short), which is worth knowing before any table
# is read: `cohort_report.py` should be pointed at the intersection, not the union.
#
# Writes to its own directory. runs_cohort/ is the record of what the paper currently
# claims and is left exactly as it is.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
PHOTOSVI_RUNS=${PHOTOSVI_RUNS:-runs}
OUT=${OUT:-$PHOTOSVI_RUNS/runs_cohort_ts_fullrank}
JOBS=${JOBS:-22}
SEEDS=${SEEDS:-21}          # inclusive upper bound
DELTA=${DELTA:-0.00}

# One thread per process: 22 fits run at once, and letting each grab the machine's
# cores makes all of them slower.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

LOG="logs/cohort_ts_fullrank_$(date +%Y%m%d-%H%M).log"
mkdir -p logs "$OUT"

{
  echo "[start] $(date -Is)  host=$(hostname)  jobs=$JOBS  out=$OUT"
  echo "[git  ] $(git rev-parse --short HEAD) $(git rev-parse --abbrev-ref HEAD)"
  echo "[guide] $("$PY" -c "import yaml;print(yaml.safe_load(open('configs/cohort_comparison.yaml'))['fit']['beta_guide'])")"
  seq 0 "$SEEDS" \
    | xargs -P "$JOBS" -n 1 sh -c "OUT=$OUT DEVICE=cpu \"$PY\" scripts/cohort_twostage.py $DELTA \$0"
  echo "[done ] $(date -Is)  sessions=$(ls "$OUT"/ts_*.json 2>/dev/null | wc -l)"
} 2>&1 | tee "$LOG"
