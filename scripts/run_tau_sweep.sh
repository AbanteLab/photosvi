#!/usr/bin/env bash
# Relaxation-temperature sweep, crossed with how the event factor is scored.
#
# tau = 0.5 was chosen rather than measured, and every result reported so far uses
# it. This is what turns it into a measured choice, and it fills the supplementary
# figure the manuscript already promises.
#
# Resumable: a cell whose JSON exists is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
# Outputs go outside the repo -- see .gitignore. Override with PHOTOSVI_RUNS.
PHOTOSVI_RUNS=${PHOTOSVI_RUNS:-runs}
OUT=${OUT:-$PHOTOSVI_RUNS/runs_tau}
JOBS=${JOBS:-12}
TAUS=${TAUS:-"0.05 0.10 0.20 0.35 0.50 0.75 1.00"}
SEEDS=${SEEDS:-"0 1 2 3 4"}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
LOG="logs/tau_sweep_$(date +%Y%m%d-%H%M).log"
mkdir -p logs "$OUT"
{
  echo "[start] $(date -Is)  jobs=$JOBS  taus='$TAUS'  seeds='$SEEDS'  out=$OUT"
  echo "[git  ] $(git rev-parse --short HEAD)"
  for t in $TAUS; do for kl in analytic relaxed; do for s in $SEEDS; do
    f=$(printf "%s/tau_%.3f_%s_%02d.json" "$OUT" "$t" "$kl" "$s")
    [ -f "$f" ] || echo "$t $kl $s"
  done; done; done | xargs -P "$JOBS" -n 3 sh -c "OUT=$OUT \"$PY\" scripts/tau_sweep.py \$0 \$1 \$2"
  echo "[done ] $(date -Is)  cells=$(ls "$OUT" | wc -l)"
} 2>&1 | tee "$LOG"
