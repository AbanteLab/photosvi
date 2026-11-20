#!/usr/bin/env bash
# The non-null cells of the cohort grid: what tab:cohort and tab:estimation are for,
# and without which both argue the reverse of what is true.
#
# At delta = 0 an estimator that compresses toward zero is accurate by construction,
# because zero is the right answer. On the null cell alone the GuPPy-count arm has the
# smallest bias and RMSE in the table while getting the SIGN of the per-session effect
# wrong on 406 of 480 sessions. Only a non-zero effect separates a short ruler from an
# accurate one -- and the same applies to the crossed/uncrossed comparison, where
# partial pooling shrinks toward zero at no cost when zero is the truth.
#
# Two arms in sequence rather than one interleaved queue: the uncrossed arm needs
# DESIGN and N_ANIMALS in its environment, and passing per-line environment through
# xargs means an empty field for the crossed arm silently shifts every argument.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
PHOTOSVI_RUNS=${PHOTOSVI_RUNS:-runs}
N=${N:-20}
JOBS=${JOBS:-4}
DELTAS=${DELTAS:-0.20 0.40 0.70}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# GPU 1 alone; cuda:0 is another user's. See run_cohort_uncrossed.sh.
GPUS=${GPUS:-1}
IFS=',' read -ra _GPU_LIST <<< "$GPUS"
_pick() { echo "${_GPU_LIST[$(( $1 % ${#_GPU_LIST[@]} ))]}"; }

# RESUME HAZARD. The check below asks whether a seed has WRITTEN its result, not
# whether it is being worked on: a fit in flight has no output file, so relaunching
# while one runs starts a second copy of it and the last to write wins. Drain first:
#   while pgrep -cf 'bin/python scripts/cohort_hierarchical\.py' >/dev/null; do sleep 60; done
# matching on 'bin/python <script>', since the xargs wrapper carries the same path and
# a looser pattern counts two per fit and matches the waiting shell itself.
emit_cells() {                      # $1 = output dir
  local i=0
  for d in $DELTAS; do
    for s in $(seq 0 $((N-1))); do
      [ -f "$1/hier_$(printf %.2f "$d")_$(printf %04d "$s").json" ] || \
        printf '%s %s cuda:%s\n' "$d" "$s" "$(_pick $i)"
      i=$((i+1))
    done
  done
}

LOG="logs/cohort_grid_$(date +%Y%m%d-%H%M).log"
mkdir -p logs
{
  echo "[start] $(date -Is)  n=$N  deltas=$DELTAS  jobs=$JOBS  gpus=$GPUS"

  CROSSED=$PHOTOSVI_RUNS/runs_cohort_frz
  mkdir -p "$CROSSED"
  echo "[arm  ] crossed -> $CROSSED"
  emit_cells "$CROSSED" | xargs -P "$JOBS" -n 3 sh -c \
      "OUT=$CROSSED \"$PY\" scripts/cohort_hierarchical.py \$0 \$1 \$2"

  UNCROSSED=$PHOTOSVI_RUNS/runs_cohort_uncrossed
  mkdir -p "$UNCROSSED"
  echo "[arm  ] uncrossed -> $UNCROSSED"
  emit_cells "$UNCROSSED" | xargs -P "$JOBS" -n 3 sh -c \
      "OUT=$UNCROSSED DESIGN=uncrossed N_ANIMALS=12 \"$PY\" scripts/cohort_hierarchical.py \$0 \$1 \$2"

  echo "[done ] $(date -Is)"
} 2>&1 | tee "$LOG"
