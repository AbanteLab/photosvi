#!/usr/bin/env bash
# Null cohorts under the full-rank global block, to measure interval coverage.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
# Outputs go outside the repo -- see .gitignore. Override with PHOTOSVI_RUNS.
PHOTOSVI_RUNS=${PHOTOSVI_RUNS:-runs}
OUT=${OUT:-$PHOTOSVI_RUNS/runs_cohort_fr}
N=${N:-24}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GUIDE=fullrank
# Which physical GPUs this study may use. GPUS is a comma-separated list and the
# seeds are dealt round-robin over it.
#
# DEFAULT IS GPU 1 ALONE, since 2026-09-14: cuda:0 is left free for another user's
# work on this machine. Memory was never the constraint -- a cohort fit takes 1.2 GB
# of 49 -- but compute is, and fits sharing an SM with someone else's training slow
# both. Restore with GPUS=0,1 only when told the card is free again.
GPUS=${GPUS:-1}
IFS=',' read -ra _GPU_LIST <<< "$GPUS"
_pick() { echo "${_GPU_LIST[$(( $1 % ${#_GPU_LIST[@]} ))]}"; }

LOG="logs/cohort_fr_$(date +%Y%m%d-%H%M).log"
mkdir -p logs "$OUT"
{
  echo "[start] $(date -Is)  n=$N  out=$OUT  guide=fullrank"
  i=0
  # RESUME HAZARD. The check below is "has this seed written its result yet", which
  # is NOT the same as "is this seed being worked on": a fit in flight has no output
  # file, so relaunching while one is running starts a SECOND copy of it on the same
  # seed. Both will finish and the last to write wins, silently. Before relaunching,
  # wait for the previous run to drain:
  #     while pgrep -cf 'bin/python scripts/cohort_hierarchical\.py' >/dev/null; do sleep 60; done
  # Match on 'bin/python <script>' and not on the script name alone: the xargs
  # `sh -c` wrapper carries the same path in ITS command line, so a looser pattern
  # counts two processes per fit -- and matches the waiting shell itself, which is
  # how a previous chain waited forever on a queue that had already drained.
  for s in $(seq 0 $((N-1))); do
    [ -f "$OUT/hier_0.00_$(printf %04d $s).json" ] || echo "0.0 $s cuda:$(_pick $i)"
    i=$((i+1))
  done | xargs -P 6 -n 3 sh -c "OUT=$OUT GUIDE=fullrank \"$PY\" scripts/cohort_hierarchical.py \$0 \$1 \$2"
  echo "[done ] $(date -Is)  cohorts=$(ls "$OUT" | wc -l)"
} 2>&1 | tee "$LOG"
