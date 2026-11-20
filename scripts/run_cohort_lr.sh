#!/usr/bin/env bash
# Linear response on the cohort contrast, over null cohorts.
#
# The question: the 8x8 global block reaches 70% coverage of the nominal 90%
# interval and stops. Does the post-hoc correction that recovers the exact width on
# a single trace close the rest? Both scales come out of the same fits, so coverage
# under each is measured on identical cohorts.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
PHOTOSVI_RUNS=${PHOTOSVI_RUNS:-runs}
OUT=${OUT:-$PHOTOSVI_RUNS/runs_cohort_lr}
N=${N:-24}
# Four concurrent fits per GPU is what the 8x8 cohort arm ran at; the LR refits are
# the same model at the same batch size, so the memory profile is unchanged.
JOBS=${JOBS:-6}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export GUIDE=${GUIDE:-fullrank} TILT_SDS=${TILT_SDS:-0.5,1.0} REFITS=${REFITS:-6000,15000}
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

LOG="logs/cohort_lr_$(date +%Y%m%d-%H%M).log"
mkdir -p logs "$OUT"
{
  echo "[start] $(date -Is)  n=$N out=$OUT guide=$GUIDE tilt_sds=$TILT_SDS refits=$REFITS"
  i=0
  for s in $(seq 0 $((N-1))); do
    [ -f "$OUT/lr_0.00_$(printf %04d $s).json" ] || echo "$s cuda:$(_pick $i)"
    i=$((i+1))
  done | xargs -P "$JOBS" -n 2 sh -c \
      "OUT=$OUT GUIDE=$GUIDE TILT_SDS=$TILT_SDS REFITS=$REFITS \"$PY\" scripts/cohort_linear_response.py \$0 \$1"
  echo "[done ] $(date -Is)  cohorts=$(ls "$OUT" | wc -l)"
} 2>&1 | tee "$LOG"
