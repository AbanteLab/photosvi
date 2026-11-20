#!/usr/bin/env bash
# The cohort interval by leaving one animal out at a time.
#
# Unlike every other arm, this one does not try to make the posterior right: it
# estimates the estimator's sampling distribution directly, which is the quantity all
# the comparisons have been against. It corrects all three contrasts at once rather
# than the one direction a tilt is aimed at -- and the QQ grid says the
# opioid-versus-vehicle contrasts are the badly calibrated ones, 4.1x and 2.5x against
# the estimand's 1.9x.
#
# REFIT=10000 comes from a budget sweep: at 2000 the leave-one-out fits barely move
# from the full-cohort fit and the variance is an artefact of not having refitted; by
# 8000 the estimates have settled, and 8000 to 25000 moves them under 10%.
#
# `fullrank` correlates the eight globals and leaves the cohort interval 1.65x too
# narrow. The QQ grid shows each drug level is 2.5-3.6x too narrow on its own, and
# the block already holds them jointly -- so what is missing is the coupling to the
# 64 z latents, which is the channel by which between-animal spread reaches mu_drug.
# This arm puts them in the block: 72 dimensions, 2628 parameters, no refits.
#
# Compare against runs_cohort_fr8 (same design, same seeds, block over the globals
# only) with scripts/paper_fig_qq.py, which reports all three contrasts at once --
# unlike linear response, this corrects every direction rather than the one tilted.
# NOTE for anything that waits on this finishing: `pgrep -f <script>` matches the
# WAITER's own command line too, since the pattern appears in it. A chain written as
#     busy=$(pgrep -fc cohort_linear_response)
# never sees zero and waits forever -- which is exactly what happened, costing nine
# hours of idle GPU. Count the python processes, not the pattern:
#     pgrep -c -f 'scripts/cohort_hierarchical\.py'
# and check it from a shell whose own command line does not contain the pattern.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
PHOTOSVI_RUNS=${PHOTOSVI_RUNS:-runs}
OUT=${OUT:-$PHOTOSVI_RUNS/runs_cohort_jack}
N=${N:-24}
JOBS=${JOBS:-6}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GUIDE=fullrank REFIT=${REFIT:-10000}
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

LOG="logs/cohort_jack_$(date +%Y%m%d-%H%M).log"
mkdir -p logs "$OUT"
{
  echo "[start] $(date -Is)  n=$N  out=$OUT  guide=fullrank refit=$REFIT"
  i=0
  for s in $(seq 0 $((N-1))); do
    [ -f "$OUT/jack_0.00_$(printf %04d $s).json" ] || echo "$s cuda:$(_pick $i)"
    i=$((i+1))
  done | xargs -P "$JOBS" -n 2 sh -c \
      "OUT=$OUT GUIDE=fullrank REFIT=$REFIT \"$PY\" scripts/cohort_jackknife.py \$0 \$1"
  echo "[done ] $(date -Is)  cohorts=$(ls "$OUT" | wc -l)"
} 2>&1 | tee "$LOG"
