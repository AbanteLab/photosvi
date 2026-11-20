#!/usr/bin/env bash
# Null cohorts without the animal x drug interaction.
#
# `z_cell` has exactly one session per coefficient in this design -- 24 cells, 24
# sessions, no (animal, drug) pair repeated -- and the real cohort is the same, so the
# interaction is indistinguishable from session noise on the drug effect. It shows:
# sigma_cell comes back at 0.054 against a simulated 0.10.
#
# The point is not simplicity. Dropping a level does not delete the variation it
# carried, it reassigns it, and the natural destination is sigma_animal -- which is
# the term that sets the floor on the contrast's interval. If it lands there, the
# interval widens for the right reason, which is the opposite of what the extended
# block did when it widened by taking variance OUT of sigma_animal.
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
OUT=${OUT:-$PHOTOSVI_RUNS/runs_cohort_nocell}
N=${N:-24}
JOBS=${JOBS:-6}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GUIDE=fullrank LEVELS=animal,b0_session
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

LOG="logs/cohort_nocell_$(date +%Y%m%d-%H%M).log"
mkdir -p logs "$OUT"
{
  echo "[start] $(date -Is)  n=$N  out=$OUT  guide=fullrank levels=animal,b0_session"
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
  done | xargs -P "$JOBS" -n 3 sh -c \
      "OUT=$OUT GUIDE=fullrank LEVELS=animal,b0_session \"$PY\" scripts/cohort_hierarchical.py \$0 \$1 \$2"
  echo "[done ] $(date -Is)  cohorts=$(ls "$OUT" | wc -l)"
} 2>&1 | tee "$LOG"
