#!/usr/bin/env bash
# Null cohorts on the UNCROSSED design, with the shipped guide.
#
# The other extreme of the design space. In the crossed cohort every animal receives
# every drug, so the animal effect enters both terms of the opioid contrast and
# cancels; what is left setting its width is the animal x drug interaction. Here each
# animal receives vehicle and exactly ONE opioid, so the opioid-versus-opioid
# contrast is between-animal: nothing cancels, sigma_animal enters directly, and the
# arithmetic floor on the contrast's standard error rises from the interaction term
# to sigma_animal * sqrt(2/6) = 0.14 with six animals per opioid group.
#
# HELD FIXED: 24 recordings, the same number the crossed arm fits (8 x 3 against
# 12 x 2). A cohort costs recordings, so that is what a design comparison has to hold
# constant; matching animal counts instead would confound the design with the sample
# size. Same seeds, same generative truth, same guide -- everything but the assignment
# of drugs to animals.
#
# Compare against runs_cohort_frz with scripts/paper_fig_designs.py.
#
# NOTE for anything that waits on this finishing: `pgrep -f <script>` matches the
# WAITER's own command line too. Count the python processes, not the pattern:
#     pgrep -c -f 'scripts/cohort_hierarchical\.py'
# from a shell whose own command line does not contain the pattern.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
PHOTOSVI_RUNS=${PHOTOSVI_RUNS:-runs}
OUT=${OUT:-$PHOTOSVI_RUNS/runs_cohort_uncrossed}
N=${N:-24}
JOBS=${JOBS:-6}
# GUIDE is deliberately unset: the config names the shipped family and this arm is
# about the design, not about the guide. Setting it here would be a second thing
# changed at once.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DESIGN=uncrossed N_ANIMALS=12
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

LOG="logs/cohort_uncrossed_$(date +%Y%m%d-%H%M).log"
mkdir -p logs "$OUT"
{
  echo "[start] $(date -Is)  n=$N  out=$OUT  design=uncrossed animals=12"
  echo "[guide] $("$PY" -c "import yaml;print(yaml.safe_load(open('configs/cohort_comparison.yaml'))['fit']['hier_beta_guide'])")"
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
      "OUT=$OUT DESIGN=uncrossed N_ANIMALS=12 \"$PY\" scripts/cohort_hierarchical.py \$0 \$1 \$2"
  echo "[done ] $(date -Is)  cohorts=$(ls "$OUT" | wc -l)"
} 2>&1 | tee "$LOG"
