#!/usr/bin/env bash
# The hierarchical benchmark over the number of animals, in the two-condition design the
# recordings have. One queue, worked in PRIORITY ORDER, so that a run cut short has finished
# the cells the paper most needs rather than a uniform fraction of all of them.
#
# Priority: the real cohort's own size first and both designs at it, then the sizes either
# side of it, then the extremes, then more seeds. The order is explicit below and is the
# only thing to edit when the priorities change.
#
# Measured 2026-10-02 on this machine: a cohort fit costs 0.112 s/step alone and 0.405 s/step
# six at a time, so six concurrent is 1.66x the throughput of one and the GPU is the limit,
# not the cohort size -- 6, 10 and 16 sessions all cost the same per step. Budgeting 20000
# steps a fit, the queue gets through about 2.7 fits an hour.
#
# Resumable: a cell whose JSON exists is skipped by the worker scripts themselves.
set -uo pipefail
cd "$(dirname "$0")/.."

# Pin the math backends HERE, in the shell, not in Python: OMP reads them at library init,
# so setting them after `import torch` is too late. Without this each fit took 25 cores --
# measured on this grid before it was added -- and six concurrent fits would have asked for
# 150 of the machine's 32, which is how the GPU ends up waiting on a thrashing CPU.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

PY=${PY:-python}
CFG=${CFG:-configs/cohort_power_48.yaml}
DEV=${DEV:-cuda:1}                   # cuda:0 belongs to another user on this machine
JOBS=${JOBS:-6}                      # concurrent fits; 6 measured as the throughput knee
SEEDS=${SEEDS:-12}
RUNS=${PHOTOSVI_RUNS:-runs}
OUT=${OUT:-$RUNS/runs_power}
mkdir -p "$OUT" logs

# Each line: <design> <n_animals> <delta> <seed-from> <seed-to>
# The real cohort is five animals, crossed, so that is what the first block measures.
queue() {
  local s=$((SEEDS - 1))
  for d in 0.0 0.4; do echo "crossed 5 $d 0 $s"; done          # the recordings' own size
  for d in 0.0 0.4; do echo "uncrossed 5 $d 0 $s"; done        # and the design comparison
  for n in 3 8; do for d in 0.0 0.4; do echo "crossed $n $d 0 $s"; done; done
  for n in 4 6 12; do for d in 0.0 0.4; do echo "crossed $n $d 0 $s"; done; done
  for d in 0.2 0.7; do echo "crossed 5 $d 0 $s"; done          # the effect-size axis, after
}

# All three arms, not just the hierarchical one. Until 2026-10-05 this grid ran
# `cohort_hierarchical.py` alone, so it could never have produced the comparison the
# benchmark section is: the two-stage and GuPPy arms were simply absent. They are run
# per cell and from the same CONFIG, DESIGN, N_ANIMALS and seed, which is what makes
# all three see the identical cohort (tests/test_cohort_arms_see_one_cohort.py).
#
# The split across hardware is the natural one: the hierarchical fit is GPU-bound and
# the other two are CPU-bound, so they run CONCURRENTLY rather than one after the
# other, and the CPU arms are niced below the GPU arm's host threads.
CPU_JOBS=${CPU_JOBS:-8}

queue | while read -r design n delta s0 s1; do
  # One directory per (design, N), with every delta inside it: cohort_report.py calibrates
  # each arm's threshold on that cell's own delta = 0 cohorts, so separating the deltas
  # would leave it nothing to calibrate against.
  cell="$OUT/${design}_n${n}"
  mkdir -p "$cell"
  echo "[cell ] $design n=$n delta=$delta seeds $s0..$s1  ($(date -Is))"

  seq "$s0" "$s1" | xargs -P "$CPU_JOBS" -I{} nice -n 15 sh -c \
    "CONFIG=$CFG DESIGN=$design N_ANIMALS=$n OUT=$cell \
     $PY scripts/cohort_twostage_guppy.py $delta {} >> logs/power_gp_${design}_n${n}.log 2>&1" &
  gp=$!
  seq "$s0" "$s1" | xargs -P "$CPU_JOBS" -I{} nice -n 15 sh -c \
    "CONFIG=$CFG DESIGN=$design N_ANIMALS=$n OUT=$cell DEVICE=cpu \
     $PY scripts/cohort_twostage.py $delta {} cpu >> logs/power_ts_${design}_n${n}.log 2>&1" &
  ts=$!
  seq "$s0" "$s1" | xargs -P "$JOBS" -I{} sh -c \
    "CONFIG=$CFG DESIGN=$design N_ANIMALS=$n OUT=$cell DEVICE=$DEV \
     $PY scripts/cohort_hierarchical.py $delta {} $DEV >> logs/power_${design}_n${n}.log 2>&1"
  wait "$gp" "$ts"
  echo "[done ] $design n=$n delta=$delta -> $(ls -1 "$cell" | wc -l) files"
done
