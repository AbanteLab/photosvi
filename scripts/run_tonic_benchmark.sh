#!/usr/bin/env bash
# The tonic-response cohort benchmark (photosvi.experiments.tonic_benchmark), all four arms.
#
#   ROOT=<run dir> SEED_LO=0 SEED_HI=49 bash scripts/run_tonic_benchmark.sh
#
# GPU (cuda:1 ONLY; cuda:0 is not ours) for the hierarchical fits, GPU_MAX at a time; CPU
# for the rest: the FLMM and post-pre summaries (cheap, sequential) and the two-stage
# single-trace fits, one process per session, CPU_MAX at a time. Done files are skipped, so
# the script can be re-run to resume. Extra fit settings reach every cell through
# FIT_SET, e.g. FIT_SET="tonic_block=true" or, for a smoke test,
# FIT_SET="stop_min_steps=3000 max_steps=6000".
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
CONFIG=${CONFIG:-configs/cohort_tonic_48.yaml}
ROOT=${ROOT:?set ROOT}
SEED_LO=${SEED_LO:-0}; SEED_HI=${SEED_HI:-49}
DPS=${DPS:-"0.00 0.25 0.50 0.75 1.00"}
GPU_MAX=${GPU_MAX:-6}; CPU_MAX=${CPU_MAX:-10}
LOG=$ROOT/logs; mkdir -p $LOG
sets=(); for kv in ${FIT_SET:-}; do sets+=(--set "$kv"); done
$PY scripts/cohort_tonic_cells.py --config $CONFIG --root $ROOT "${sets[@]}" > /dev/null
gpu_running() { ps -eo args | grep -c "^$PY -u scripts/cohort_hierarchical.py"; }
cpu_running() { ps -eo args | grep -c "^$PY scripts/cohort_twostage.py"; }

hier() {  # dp seed
  local cell=$ROOT/dp$1
  [ -f "$cell/hier_0.00_$(printf %04d $2).json" ] && return
  while [ "$(gpu_running)" -ge $GPU_MAX ]; do sleep 30; done
  OMP_NUM_THREADS=2 CONFIG=$cell/config_dp$1.yaml DESIGN=crossed N_ANIMALS=5 LEVELS=animal,cell \
    OUT=$cell DEVICE=cuda:1 $PY -u scripts/cohort_hierarchical.py 0.0 $2 cuda:1 \
    >> $LOG/hier_dp$1.log 2>&1 &
  sleep 5
}
ts() {  # dp seed
  local cell=$ROOT/dp$1
  for k in $(seq 0 9); do
    [ -f "$cell/ts_0.00_$(printf %04d $2)_$(printf %02d $k).json" ] && continue
    while [ "$(cpu_running)" -ge $CPU_MAX ]; do sleep 30; done
    OMP_NUM_THREADS=1 CONFIG=$cell/config_dp$1.yaml DESIGN=crossed N_ANIMALS=5 SESSIONS=$k \
      OUT=$cell nice -n 15 $PY scripts/cohort_twostage.py 0.0 $2 cpu >> $LOG/ts_dp$1.log 2>&1 &
    sleep 2
  done
}
summ() {  # dp seed
  local cell=$ROOT/dp$1
  [ -f "$cell/tonic_$1_$(printf %04d $2).json" ] && return
  OMP_NUM_THREADS=1 CONFIG=$cell/config_dp$1.yaml DESIGN=crossed N_ANIMALS=5 OUT=$cell \
    nice -n 15 $PY scripts/cohort_tonic_summary.py $1 $2 >> $LOG/summary.log 2>&1
}

( for s in $(seq $SEED_LO $SEED_HI); do for d in $DPS; do summ $d $s; done; done
  for s in $(seq $SEED_LO $SEED_HI); do for d in $DPS; do ts $d $s; done; done; wait ) &
for s in $(seq $SEED_LO $SEED_HI); do for d in $DPS; do hier $d $s; done; done
wait
$PY scripts/cohort_tonic_report.py --runs $ROOT --json $ROOT/report.json
