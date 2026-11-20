#!/usr/bin/env bash
# The paper's final benchmark runs (frozen model, tag paper-final-v1, 2026-10-21).
#
#   bash scripts/run_final_benchmarks.sh smoke        # 1 cell per stage, into $ROOT/smoke_*
#   bash scripts/run_final_benchmarks.sh production   # detection -> cohort mid -> cohort low/high
#
# Resumable: every arm skips the files it has already written, so re-running the same
# command after an interruption finishes what is missing and nothing else.
# GPU: cuda:1 ONLY (cuda:0 belongs to someone else). CPU fits are niced; the two-stage
# fits are paused by the thermal guard at >= 90 C.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
DEV=${DEV:-${PHOTOSVI_RUNS:-runs}}
LOG=${LOG:-$DEV/final_benchmarks.log}
DCFG=configs/final_detection_48.yaml
CCFG=configs/final_cohort_48.yaml
GPU_MAX=${GPU_MAX:-7}       # hierarchical fits at once on cuda:1
CPU_MAX=${CPU_MAX:-24}      # single-thread CPU fits at once (two-stage, detection)
MODE=${1:-production}
# Every model setting comes from the configs: clear the overrides the earlier studies used.
unset MARK_MODE SIGMA_Y_PER_SESSION TONIC_BLOCK MARK_PRIOR MARK_PRIOR_MU_SD MARK_PRIOR_LS_SD \
      MARK_INIT FIX_MARKS MARK_GATE_FLOOR TAU BETA_PRIOR SWAP_DRUGS GUIDE COUPLE SESSIONS \
      SIM_SIGMA_CELL SIM_SIGMA_ANIMAL SIM_SIGMA_B0SES

say() { echo "[final $(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
n_gpu() { ps -eo args | grep -cE "^$PY -u scripts/cohort_hierarchical.py .* cuda:1$"; }
n_cpu() { ps -eo args | grep -cE "^$PY (-u )?scripts/(cohort_twostage.py|benchmark_guppy.py) "; }
wait_gpu() { while [ "$(n_gpu)" -ge "$GPU_MAX" ]; do sleep 30; done; }
wait_cpu() { while [ "$(n_cpu)" -ge "$CPU_MAX" ]; do sleep 30; done; }
sy_of() { case $1 in mid) echo 0.016757;; high) echo 0.011171;; low) echo 0.022343;; esac; }

manifest() {  # dir
  mkdir -p "$1"
  { echo "date: $(date -Iseconds)"
    echo "code_sha: $(git rev-parse HEAD)"
    echo "code_tag: $(git describe --tags --exact-match 2>/dev/null || echo none)"
    echo "code_dirty: $(git status --porcelain --untracked-files=no | wc -l)"
    echo "python: $($PY -c 'import sys; print(sys.version.split()[0])')"
    $PY -c 'import torch, pyro, numpy, scipy; print(f"torch: {torch.__version__}\npyro: {pyro.__version__}\nnumpy: {numpy.__version__}\nscipy: {scipy.__version__}")'
    echo "detection_config_sha256: $(sha256sum $DCFG | cut -d' ' -f1)"
    echo "cohort_config_sha256: $(sha256sum $CCFG | cut -d' ' -f1)"
  } > "$1/MANIFEST.txt"
}

# ------------------------------------------------------------------ detection
det_cell() {  # out sigma seed
  [ -f "$1/bm_$(printf %.2f $2)_$(printf %02d $3).json" ] && return
  wait_cpu
  OMP_NUM_THREADS=1 BM_CONFIG=$DCFG OUT=$1 nice -n 10 \
    $PY scripts/benchmark_guppy.py $3 $2 cpu >> "$1/log.txt" 2>&1 &
  sleep 2
}
fig_trace() {  # out sigma seed
  [ -f "$1" ] && return
  wait_cpu
  OMP_NUM_THREADS=2 nice -n 10 $PY scripts/paper_fig_fit_data.py --config $DCFG \
    --sigma-y $2 --seed $3 --out "$1" >> "$(dirname $1)/log_traces.txt" 2>&1 &
  sleep 2
}
detection() {  # out seeds...
  local out=$1; shift
  manifest "$out"
  for seed in "$@"; do for s in 2.0 3.0 4.0; do det_cell "$out" $s $seed; done; done
  # The traces Fig. 2 and the lag numbers are drawn from: the benchmark cells at seed 0.
  mkdir -p "$out/traces"
  fig_trace "$out/traces/fd48_3.00.json" 3.0 0
  fig_trace "$out/traces/fit_2.00.json" 2.0 0
}

# ------------------------------------------------------------------ cohorts
hier() {  # cell design n levels delta seed level
  local f="$1/hier_$(printf %.2f $5)_$(printf %04d $6).json"
  [ -f "$f" ] && return
  wait_gpu
  OMP_NUM_THREADS=2 CONFIG=$CCFG DESIGN=$2 N_ANIMALS=$3 LEVELS=$4 SIM_SIGMA_Y=$(sy_of $7) \
    OUT=$1 DEVICE=cuda:1 $PY -u scripts/cohort_hierarchical.py $5 $6 cuda:1 >> "$1/log_hier.txt" 2>&1 &
  sleep 5
}
counts() {  # cell delta seed level  (GuPPy and z-peaks arms: seconds per cohort)
  for arm in guppy zpeaks; do
    OMP_NUM_THREADS=1 CONFIG=$CCFG DESIGN=crossed N_ANIMALS=5 SIM_SIGMA_Y=$(sy_of $4) OUT=$1 \
      nice -n 15 $PY scripts/cohort_twostage_$arm.py $2 $3 >> "$1/log_counts.txt" 2>&1
  done
}
twostage() {  # cell delta seed level
  for k in $(seq 0 9); do
    [ -f "$1/ts_$(printf %.2f $2)_$(printf %04d $3)_$(printf %02d $k).json" ] && continue
    wait_cpu
    OMP_NUM_THREADS=1 CONFIG=$CCFG DESIGN=crossed N_ANIMALS=5 SIM_SIGMA_Y=$(sy_of $4) SESSIONS=$k \
      OUT=$1 nice -n 15 $PY scripts/cohort_twostage.py $2 $3 cpu >> "$1/log_ts.txt" 2>&1 &
    sleep 2
  done
}
cohort_gpu() {  # level root seeds deltas [uncrossed]
  local lvl=$1 root=$2 seeds=$3 deltas=$4 unc=${5:-0}
  local cell=$root/crossed_n5; mkdir -p $cell; manifest $cell
  for s in $seeds; do
    for d in $deltas; do hier $cell crossed 5 animal,cell $d $s $lvl; done
    if [ "$unc" = 1 ]; then
      mkdir -p $root/uncrossed_n10; manifest $root/uncrossed_n10
      hier $root/uncrossed_n10 uncrossed 10 animal 0.0 $s $lvl
    fi
  done
}
cohort_cpu() {  # level root seeds deltas
  local lvl=$1 root=$2 seeds=$3 deltas=$4 cell=$2/crossed_n5
  mkdir -p $cell
  for s in $seeds; do for d in $deltas; do counts $cell $d $s $lvl; done; done
  for s in $seeds; do for d in $deltas; do twostage $cell $d $s $lvl; done; done
}

S50=$(seq 0 49 | tr '\n' ' '); S20=$(seq 0 19 | tr '\n' ' ')
D5="0.0 0.25 0.5 0.75 1.0"; D3="0.0 0.5 1.0"

case $MODE in
  smoke)
    R=$DEV/runs_final_smoke
    say "smoke: start ($(git rev-parse --short HEAD))"
    detection $R/detection 0
    cohort_gpu mid $R/cohort_mid "0" "0.0 1.0" 1
    cohort_cpu mid $R/cohort_mid "0" "0.0 1.0"
    wait; say "smoke: done";;
  production)
    say "production: start ($(git rev-parse --short HEAD), $(git describe --tags --exact-match 2>/dev/null))"
    # GPU (hierarchical) and CPU (detection, then two-stage) queues run side by side.
    ( cohort_gpu mid $DEV/runs_final_cohort_mid "$S50" "$D5" 1
      say "gpu: medium queued"
      cohort_gpu high $DEV/runs_final_cohort_high "$S20" "$D3"
      cohort_gpu low $DEV/runs_final_cohort_low "$S20" "$D3"
      say "gpu: low/high queued"; wait; say "gpu: all hierarchical done" ) &
    ( detection $DEV/runs_final_detection $(seq 0 9)
      while [ "$(ps -eo args | grep -c "^$PY scripts/benchmark_guppy.py")" -gt 0 ]; do sleep 60; done
      say "cpu: detection done"
      cohort_cpu mid $DEV/runs_final_cohort_mid "$S50" "$D5"
      say "cpu: medium queued"
      cohort_cpu high $DEV/runs_final_cohort_high "$S20" "$D3"
      cohort_cpu low $DEV/runs_final_cohort_low "$S20" "$D3"
      say "cpu: low/high queued"; wait; say "cpu: all two-stage done" ) &
    wait; say "production: ALL DONE";;
  *) echo "usage: $0 smoke|production"; exit 2;;
esac
