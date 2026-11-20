#!/usr/bin/env bash
# The single-trace detection benchmark at 30 seeds per SNR level, recording the tonic
# (2026-10-27). Same model and simulator as runs_final_detection (configs/
# final_detection_48.yaml); seeds 0-9 reproduce those runs, 10-29 are new. CPU only, so
# every cell runs on the device the 10-seed runs used; niced, and no new cell starts while
# the CPU is at or above HOT degrees.
#
#   scripts/run_detection_tonic.sh [OUT] [PARALLEL]
#
# WAIT_FOR=<pattern> holds the launch until no process matches the pattern (the
# production two-stage fits, say).
set -u
OUT=${1:-${PHOTOSVI_RUNS:-runs}/runs_final_detection_t30_tonic}
P=${2:-24}
HOT=${HOT:-88}
PY=${PY:-python}
DCFG=configs/final_detection_48.yaml
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
log() { echo "[dettonic $(date '+%m-%d %H:%M')] $*" | tee -a "$OUT/log_launch.txt"; }
cpu_t() { for h in /sys/class/hwmon/hwmon*; do
  [ "$(cat $h/name 2>/dev/null)" = k10temp ] && echo $(( $(cat $h/temp1_input)/1000 )); done; }

if [ -n "${WAIT_FOR:-}" ]; then
  log "waiting for '$WAIT_FOR' to finish"
  while pgrep -f "$WAIT_FOR" > /dev/null; do sleep 120; done
fi
log "start: P=$P, out $OUT"
for seed in $(seq 0 29); do for s in 2.0 3.0 4.0; do
  [ -f "$OUT/bm_$(printf %.2f $s)_$(printf %02d $seed).json" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 20; done
  while [ "$(cpu_t)" -ge "$HOT" ]; do sleep 60; done
  OMP_NUM_THREADS=1 PYTHONPATH=src BM_CONFIG=$DCFG OUT=$OUT nice -n 10 \
    $PY scripts/benchmark_guppy.py $seed $s cpu >> "$OUT/log_runs.txt" 2>&1 &
done; done
wait
log "all done: $(ls $OUT/bm_*.json | wc -l) cells"
