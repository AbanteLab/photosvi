#!/usr/bin/env bash
# The hierarchical arm of the two-stage comparison: one fit per simulated cohort.
#
# GPU, because the per-step cost is flat in batch size (measured 2026-09-08: 55 ms at
# B = 16 as at B = 1), so a 24-session cohort costs what a single trace costs. The
# same fit on CPU is ~24x that.
#
# Both cards are used and several processes share each: a fit is launch-bound and
# leaves the card mostly idle, so concurrency should be close to linear. PER_GPU is
# the knob -- watch nvidia-smi and the per-cell wall times before raising it.
#
# Resumable: a cohort whose JSON already exists is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
# Outputs go outside the repo -- see .gitignore. Override with PHOTOSVI_RUNS.
PHOTOSVI_RUNS=${PHOTOSVI_RUNS:-runs}
OUT=${OUT:-$PHOTOSVI_RUNS/runs_cohort}
GPUS=${GPUS:-"0 1"}
PER_GPU=${PER_GPU:-3}
JOBS=$(( $(echo $GPUS | wc -w) * PER_GPU ))

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

LOG="logs/cohort_hier_$(date +%Y%m%d-%H%M).log"
mkdir -p logs "$OUT"

{
  echo "[start] $(date -Is)  host=$(hostname)  gpus='$GPUS' per_gpu=$PER_GPU jobs=$JOBS out=$OUT"
  echo "[git  ] $(git rev-parse --short HEAD) $(git rev-parse --abbrev-ref HEAD)"

  "$PY" - "$GPUS" "$OUT" <<'P' | xargs -P "$JOBS" -n 3 sh -c "OUT=$OUT \"$PY\" scripts/cohort_hierarchical.py \$0 \$1 \$2"
import os, sys, yaml
gpus = sys.argv[1].split()
out = sys.argv[2]
cfg = yaml.safe_load(open("configs/cohort_comparison.yaml"))
reps = cfg["grid"]["n_replicates"]
i = 0
for d in cfg["grid"]["delta"]:
    n = reps.get(f"{d:.2f}", reps["default"])
    for s in range(n):
        if os.path.exists(f"{out}/hier_{d:.2f}_{s:04d}.json"):
            continue
        print(d, s, f"cuda:{gpus[i % len(gpus)]}")
        i += 1
P
  echo "[done ] $(date -Is)  cohorts=$(ls "$OUT"/hier_*.json 2>/dev/null | wc -l)"
} 2>&1 | tee "$LOG"
