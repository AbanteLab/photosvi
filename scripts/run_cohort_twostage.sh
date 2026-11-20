#!/usr/bin/env bash
# The two-stage arm: per-session single-trace fits, one process per cohort.
#
# CPU fan-out, because each fit is B = 1 and 24 of them run at once here against one
# on a card. One process per COHORT rather than per session: the forward sampler
# walks the time axis in Python (3 s per session), so regenerating the cohort once
# per session would add ~72 h of pure simulation to the study.
#
# Resumable at session granularity -- an interrupted cohort picks up where it left
# off rather than refitting the sessions it already wrote.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
# Outputs go outside the repo -- see .gitignore. Override with PHOTOSVI_RUNS.
PHOTOSVI_RUNS=${PHOTOSVI_RUNS:-runs}
OUT=${OUT:-$PHOTOSVI_RUNS/runs_cohort}
JOBS=${JOBS:-24}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

LOG="logs/cohort_ts_$(date +%Y%m%d-%H%M).log"
mkdir -p logs "$OUT"

{
  echo "[start] $(date -Is)  host=$(hostname)  jobs=$JOBS  out=$OUT"
  echo "[git  ] $(git rev-parse --short HEAD) $(git rev-parse --abbrev-ref HEAD)"
  "$PY" - "$OUT" <<'P' | xargs -P "$JOBS" -n 2 sh -c "OUT=$OUT DEVICE=cpu \"$PY\" scripts/cohort_twostage.py \$0 \$1"
import os, sys, yaml
out = sys.argv[1]
cfg = yaml.safe_load(open("configs/cohort_comparison.yaml"))
reps, n_sess = cfg["grid"]["n_replicates"], cfg["cohort"]["n_animals"] * 3
for d in cfg["grid"]["delta"]:
    for s in range(reps.get(f"{d:.2f}", reps["default"])):
        done = sum(os.path.exists(f"{out}/ts_{d:.2f}_{s:04d}_{k:02d}.json")
                   for k in range(n_sess))
        if done < n_sess:
            print(d, s)
P
  echo "[done ] $(date -Is)  sessions=$(ls "$OUT"/ts_*.json 2>/dev/null | wc -l)"
} 2>&1 | tee "$LOG"
