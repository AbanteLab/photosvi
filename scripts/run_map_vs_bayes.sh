#!/usr/bin/env bash
# MAP vs Bayesian tonic offsets: same cohorts, both ways, at T/4.
#
# Answers one question -- how much of the 3.3x posterior overconfidence comes from
# b and drift being point estimates rather than latents -- and answers it as a RATIO
# between two arms at fixed T, which does not need the full recording.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
# Outputs go outside the repo -- see .gitignore. Override with PHOTOSVI_RUNS.
PHOTOSVI_RUNS=${PHOTOSVI_RUNS:-runs}
OUT=${OUT:-$PHOTOSVI_RUNS/runs_offsets}
GPUS=${GPUS:-"0 1"}
PER_GPU=${PER_GPU:-3}
JOBS=$(( $(echo $GPUS | wc -w) * PER_GPU ))

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

LOG="logs/offsets_$(date +%Y%m%d-%H%M).log"
mkdir -p logs "$OUT"
{
  echo "[start] $(date -Is)  jobs=$JOBS  out=$OUT"
  echo "[git  ] $(git rev-parse --short HEAD)"
  "$PY" - "$GPUS" "$OUT" <<'P' | xargs -P "$JOBS" -n 3 sh -c "OUT=$OUT \"$PY\" scripts/map_vs_bayes_offsets.py \$0 \$1 \$2"
import os, sys, yaml
gpus = sys.argv[1].split(); out = sys.argv[2]
cfg = yaml.safe_load(open("configs/map_vs_bayes_offsets.yaml"))
n = cfg["grid"]["n_replicates"]["0.00"]
i = 0
# Interleave the arms so a partial run still has both, and alternate the cards so
# neither arm is systematically the one sharing a busier GPU.
for s in range(n):
    for arm in ("map", "bayes"):
        if os.path.exists(f"{out}/{arm}_{s:03d}.json"):
            continue
        print(arm, s, f"cuda:{gpus[i % len(gpus)]}")
        i += 1
P
  echo "[done ] $(date -Is)  cells=$(ls "$OUT" | wc -l)"
} 2>&1 | tee "$LOG"
