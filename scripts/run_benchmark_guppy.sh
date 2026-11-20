#!/usr/bin/env bash
# GuPPy benchmark: 5 noise levels x 10 seeds, our model against the field-standard
# two-threshold MAD detector, scored by one routine at one tolerance.
#
# Re-run 2026-09-08 on the fixed model (analytic Bernoulli KL, tied+gated marks,
# lr 1e-2, plateau stopping). The previous pass used a fixed 15000 steps at lr 1e-3,
# which the coverage re-run showed to be short -- so the numbers it produced are the
# old optimiser's, not the model we intend to publish.
#
# Resumable: a cell whose JSON already exists is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
# DEVICE=cpu is deliberate, not a default nobody revisited. Each cell is an
# independent B = 1 fit, and measured 2026-09-08 the two are comparable per fit
# (~29 ms/step CPU vs ~35 ms/step GPU at T = 60000) while 24 CPU fits run at once
# against one on the card. Set DEVICE=cuda to run cells on the GPU anyway -- correct,
# just slower in aggregate. See src/photosvi/inference/points/device.py.
DEVICE=${DEVICE:-cpu}
JOBS=${JOBS:-24}
# Outputs go outside the repo -- see .gitignore. Override with PHOTOSVI_RUNS.
PHOTOSVI_RUNS=${PHOTOSVI_RUNS:-runs}
OUT=${OUT:-$PHOTOSVI_RUNS/runs_benchmark_v2}

# Pin the math backends HERE, in the shell. os.environ.setdefault inside Python runs
# after `import torch` has already built its OpenMP pool and is silently ineffective.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

LOG="logs/benchmark_guppy_$(date +%Y%m%d-%H%M).log"
mkdir -p logs "$OUT"

{
  echo "[start] $(date -Is)  host=$(hostname)  jobs=$JOBS  out=$OUT  device=$DEVICE"
  echo "[git  ] $(git rev-parse --short HEAD) $(git rev-parse --abbrev-ref HEAD)"
  "$PY" - <<'P' | xargs -P "$JOBS" -n 2 sh -c "OUT=$OUT DEVICE=$DEVICE \"$PY\" scripts/benchmark_guppy.py \$0 \$1"
import yaml
c = yaml.safe_load(open("configs/benchmark_guppy.yaml"))
for s in c["grid"]["sigma_y"]:
    for d in c["grid"]["seeds"]:
        print(d, s)
P
  echo "[done ] $(date -Is)  cells=$(ls "$OUT" | wc -l)"
} 2>&1 | tee "$LOG"
