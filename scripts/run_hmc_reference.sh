#!/usr/bin/env bash
# Five datasets, each fit twice: variational and NUTS, same model, same data.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
# Outputs go outside the repo -- see .gitignore. Override with PHOTOSVI_RUNS.
PHOTOSVI_RUNS=${PHOTOSVI_RUNS:-runs}
OUT=${OUT:-$PHOTOSVI_RUNS/runs_hmc}
SEEDS=${SEEDS:-"0 1 2 3 4"}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOG="logs/hmc_$(date +%Y%m%d-%H%M).log"
mkdir -p logs "$OUT"
{
  echo "[start] $(date -Is)  seeds='$SEEDS'  out=$OUT"
  echo "$SEEDS" | tr ' ' '\n' | xargs -P 5 -I{} \
    "$PY" scripts/hmc_reference.py {} --T 8000 --beta0 -2.2 --inj-frac 0.5 \
        --samples 500 --warmup 500 --max-tree-depth 8 --svi-steps 8000 --out "$OUT"
  echo "[done ] $(date -Is)  cells=$(ls "$OUT" | wc -l)"
} 2>&1 | tee "$LOG"
