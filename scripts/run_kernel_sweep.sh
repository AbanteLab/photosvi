#!/usr/bin/env bash
# Kernel-basis sweep: does a mean-free phasic basis (sum(h) = 0) unbias beta_inj?
#
# Three configs, identical but for the kernel basis, 15 cells each (5 arms x 3 seeds):
#
#   nonneg      gen biexp    / fit biexp      status quo: non-negative kernel, sum(h) > 0
#   k5          gen K5       / fit K5         well-specified: is the mean-free basis better?
#   k5_misspec  gen biexp    / fit K5         physical truth, mean-free fit: the real-data case
#
# K5 is the biphasic kernel selected 2026-09-07 from diagnostics/kernel_candidates.png:
# a fast positive lobe minus a scaled slower one, so sum(h) = 0 and h returns to zero.
#
# Resumable: a cell whose JSON already exists is skipped, so re-running after an
# interruption picks up where it stopped. Pass --force in WORKER_ARGS to recompute.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
JOBS=${JOBS:-24}                 # 32 cores, one thread per worker (see the export below)

# Pin the math backends HERE, in the shell, not in Python. `pin_threads()` uses
# os.environ.setdefault, which runs long after `import torch` has already built its
# OpenMP pool -- measured: 51 threads and 210% CPU per worker, load 94 on 32 cores.
# OMP reads these at library init, so they only bite if they are set before exec.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
MODES=${MODES:-"nonneg k5 k5_misspec"}
# Outputs go outside the repo -- see .gitignore. Override with PHOTOSVI_RUNS.
PHOTOSVI_RUNS=${PHOTOSVI_RUNS:-runs}
OUT=${OUT:-$PHOTOSVI_RUNS/runs_kernel}
LOG="logs/kernel_sweep_$(date +%Y%m%d-%H%M).log"
mkdir -p logs "$OUT"

{
  echo "[start] $(date -Is)  host=$(hostname)  jobs=$JOBS  modes=$MODES"
  echo "[git  ] $(git rev-parse --short HEAD) $(git rev-parse --abbrev-ref HEAD)"
  for m in $MODES; do
    cfg="configs/points_guide_sweep_${m}.yaml"
    echo "=== $m -> $OUT/$m  ($(date -Is)) ==="
    "$PY" -m photosvi.cli.main points sweep --config "$cfg" --print-grid \
      | xargs -P "$JOBS" -I{} sh -c "\"$PY\" -m photosvi.cli.main points sweep \
          --config \"$cfg\" {} --out \"$OUT/$m\" --quiet"
    echo "=== $m done: $(ls -1 "$OUT/$m" 2>/dev/null | wc -l)/15 cells ==="
  done
  echo "[end  ] $(date -Is)"
} 2>&1 | tee "$LOG"
