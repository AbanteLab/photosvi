#!/usr/bin/env bash
# Real nociceptin cohort, both batches, final model (2026-10-21). Runs from this worktree's
# sources (PYTHONPATH) so the production benchmark in the main checkout is untouched.
#   (1) full: 7 animals x 2 conditions; (2) without DL56's morphine-only session.
set -u
cd "$(dirname "$0")/.."
WT=$PWD
PY=${PY:-python}
R=${R:-${PHOTOSVI_RUNS:-runs}/runs_real_final}
mkdir -p $R/logs
export PYTHONPATH=$WT/src OMP_NUM_THREADS=3 DETREND=0 MASK_MIN_S=120 MIN_STEPS=60000 STEPS=100000 \
       DEVICE=cuda:1 DESIGN=full LEVELS=animal,cell BATCHES=batch1,batch2
{
  echo "worktree $WT  branch $(git rev-parse --abbrev-ref HEAD)  sha $(git rev-parse HEAD)"
  echo "started $(date -Is)"
  echo "config configs/final_cohort_48.yaml sha256 $(sha256sum configs/final_cohort_48.yaml | cut -c1-16)"
  echo "kernel-config configs/final_detection_48.yaml sha256 $(sha256sum configs/final_detection_48.yaml | cut -c1-16)"
  echo "env DETREND=$DETREND MASK_MIN_S=$MASK_MIN_S MIN_STEPS=$MIN_STEPS STEPS=$STEPS LEVELS=$LEVELS BATCHES=$BATCHES"
  echo "ALS lambda per fs: $($PY -c 'from photosvi.data.bonaventura3 import ALS_LAMBDA_PER_FS as a; print(a)')"
  echo "per-session metadata (format, fs, injection source, sigma_y, mask) is written into each result JSON (sessions_meta)"
} > $R/MANIFEST.txt
run() {  # name exclude
  EXCLUDE=$2 $PY -u scripts/cohort_bonaventura3.py --config configs/final_cohort_48.yaml \
    --kernel-config configs/final_detection_48.yaml --out $R/$1.json > $R/logs/$1.log 2>&1
  echo "$1 exit $? $(date -Is)" >> $R/MANIFEST.txt
}
run full "" &
run full_noDL56sali "DL56:sali" &
wait
