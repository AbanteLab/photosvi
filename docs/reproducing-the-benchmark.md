# Reproducing the detection benchmark

The claim that PhotoSVI detects transients more accurately than GuPPy and than a
prominence rule on the z-scored trace (z-peaks) can only be checked if it can be re-run.
This page gives the configuration, the seeds, the settings of each method and the
commands.

## What the benchmark does

Ninety simulated recordings, thirty seeds at each of three noise levels, with the length,
sampling rate and injection time of the experimental recordings (40 min at 48 Hz,
injection at 10 min). The transient-to-noise ratios are 2.9, 2.0 and 1.5, and the drug
doubles the event rate. The kernel and the autoregressive noise are those calibrated on
the experimental recordings. All three detectors are scored by one routine, at one
tolerance, after the same non-maximum suppression.

Everything is in `configs/final_detection_48.yaml`.

## Fairness

Four choices decide whether the comparison means anything, and all four are made
against PhotoSVI:

- **GuPPy and z-peaks run at their published defaults** (for GuPPy, 15 s chunks and
  2.0/3.0 MAD thresholds) and are not tuned on these traces.
- **The noise scale is estimated from each trace**, not taken from the simulation,
  as GuPPy estimates its own per chunk.
- **$F_1$ is taken at each method's best threshold**, so no method is penalised for its
  default calling rule.
- **GuPPy and z-peaks report the peak of a transient and PhotoSVI its onset.** Each
  detector's systematic offset is estimated on the same recordings and subtracted before
  matching, which gives every detector its best possible alignment.

## Run it

```bash
bash scripts/run_detection_tonic.sh OUT_DIR 24
```

The second argument is the number of parallel workers (one thread each, CPU only). Each
cell writes `bm_<sigma_y>_<seed>.json`, and a run that is interrupted resumes where it
stopped, because cells whose JSON exists are skipped.

## Read it

```bash
python scripts/paper_benchmark_numbers.py --runs OUT_DIR \
    --config configs/final_detection_48.yaml --out benchmark_numbers.tex
```

writes the numbers of the paper's detection table as LaTeX macros, so that no number
in the paper is typed by hand. The manuscript's `Makefile` lists the exact commands for
the figures as well.

## What to expect

| transient-to-noise ratio | PhotoSVI | GuPPy | z-peaks |
|---|---|---|---|
| 2.9 | 0.783 | 0.669 | 0.663 |
| 2.0 | 0.676 | 0.559 | 0.568 |
| 1.5 | 0.585 | 0.476 | 0.494 |

Mean $F_1$ over the thirty seeds. At a ratio of 2.0 and 90% precision, PhotoSVI recovers
51% of the transients, against 32% for both GuPPy and z-peaks.
