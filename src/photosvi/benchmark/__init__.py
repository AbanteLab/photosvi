"""Detection benchmark harness: score event detectors against known onsets.

The Stage 0 harness of the matched-filter experiment, promoted from notebook to package.
One interface for three detectors (matched filter, GuPPy, a fitted posterior q), event-
matched PR/recall/F1 plus tolerance-free per-bin ROC, a real-grounded synthetic generator,
and responsive plotly views. Typical use:

    from photosvi.benchmark import run_benchmark, make_synth
    from photosvi.models.points import bi_exponential_kernel

    k_fit  = bi_exponential_kernel(125, k_rise=2.0, k_decay=1.0)
    k_true = bi_exponential_kernel(125, k_rise=2.0, k_decay=1.3)   # faster -> deliberate mismatch
    y, onsets = make_synth(k_true, seed=1)
    res = run_benchmark(y, onsets, k_fit, inj_t=15_000)
    res.table()               # metrics DataFrame (Styler)
    res.dashboard().show()    # PR curve + trace panel, responsive

Plotting requires plotly; the core harness (detectors, metrics, synth) does not.
"""

from photosvi.benchmark.detectors import (
    calibrate_lag,
    guppy_detect,
    matched_filter_score,
    nms,
    peaks_from_score,
    zpeaks_detect,
)
from photosvi.benchmark.harness import BenchmarkResult, DetectorOutcome, run_benchmark
from photosvi.benchmark.metrics import event_metrics, match_events, perbin_auc, pr_curve
from photosvi.benchmark.synth import make_synth

__all__ = [
    "BenchmarkResult",
    "DetectorOutcome",
    "calibrate_lag",
    "event_metrics",
    "guppy_detect",
    "make_synth",
    "match_events",
    "matched_filter_score",
    "nms",
    "peaks_from_score",
    "perbin_auc",
    "pr_curve",
    "run_benchmark",
    "zpeaks_detect",
]
