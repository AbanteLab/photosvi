"""Run every detector on one trace and score them -- the practical entry point.

`run_benchmark(y, onsets, kernel, inj_t=...)` scores the matched filter and GuPPy (plus
any extra per-bin detectors, e.g. a fitted posterior q) and returns a `BenchmarkResult`
carrying the per-detector metrics, PR curves, and detections. The result object exposes
the plots directly (`res.table()`, `res.pr_figure()`, `res.dashboard()`), so the plotly
dependency is only imported when a figure is actually requested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from photosvi.benchmark.detectors import (
    calibrate_lag,
    guppy_detect,
    matched_filter_score,
    nms,
    peaks_from_score,
    zpeaks_detect,
)
from photosvi.benchmark.metrics import event_metrics, perbin_auc, pr_curve
from photosvi.data.loader import DEFAULT_FS


@dataclass
class DetectorOutcome:
    """One detector's result on one trace."""

    name: str
    times: np.ndarray            # detected event bins (raw, before lag correction)
    scores: np.ndarray           # per-detection score
    lag: int                     # calibrated onset offset (bins)
    metrics: dict                # pr_auc, recall_at, best_f1, n_pred, n_true, (perbin_roc)
    curve: tuple                 # (recall, precision) arrays
    perbin: np.ndarray | None    # per-bin score if the detector produces one, else None


@dataclass
class BenchmarkResult:
    """All detectors on one trace, plus the data needed to plot them."""

    outcomes: dict               # name -> DetectorOutcome, insertion-ordered
    y: np.ndarray                # [T] trace
    onsets: np.ndarray           # true event bins
    tol: int                     # matching tolerance (bins)
    fs: float

    def table(self):
        from photosvi.benchmark.plots import metrics_table
        return metrics_table(self)

    def pr_figure(self):
        from photosvi.benchmark.plots import pr_figure
        return pr_figure(self)

    def trace_figure(self, **kw):
        from photosvi.benchmark.plots import trace_figure
        return trace_figure(self, **kw)

    def dashboard(self, **kw):
        from photosvi.benchmark.plots import dashboard
        return dashboard(self, **kw)


def _as_1d(y) -> np.ndarray:
    if hasattr(y, "numpy"):
        y = y.detach().numpy() if hasattr(y, "detach") else y.numpy()
    y = np.asarray(y, dtype=float)
    return y[0] if y.ndim == 2 else y


def run_benchmark(
    y,
    onsets: np.ndarray,
    kernel,
    *,
    inj_t: int,
    fs: float = DEFAULT_FS,
    tol_s: float = 0.35,
    refractory: int = 8,
    detrend: bool = True,
    extra: dict | None = None,
) -> BenchmarkResult:
    """Score matched-filter + GuPPy (+ any `extra` per-bin detectors) on one trace.

    `extra` maps a detector name to an already-computed per-bin score array (e.g. a fitted
    q posterior), scored the same way as the matched filter. `tol_s` is the match tolerance
    in seconds -- keep it well under the inter-event interval or the metric degenerates.

    `refractory` is the non-maximum suppression window, in bins, and it is applied to EVERY
    detector. Until 2026-10-01 it was applied to none of them, which quietly favoured smooth
    scores: a matched filter is a convolution and has few local maxima by construction, while
    a per-bin posterior is rough and contributes a cluster per transient. On one 48 Hz cell
    the matched filter offered 1409 candidates and the fitted posterior 33704, for 427 true
    events -- so precision was being compared across candidate sets differing 24-fold, for
    reasons that have nothing to do with either detector's quality. `nms`'s own docstring
    says it has to be predeclared and identical across everything compared; now it is.
    """
    y_np = _as_1d(y)
    T = len(y_np)
    tol = int(round(tol_s * fs))
    onsets = np.asarray(onsets)

    # How far a detector's timestamps may sit from the onset before the offset is treated
    # as a real miss rather than a known lag. It has to cover the KERNEL's peak, because a
    # raw-peak detector like GuPPy fires there by construction: its detections are late by
    # exactly that much. `calibrate_lag`'s default of 40 bins is 1.6 s at 25 Hz and only
    # 0.83 s at 48, while K9 peaks at 1.23 s -- so on a 48 Hz benchmark with a slow kernel
    # every GuPPy detection fell outside the window and scored as a miss, putting its F1
    # at 0.13 even where the matched filter got 0.86. That was a measurement artefact
    # about to be published as a claim about someone else's method.
    lag_cap = int(np.argmax(np.asarray(_as_1d(kernel)))) + int(round(0.5 * fs))

    mf = matched_filter_score(y, kernel, inj_t, dt=1.0 / fs, detrend=detrend)
    def _cand(score):
        return nms(*peaks_from_score(score), refractory=refractory)

    registry: dict[str, tuple] = {
        "matched-filter": (mf, *_cand(mf)),
        "guppy": (None, *nms(*guppy_detect(y_np, fs=fs), refractory=refractory)),
        # The Bonaventura lab's prominence rule on the z-scored trace (2026-10-20).
        "z-peaks": (None, *nms(*zpeaks_detect(y_np, inj_t, fs=fs), refractory=refractory)),
    }
    if extra:
        for name, score in extra.items():
            score = np.asarray(score, dtype=float)
            registry[name] = (score, *_cand(score))

    outcomes: dict[str, DetectorOutcome] = {}
    for name, (perbin, pt, ps) in registry.items():
        lag = calibrate_lag(pt, onsets, max_lag=lag_cap)
        m = event_metrics(pt, ps, onsets, tol, lag)
        m["lag"] = lag
        if perbin is not None:
            m["perbin_roc"] = round(perbin_auc(perbin, onsets, T), 3)
        outcomes[name] = DetectorOutcome(
            name=name,
            times=np.asarray(pt),
            scores=np.asarray(ps),
            lag=lag,
            metrics=m,
            curve=pr_curve(pt, ps, onsets, tol, lag),
            perbin=perbin,
        )
    return BenchmarkResult(outcomes=outcomes, y=y_np, onsets=onsets, tol=tol, fs=fs)
