"""Event detectors reduced to one interface: a per-bin score, its local maxima.

Three ways to score "how event-like is this bin", all producing discrete detections
via `peaks_from_score`, so they can be matched against known onsets on equal footing:

- `matched_filter_score` -- correlate the (detrended) trace with the transient kernel;
  optimal linear detector for a known shape in white noise. Peaks *at* the onset.
- `guppy_detect` -- GuPPy's two-threshold MAD rule per chunk; a robust local
  "is this bump big vs nearby noise". Peaks at the transient *peak* (a lag after onset).
- `zpeaks_detect` -- the prominence rule of the Bonaventura lab's pipeline ("z-peaks"):
  Savitzky-Golay smoothing, z-score against the pre-injection segment, every peak with
  prominence >= 1.6 z. Peaks at the transient *peak*, like GuPPy.
- (a fitted posterior `q` is passed in as an already-computed per-bin score.)

`calibrate_lag` recovers each detector's systematic onset offset so matching is fair.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import argrelextrema, find_peaks, savgol_filter

from photosvi.data.loader import DEFAULT_FS
from photosvi.models.points import fit_tonic_baseline

# Samples of pre-injection recording needed before its scale is worth estimating.
# Two seconds at 25 Hz is already 50; this is deliberately generous, because falling
# back to the whole trace is a known bias and should happen only when the alternative
# is dividing by the standard deviation of a handful of points.
_MIN_BASELINE = 250


def matched_filter_score(
    y: torch.Tensor,
    kernel: torch.Tensor,
    inj_t: int,
    *,
    dt: float = 1.0 / DEFAULT_FS,
    detrend: bool = True,
    scale: str = "baseline",
) -> np.ndarray:
    """Per-bin matched-filter z: correlate the detrended residual with `kernel`, standardize.

    `y` is a torch tensor [T] or [1, T]; returns a numpy [T] score. Unlike the notebook
    `classical_init` warm-start this does **not** clamp to logits, so the ranking of strong
    peaks is preserved -- essential for PR/ROC scoring. With `detrend` the tonic baseline is
    fit and subtracted (real traces); without, only the median is removed (pure synthetic).

    `scale` chooses what the score is standardized against.

    * ``"baseline"`` (default since 2026-09-14) uses the PRE-INJECTION segment only.
    * ``"trace"`` uses the whole recording, which is what this function did before and
      what every result committed up to that date was produced with.

    **Why the default changed.** Standardizing by the whole trace puts the drug's own
    response into the denominator, so a strong response suppresses the score everywhere
    -- including in the baseline, where the events being counted are the denominator of
    the estimand. The bias therefore runs one way: the larger the true rate increase,
    the emptier the baseline looks, and the larger the apparent rate ratio.

    Measured on 22 real opioid recordings. Under ``"trace"``, five sessions had ZERO
    baseline detections at the shipped threshold and all five were opioid sessions;
    under ``"baseline"`` none did, the sparsest having 19. Vehicle sessions barely
    moved (ratio of post- to pre-injection score scale 0.96-1.08) while opioid sessions
    moved a great deal (1.16-3.13). That asymmetry is the signature: lowering the
    threshold would have moved every session, and only the affected ones move here.

    The baseline segment needs enough samples to estimate a scale from; below
    `_MIN_BASELINE` this falls back to the whole trace rather than dividing by noise.
    """
    yt = y if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.float32)
    if yt.dim() == 1:
        yt = yt[None]
    T = yt.shape[-1]
    M = int(kernel.numel())
    if detrend:
        x0 = fit_tonic_baseline(yt, inj_t, dt=dt).baseline(T, inj_t).detach()
        resid = yt - x0
    else:
        resid = yt - yt.median()
    mf = F.conv1d(F.pad(resid.unsqueeze(1), (0, M - 1)), kernel.view(1, 1, M)).squeeze(1)
    if scale not in ("baseline", "trace"):
        raise ValueError(f"scale must be 'baseline' or 'trace' (got {scale!r})")
    ref = mf[:, :inj_t] if (scale == "baseline" and inj_t >= _MIN_BASELINE) else mf
    z = (mf - ref.median()) / (ref.std() + 1e-6)
    return z[0].numpy()


def guppy_detect(
    y: np.ndarray,
    *,
    fs: float = DEFAULT_FS,
    window_s: float = 15.0,
    high_amp_filt: float = 2.0,
    transients_thresh: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Faithful GuPPy two-threshold MAD transient rule, per non-overlapping chunk.

    Per `window_s` chunk: `thr1 = median + high_amp_filt*MAD` excludes high-amplitude
    samples so they do not inflate the noise estimate; on the remainder
    `thr2 = filtMedian + transients_thresh*MAD`; interior local maxima above thr2 are
    transients. Returns (times, scores) where score is the GuPPy amplitude peak-filtMedian.
    Reimplements LernerLab/GuPPy `transients.py`; confirm defaults against the install.
    """
    y = np.asarray(y, dtype=float)
    n = int(round(window_s * fs))
    times: list[int] = []
    scores: list[float] = []
    for start in range(0, len(y), n):
        seg = y[start:start + n]
        if seg.size < 3:
            continue
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        if mad == 0:
            continue
        keep = seg[seg <= med + high_amp_filt * mad]
        if keep.size < 3:
            keep = seg
        fmed = np.median(keep)
        fmad = np.median(np.abs(keep - fmed))
        if fmad == 0:
            continue
        thr = fmed + transients_thresh * fmad
        for m in argrelextrema(seg, np.greater)[0]:
            if seg[m] > thr:
                times.append(start + int(m))
                scores.append(float(seg[m] - fmed))
    return np.asarray(times, dtype=int), np.asarray(scores, dtype=float)


def zpeaks_detect(
    y: np.ndarray,
    inj_t: int,
    *,
    fs: float = DEFAULT_FS,
    smooth_s: float = 0.1,
    polyorder: int = 2,
    min_prominence: float = 1.6,
) -> tuple[np.ndarray, np.ndarray]:
    """The Bonaventura lab's transient rule ("z-peaks"), returning (times, prominences).

    Reimplements `GREEN_ISO_awake_inj.m` of BonaventuraLab/fiber-photometry (Zenodo
    10.5281/zenodo.17657389), after its dF/F step:

    * the sensor trace is smoothed with a Savitzky-Golay filter over `smooth_s` seconds
      (MATLAB `smoothdata(..., 'sgolay', 0.1*sr)`, whose default degree is 2). The
      pipeline smooths the raw sensor channel before dF/F; on a dF/F trace -- all a
      simulation has -- the filter is applied to the dF/F, which for a linear filter and
      a slowly varying denominator is the same up to that denominator;
    * z = (dF/F - median(pre)) / std(pre), with `pre` the samples before the injection
      and the sample standard deviation (MATLAB `std`, N - 1);
    * every peak with prominence >= `min_prominence` (MATLAB `findpeaks(...,
      'MinPeakProminence', 1.6)`, no width or distance constraint) is a transient,
      scored by its prominence.

    `scipy.signal.find_peaks` defines prominence as MATLAB does (height above the higher
    of the two lowest points between the peak and the nearest higher peak, or the end of
    the signal, on either side) and, like it, excludes the first and last sample and
    reports the middle of a flat peak. The window is rounded to the nearest odd number of
    samples, at least `polyorder + 1`; MATLAB's edge handling of `smoothdata` and scipy's
    `mode="interp"` differ in the first and last half-window only.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    win = max(int(round(smooth_s * fs)), polyorder + 1)
    if win % 2 == 0:
        win += 1
    if win <= y.size:
        y = savgol_filter(y, win, polyorder, mode="interp")
    pre = y[:inj_t] if inj_t >= 2 else y
    sd = float(np.std(pre, ddof=1))
    if not np.isfinite(sd) or sd == 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    z = (y - np.median(pre)) / sd
    loc, props = find_peaks(z, prominence=min_prominence)
    return loc.astype(int), np.asarray(props["prominences"], dtype=float)


def peaks_from_score(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interior local maxima of a per-bin score -> (times, scores at those times)."""
    score = np.asarray(score, dtype=float)
    m = argrelextrema(score, np.greater)[0]
    return m, score[m]


def nms(
    times: np.ndarray, scores: np.ndarray, *, refractory: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy non-maximum suppression: keep the best peak, drop its neighbours, repeat.

    `peaks_from_score` returns every interior local maximum, so a single transient in a
    noisy score contributes a cluster of them. Without suppression that cluster is
    scored as several detections, which inflates the prediction count and depresses
    precision for reasons that have nothing to do with the detector's quality.

    `refractory` is in bins and is a *scoring* choice, not a biological claim: it should
    be predeclared and identical across everything being compared. Returns the survivors
    in time order.
    """
    times = np.asarray(times)
    scores = np.asarray(scores, dtype=float)
    if len(times) == 0:
        return times, scores
    keep, taken = [], np.zeros(len(times), dtype=bool)
    for i in np.argsort(-scores):
        if taken[i]:
            continue
        keep.append(i)
        taken |= np.abs(times - times[i]) <= refractory
    order = np.sort(np.asarray(keep, dtype=int))
    return times[order], scores[order]


def calibrate_lag(pred_t: np.ndarray, onsets: np.ndarray, *, max_lag: int = 40) -> int:
    """Detector's systematic onset offset = median(detection - nearest onset), robust window.

    Matched filter lands ~on the onset (lag ~0); a raw-peak detector like GuPPy lands a
    kernel-peak lag later. Subtracting this before matching keeps the comparison fair.
    """
    pred_t = np.asarray(pred_t)
    onsets = np.asarray(onsets)
    if len(pred_t) == 0 or len(onsets) == 0:
        return 0
    d = np.array([p - onsets[np.argmin(np.abs(p - onsets))] for p in pred_t])
    d = d[np.abs(d) < max_lag]
    return int(np.median(d)) if len(d) else 0
