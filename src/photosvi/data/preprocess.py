"""Fiber-photometry preprocessing: raw two-channel recording to corrected dF/F.

A port of the pipeline the recordings in this project were produced with, kept in
`vendor/bonaventura_pipeline/GREEN_ISO_awake_inj.m` so the two can be compared rather
than described. Only the preprocessing half is here. That script continues into peak
detection and five-minute binning, which is a different analysis of the same data and
is exactly what this package replaces -- porting it would ship two answers to one
question.

The pipeline, in order:

1. smooth the isosbestic control with a moving average
2. smooth the biosensor channel with a Savitzky-Golay filter
3. trim a second from each end, where the acquisition has artefacts
4. fit an asymmetric-least-squares baseline to the biosensor channel
5. regress that baseline on the control over the PRE-INJECTION window only, and use
   the fitted control as F0
6. dF/F = (F - F0) / F0
7. downsample to the analysis rate

Step 5 is the part worth understanding. The isosbestic channel tracks everything that
is not the biosensor -- movement, bleaching, fibre coupling -- but arrives on its own
scale, so it has to be mapped onto the biosensor's before it can be divided out. The
map is fitted on the pre-injection window because that is the stretch where the drug
contributes nothing, and fitting it over the whole recording would let the drug
response pull the very baseline it is meant to be measured against.

**Not verified against the original.** The MATLAB reads Doric HDF5 through a
vendor-supplied extractor and we have no `.doric` file, so `read_doric` is deliberately
absent and the numerical core below is what can be tested today. The recordings this
project holds were also probably produced by a slightly earlier version of that script,
so agreement is a thing to measure once raw files exist, not a thing to assume.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.signal import savgol_filter

__all__ = [
    "als_baseline",
    "correct_isosbestic",
    "downsample",
    "moving_average",
    "savgol_seconds",
]

# Defaults transcribed from the MATLAB, in seconds rather than samples so they mean
# the same thing at any acquisition rate. The script writes them as `0.5*sr`,
# `0.1*sr`, `600*sr`, which is the same intent expressed in the units it had.
ISO_SMOOTH_S = 0.5
GREEN_SMOOTH_S = 0.1
TRIM_S = 1.0
FLATZONE_S = 600.0
# The script's `lambda = sr * 1e4`, confirmed against the .mat files, which store it:
# `lambda = 480000` at `sr = 48`. This said 1e6 until 2026-10-01 -- a hundredfold
# stiffer baseline than the original. Measured on the five recordings of the final
# dataset, the correction moves the six calibration statistics by 0.1% to 2.7% and the
# peak-to-noise anchor by 0.4%: at this lambda the ALS baseline is already stiff enough
# that making it stiffer changes little, because the signal carries little at the scale
# lambda controls. So nothing measured before this is invalidated, but the value was
# wrong and the comment asserted something false about the original.
ALS_LAMBDA_PER_FS = 1e4
ALS_P = 0.01
ALS_ITERS = 10


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Centred running mean, matching MATLAB `conv(y, ones(w)/w, 'same')`.

    The ends are convolved against fewer samples in both implementations, so the
    first and last `window // 2` points are biased toward zero. That is why the
    pipeline trims a second from each end afterwards rather than before.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window == 1:
        return np.asarray(y, dtype=float).copy()
    k = np.ones(window) / window
    return np.convolve(np.asarray(y, dtype=float), k, mode="same")


def savgol_seconds(y: np.ndarray, fs: float, seconds: float, order: int = 2
                   ) -> np.ndarray:
    """Savitzky-Golay over a window given in seconds.

    MATLAB's `smoothdata(y, 'sgolay', w)` takes a window LENGTH and uses a quadratic
    by default; scipy wants an odd window and an explicit order, so the length is
    rounded up to odd here. A window shorter than the polynomial order is not a
    smoother, so it is refused rather than silently widened.
    """
    w = int(round(seconds * fs))
    if w % 2 == 0:
        w += 1
    if w <= order:
        raise ValueError(
            f"a {seconds} s window at {fs} Hz is {w} samples, which cannot carry an "
            f"order-{order} polynomial; smooth before downsampling, not after")
    return savgol_filter(np.asarray(y, dtype=float), w, order)


def als_baseline(y: np.ndarray, lam: float, p: float = ALS_P,
                 n_iter: int = ALS_ITERS) -> np.ndarray:
    """Asymmetric least squares baseline, ported from the MATLAB `als_baseline`.

    Minimises `w * (y - z)^2 + lam * ||D2 z||^2` with weights that are `p` where the
    signal sits above the current baseline and `1 - p` where it sits below, iterated.
    With `p` small the baseline is pulled toward the troughs and ignores peaks, which
    is what makes it a baseline for a signal whose events are one-sided.

    `lam` controls stiffness and scales with the sampling rate, because the second
    difference it penalises is per sample: the MATLAB passes `sr * 1e6`, so a
    recording at twice the rate gets twice the smoothing parameter and the same
    baseline in seconds.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.size
    if n < 3:
        raise ValueError("need at least 3 samples for a second-difference penalty")
    d = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    dtd = lam * (d.T @ d)
    w = np.ones(n)
    z = y.copy()
    for _ in range(n_iter):
        wd = sp.diags(w, 0, format="csc")
        z = spla.spsolve((wd + dtd).tocsc(), w * y)
        w = p * (y > z) + (1.0 - p) * (y <= z)
    return z


def downsample(y: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Block-mean decimation to `fs_out`.

    Not in the MATLAB -- the recordings arrive at 25 Hz having been decimated from the
    acquisition rate by a step we do not have. A block mean is the conservative
    assumption: it is what most pipelines do and it is linear, so it commutes with the
    dF/F ratio's numerator. If the real step turns out to be a decimating filter plus
    subsampling, the difference is a further mild low-pass and this is the milder of
    the two.

    **This is a second smoother stacked on the Savitzky-Golay**, and at these rates it
    is the one that matters: 0.1 s of Savitzky-Golay at 1205 Hz is 2.5 samples at
    25 Hz, below the resolution the decimation leaves. Transients are the thing this
    package estimates, so which smoothing the data has already had is not a detail.
    """
    if fs_out > fs_in:
        raise ValueError(f"cannot upsample: {fs_in} Hz to {fs_out} Hz")
    factor = fs_in / fs_out
    if abs(factor - round(factor)) > 1e-6:
        raise ValueError(
            f"{fs_in} Hz to {fs_out} Hz is a factor of {factor:.4f}, not an integer; "
            "resampling at a non-integer ratio needs interpolation and a decision "
            "about which, so it is not made here")
    k = int(round(factor))
    y = np.asarray(y, dtype=float).reshape(-1)
    usable = (y.size // k) * k
    return y[:usable].reshape(-1, k).mean(axis=1)


def correct_isosbestic(
    green: np.ndarray,
    isosbestic: np.ndarray,
    fs: float,
    *,
    inj_t: int,
    iso_smooth_s: float = ISO_SMOOTH_S,
    green_smooth_s: float = GREEN_SMOOTH_S,
    trim_s: float = TRIM_S,
    flatzone_s: float = FLATZONE_S,
    als_lambda: float | None = None,
    als_p: float = ALS_P,
    smooth_green: bool = True,
) -> dict:
    """The pipeline through dF/F, returning the trace and what produced it.

    `inj_t` is the injection sample in the INPUT indexing. The window the isosbestic
    scaling is fitted on runs from the start of the trimmed recording to whichever comes
    first, `flatzone_s` or the injection -- so the map cannot be pulled by the response it
    exists to measure. It is returned shifted by the trim so a caller can keep track.

    Until 2026-09-30 the window was always the first `flatzone_s` seconds and `inj_t` was
    only bounds-checked, which is fine for a recording injected after ten minutes and
    silently wrong for one injected before. Found while asking what depends on the
    injection time in the Bonaventura files, where it happens not to bite: the dF/F there
    is bit-identical for any `inj_t`.

    `smooth_green=False` skips the Savitzky-Golay. The original applies it because it
    goes on to call `findpeaks`, which needs a signal without sample-scale noise. This
    package estimates the event train under a noise model instead, so the smoothing
    removes evidence it would otherwise use -- available, not recommended, and the
    default matches the original so that a comparison is possible at all.

    Returns a dict with `dff`, the fitted `f0`, the ALS `baseline`, the regression
    coefficients, and the shifted `inj_t`.
    """
    green = np.asarray(green, dtype=float).reshape(-1)
    isosbestic = np.asarray(isosbestic, dtype=float).reshape(-1)
    if green.size != isosbestic.size:
        raise ValueError(
            f"channels differ in length: green {green.size}, isosbestic "
            f"{isosbestic.size}")

    con = moving_average(isosbestic, max(1, int(round(iso_smooth_s * fs))))
    g = savgol_seconds(green, fs, green_smooth_s) if smooth_green else green.copy()

    trim = int(round(trim_s * fs))
    if green.size <= 2 * trim + 3:
        raise ValueError("recording is shorter than the trim from both ends")
    sl = slice(trim, green.size - trim)
    g, con = g[sl], con[sl]
    inj_shifted = inj_t - trim

    lam = fs * ALS_LAMBDA_PER_FS if als_lambda is None else als_lambda
    baseline = als_baseline(g, lam, als_p)

    flat = int(round(flatzone_s * fs))
    if flat > g.size:
        raise ValueError(
            f"the pre-injection window is {flatzone_s} s but the recording is "
            f"{g.size / fs:.0f} s after trimming")
    if not 0 < inj_shifted <= g.size:
        raise ValueError(f"inj_t {inj_t} is outside the trimmed recording")
    # Fit the control onto the ALS baseline over the pre-injection stretch only. The
    # drug contributes nothing there, so the map cannot be pulled by the response it
    # is meant to let us measure.
    win = min(flat, inj_shifted)
    if win < 3:
        raise ValueError(
            f"the pre-injection window is {win} samples ({win / fs:.1f} s): inj_t "
            f"{inj_t} leaves nothing before the injection to fit the control on")
    a, b = np.polyfit(con[:win], baseline[:win], 1)
    f0 = con * a + b
    if np.any(f0 == 0):
        raise ValueError("the scaled control crosses zero; dF/F would divide by it")
    return dict(dff=(g - f0) / f0, f0=f0, baseline=baseline, slope=float(a),
                intercept=float(b), inj_t=inj_shifted, fs=fs)
