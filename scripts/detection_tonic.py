"""The tonic side of the single-trace detection benchmark: what each method says the slow
injection response was.

One summary for every curve, so that the methods are compared on the same two numbers:

  * peak height -- the maximum of the curve after the injection minus its mean over the
    pre-injection reference stretch (the lab's baseline window, 300 s to the injection);
  * peak time -- seconds from the injection to that maximum.

The curves are the simulated truth, PhotoSVI's fitted tonic x_hat(t), and the Bonaventura
lab's two slow components of the raw trace (process_merged_data.m): a causal 4th-order
Butterworth low-pass at 0.1 Hz applied with `lfilter`, and the asymmetric-least-squares
baseline (lambda 1e6 * fs, p 0.01: the lab's 1e6 scaled with the sampling rate, as in the
paper's preprocessing, since the lab applies it to smoothed, decimated traces). Both are applied to the trace in its own units, which for
the simulator are the dF/F units the lab filters; no z-score.
"""
from __future__ import annotations

import numpy as np

PRE_START_S = 300.0   # start of the pre-injection reference stretch, as the lab's baseline
LP_HZ, LP_ORDER = 0.1, 4
ALS_LAM_PER_HZ, ALS_P = 1e6, 0.01


def lowpass(y: np.ndarray, fs: float) -> np.ndarray:
    """The lab's low-pass: causal Butterworth, order 4, 0.1 Hz, `lfilter` from rest."""
    from scipy.signal import butter, lfilter
    b, a = butter(LP_ORDER, LP_HZ / (fs / 2), "low")
    return lfilter(b, a, y)


def als(y: np.ndarray, fs: float) -> np.ndarray:
    """The lab's ALS baseline (lambda 1e6 * fs, p 0.01, 10 reweightings)."""
    from photosvi.experiments.tonic_benchmark import als_baseline
    return als_baseline(np.asarray(y, dtype=float), lam=ALS_LAM_PER_HZ * fs, p=ALS_P)


def summarize(curve: np.ndarray, inj: int, fs: float) -> dict:
    """Peak height above the pre-injection level, and the peak's time after injection."""
    curve = np.asarray(curve, dtype=float)
    pre = float(curve[int(round(PRE_START_S * fs)):inj].mean())
    k = int(np.argmax(curve[inj:]))
    return {"pre_level": pre, "peak_height": float(curve[inj + k] - pre),
            "peak_time_s": k / fs}


def area_rel_err(tonic: dict, method: str) -> float:
    """Relative error (%) of a method's area under the tonic response, from the stored curves.

    The area is that of the curve above its mean from PRE_START_S to the injection, from the
    injection to the end of the recording, against the same area of the simulated tonic. It
    is the lab's AUC measure over the whole post-injection window. Computed from the 1-s
    curves of the record, so it needs no refit; the PhotoSVI curve is its fitted tonic only.
    """
    st = tonic["curve_step_s"]
    inj, pre0 = int(round(tonic["curve_inj_s"] / st)), int(round(PRE_START_S / st))

    def area(x):
        x = np.asarray(x, dtype=float)
        return float((x[inj:] - x[pre0:inj].mean()).sum() * st)

    return 100 * (area(tonic["curves"][method]) / area(tonic["curves"]["true"]) - 1)


def downsample(curve: np.ndarray, fs: float, step_s: float = 1.0) -> list[float]:
    """Every `step_s` seconds, starting at t = 0 (sample, not average: the curves are slow)."""
    return [float(v) for v in np.asarray(curve, dtype=float)[::int(round(step_s * fs))]]


def bin_mean(y: np.ndarray, fs: float, step_s: float = 1.0) -> list[float]:
    """The raw trace averaged over `step_s` bins, for drawing behind the slow curves."""
    n = int(round(step_s * fs))
    y = np.asarray(y, dtype=float)
    return [float(v) for v in y[:y.size // n * n].reshape(-1, n).mean(1)]


def tonic_record(y: np.ndarray, x_true: np.ndarray, x_fit: np.ndarray, true_params: dict,
                 fit_params: dict, inj: int, fs: float, step_s: float = 1.0) -> dict:
    """Everything the JSON keeps about the tonic, for one recording."""
    lp, base = lowpass(y, fs), als(y, fs)
    curves = {"true": x_true, "photosvi": x_fit, "lowpass": lp, "als": base}
    return {
        "true_params": true_params,
        "fit_params": fit_params,
        "summary": {n: summarize(c, inj, fs) for n, c in curves.items()},
        "curve_step_s": step_s,
        "curve_inj_s": inj / fs,
        "curves": {n: downsample(c, fs, step_s) for n, c in curves.items()}
                  | {"data": bin_mean(y, fs, step_s)},
    }
