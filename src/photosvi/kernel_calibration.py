"""Calibrating the transient kernel to a sensor, from the user's own recordings.

The kernel h is fixed during a fit: it is the shape one event leaves in the trace, and the
model reads every transient through it. The shipped one was fitted to dLight1.3b in the
nucleus accumbens (K10, 285 isolated transients from five animals). A lab that
records another sensor -- or the same sensor in another region, where release and clearance
differ -- has a different shape, and the shape is not a detail: a too-slow kernel biased the
rate ratio upward by 19% and a too-fast one downward by 13%.

So the kernel is calibrated before the fit, by a procedure that does not involve the model.
It is the one that produced the shipped kernel, made into a function:

1. **Isolated transients.** Linear detrend and the 0.1-10 Hz band-pass of
   `experiments.marks`, then the peaks that stand clear of the noise with no other peak
   nearby, aligned on their maxima and scaled to peak one. Isolation matters because a
   peak-triggered average of overlapping transients is a sum of them, and a kernel fitted
   to it comes out too slow.
2. **Each candidate family fitted to their average**, through the same band-pass as the
   data -- the filter changes a transient's shape, so the kernel is compared to the data as
   the data would show it, never to the band-passed data directly. Every candidate keeps
   the two properties the model needs: sum(h) = 0, so the phasic carrier is mean-free and
   the event rate cannot compete with the tonic for the signal mean, and peak one, so a
   mark reads as the transient's height.
3. **Compared on held-out transients.** The family with the lowest held-out error is
   suggested. The shipped kernels are scored on the same folds, unfitted, so the user can
   see whether calibrating bought anything.

The suggestion is a suggestion. Which family wins on a residual and which one recovers the
rate in simulation are different questions; K7 and K8 were chosen between on the second,
so the interface shows every candidate against the data and lets the user choose.

What this cannot do: with few isolated transients the average is noise and the fit chases
it, so below `min_transients` nothing is fitted and the shipped kernel is suggested with a
warning. And the average is biased in a known direction -- peaks are selected on the noise
riding on them, which sharpens the apex slightly -- the same bias the shipped kernel has.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.signal import butter, detrend, filtfilt, find_peaks

from photosvi.models.points.events import _bigamma_raw, _bigamma_support, build_kernel

__all__ = [
    "SENSORS",
    "IsolatedTransients",
    "KernelCandidate",
    "KernelCalibration",
    "calibrate_kernel",
    "extract_isolated_transients",
]

BAND_HZ = (0.1, 10.0)

# Starting points, NOT measurements. A transient in a photometry trace is the sensor's
# response to a change in concentration that is itself shaped by release and clearance, so
# it is slower than the sensor's published on/off kinetics -- dLight1.3b switches off in
# about 0.1 s in vitro and its transients in the accumbens are 0.5 s wide. These values are
# the time to peak and the decay time constant of a typical in-vivo transient, rounded, and
# serve only to start the fit somewhere sensible; the fit moves away from them, and every
# family is also started from every other sensor's values.
SENSORS: dict[str, dict[str, float]] = {
    "dLight1.3b": dict(peak_s=0.45, decay_s=0.45),
    "GRAB-DA2m": dict(peak_s=0.35, decay_s=1.0),
    "GCaMP6f": dict(peak_s=0.12, decay_s=0.35),
    "GCaMP6s": dict(peak_s=0.3, decay_s=1.2),
    "jGCaMP8m": dict(peak_s=0.06, decay_s=0.15),
}

# The shipped dLight1.3b kernel of the paper, K10, in seconds so it builds at
# any sampling rate. It is also the API's default (`api.DEFAULT_KERNEL`, kept equal by a
# test; not imported, because the API imports this module).
K10 = dict(family="bigamma", shape_fast=5.68, rate_fast=9.98, shape_slow=11.5,
           rate_slow=3.59)
K10_SUPPORT_S = 8.0

# Longest support a fitted kernel may have. A slow lobe that needs more than this is a
# drift, not a transient, and the tonic is what models drift.
MAX_KERNEL_S = 20.0


def _bandpass(fs: float):
    """The band-pass of `experiments.marks`, with the top edge kept below Nyquist."""
    hi = min(BAND_HZ[1], 0.45 * fs)
    return butter(3, [BAND_HZ[0] / (fs / 2), hi / (fs / 2)], "band")


# --------------------------------------------------------------------------- transients
@dataclass
class IsolatedTransients:
    """Band-passed transients aligned on their maxima, each scaled to peak one.

    `snippets[i]` runs from `pre` bins before the peak to `post` bins after it, so
    `t_s` is zero at the peak. `source[i]` is the recording it came from.
    """

    snippets: np.ndarray
    times_s: np.ndarray
    source: np.ndarray
    fs: float
    pre: int
    post: int

    @property
    def n(self) -> int:
        return int(self.snippets.shape[0])

    @property
    def t_s(self) -> np.ndarray:
        return (np.arange(self.pre + self.post) - self.pre) / self.fs

    @property
    def mean(self) -> np.ndarray:
        return self.snippets.mean(axis=0) if self.n else np.zeros(self.pre + self.post)

    @property
    def sd(self) -> np.ndarray:
        return (self.snippets.std(axis=0, ddof=1) if self.n > 1
                else np.zeros(self.pre + self.post))


def extract_isolated_transients(
    traces: np.ndarray | Sequence[np.ndarray],
    fs: float,
    masks: Sequence[np.ndarray | None] | None = None,
    *,
    prominence_sd: float = 4.0,
    floor_sd: float = 3.0,
    neighbour_frac: float = 0.5,
    isolation_s: float = 4.0,
    pre_s: float = 1.0,
    post_s: float = 4.0,
) -> IsolatedTransients:
    """Find the transients that stand alone, and cut them out aligned on their maxima.

    A transient is kept when its prominence exceeds `prominence_sd` robust standard
    deviations of the band-passed trace and no other peak lies within `isolation_s` of it
    on either side, counting as a neighbour any peak that is itself selectable and has at
    least `neighbour_frac` of the transient's prominence. `floor_sd` is the detection
    threshold for peaks at all. Smaller neighbours are let through on purpose: the noise's
    own maxima clear three standard deviations every few seconds, so counting them would
    disqualify nearly everything, and they land at random lags where the average washes
    them out. This is the selection the shipped kernel was fitted to (a 4 s exclusion
    among peaks above four standard deviations), with the larger neighbour required to
    be absent rather than merely smaller.

    `masks` mark the bins that count, as in `api.Session.mask`. A transient whose window,
    or the second either side of it, touches a masked bin is dropped: the band-pass spreads
    an artefact into its neighbourhood, and the injection's handling artefact is exactly
    the kind of excursion this would otherwise select.
    """
    if isinstance(traces, np.ndarray) and traces.ndim == 1:
        traces = [traces]
    traces = [np.asarray(y, float).reshape(-1) for y in traces]
    if masks is None:
        masks = [None] * len(traces)
    if len(masks) != len(traces):
        raise ValueError(f"{len(masks)} masks for {len(traces)} recordings")
    b, a = _bandpass(fs)
    pre, post = int(round(pre_s * fs)), int(round(post_s * fs))
    guard, iso = int(round(1.0 * fs)), int(round(isolation_s * fs))
    width = max(1, int(0.04 * fs))

    snips, times, src = [], [], []
    for r, (y, m) in enumerate(zip(traces, masks)):
        keep = np.ones(y.size, bool) if m is None else np.asarray(m, bool).reshape(-1)
        if keep.size != y.size:
            raise ValueError(f"recording {r}: mask has {keep.size} bins, trace {y.size}")
        if y.size < 3 * max(pre + post, 64):
            continue
        z = filtfilt(b, a, detrend(y, type="linear"))
        # The noise scale from the bins that count only, so a masked artefact cannot
        # inflate it and raise the threshold for the whole recording.
        sd = float(np.median(np.abs(z[keep] - np.median(z[keep]))) / 0.6745)
        if sd <= 0:
            continue
        pk, props = find_peaks(z, prominence=floor_sd * sd, width=width)
        prom = props["prominences"]
        for i, p in enumerate(pk):
            if prom[i] < prominence_sd * sd:
                continue
            lo, hi = p - pre, p + post
            if lo - guard < 0 or hi + guard > y.size:
                continue
            if not keep[lo - guard:hi + guard].all():
                continue
            near = ((np.abs(pk - p) < iso)
                    & (prom >= max(prominence_sd * sd, neighbour_frac * prom[i])))
            if near.sum() > 1:
                continue
            snips.append(z[lo:hi] / z[p])
            times.append(p / fs)
            src.append(r)

    return IsolatedTransients(
        snippets=np.asarray(snips, float).reshape(-1, pre + post),
        times_s=np.asarray(times, float), source=np.asarray(src, int),
        fs=float(fs), pre=pre, post=post)


# ----------------------------------------------------------------------------- families
# Each family is fitted in an unconstrained parameterization that holds its ordering
# constraints by construction, so the optimizer cannot step into a kernel that
# `build_kernel` refuses: a raising objective is a flat wall to Nelder-Mead.
def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _exp(x):
    return math.exp(min(max(x, -30.0), 30.0))


def _biexp_spec(th):
    kd = _exp(th[0])
    return dict(family="biexp", k_rise=kd * (1 + _exp(th[1])), k_decay=kd,
                normalize="zerosum")


def _biexp_theta(s):
    return [math.log(s["k_decay"]), math.log(s["k_rise"] / s["k_decay"] - 1)]


def _biphasic_spec(th):
    kdf = _exp(th[0])
    kds = kdf * _sig(th[2])
    return dict(family="biphasic", k_rise_fast=kdf * (1 + _exp(th[1])), k_decay_fast=kdf,
                k_rise_slow=kds * (1 + _exp(th[3])), k_decay_slow=kds)


def _biphasic_theta(s):
    return [math.log(s["k_decay_fast"]), math.log(s["k_rise_fast"] / s["k_decay_fast"] - 1),
            _logit(s["k_decay_slow"] / s["k_decay_fast"]),
            math.log(s["k_rise_slow"] / s["k_decay_slow"] - 1)]


def _bigamma_spec(th):
    rf = _exp(th[1])
    return dict(family="bigamma", shape_fast=1 + _exp(th[0]), rate_fast=rf,
                shape_slow=1 + _exp(th[2]), rate_slow=rf * _sig(th[3]))


def _bigamma_theta(s):
    return [math.log(s["shape_fast"] - 1), math.log(s["rate_fast"]),
            math.log(s["shape_slow"] - 1), _logit(s["rate_slow"] / s["rate_fast"])]


def _rise_for_peak(kd: float, peak_s: float) -> float:
    """The rise rate that puts a difference of exponentials' maximum at `peak_s`."""
    peak_s = min(peak_s, 0.9 / kd)       # ln(x)/(x-1)/kd never exceeds 1/kd
    return brentq(lambda kr: math.log(kr / kd) / (kr - kd) - peak_s, kd * 1.0001, kd * 1e4)


def _start(family: str, peak_s: float, decay_s: float) -> dict:
    kd = 1.0 / decay_s
    kr = _rise_for_peak(kd, peak_s)
    if family == "biexp":
        return dict(family="biexp", k_rise=kr, k_decay=kd, normalize="zerosum")
    if family == "biphasic":
        # The shipped biphasic kernels subtract a lobe about half as fast as the positive
        # one (K8: 3.30/1.65 minus 1.65/0.825).
        return dict(family="biphasic", k_rise_fast=kr, k_decay_fast=kd,
                    k_rise_slow=kd, k_decay_slow=kd / 2.0)
    # Gamma lobes with the shapes of K10 and the slow mode six times later, as there.
    return dict(family="bigamma", shape_fast=5.68, rate_fast=4.68 / peak_s,
                shape_slow=11.5, rate_slow=10.5 / (6.0 * peak_s))


FAMILIES = {
    "biphasic": (_biphasic_spec, _biphasic_theta),
    "bigamma": (_bigamma_spec, _bigamma_theta),
    "biexp": (_biexp_spec, _biexp_theta),
}


def _length(spec: dict, fs: float, *, final: bool) -> int:
    """The support a kernel needs, in bins.

    Each family needs it for its own reason. The zero-sum bi-exponential is mean-removed
    over its support, so its shape depends on it: cut at the decay's 1% point, in the fit
    and in the result alike. The biphasic family is mean-free at any length; its tail is
    the slow lobe's. The bigamma one is not, and `bigamma_kernel` refuses a support that
    deepens the undershoot -- in the fit it is built long enough to have converged, and the
    result gets the shortest support that passes that check.
    """
    cap = int(MAX_KERNEL_S * fs)
    fam = spec["family"]
    if fam == "biexp":
        n = math.ceil(4.6 / spec["k_decay"] * fs)
    elif fam == "biphasic":
        n = math.ceil(4.6 / spec["k_decay_slow"] * fs)
    else:
        far = int(8.0 * spec["shape_slow"] / spec["rate_slow"] * fs)
        if final:
            n, _ = _bigamma_support(spec["shape_fast"], spec["rate_fast"],
                                    spec["shape_slow"], spec["rate_slow"], 1.0 / fs, 0.05)
        else:
            n = far
    return int(min(max(n, 8), cap))


def _kernel(spec: dict, length: int, fs: float) -> np.ndarray:
    if spec["family"] == "bigamma":
        # No support check inside the fit: the support is chosen by `_length` and the
        # check costs a bisection per evaluation.
        h = _bigamma_raw(length, spec["shape_fast"], spec["rate_fast"],
                         spec["shape_slow"], spec["rate_slow"], 1.0 / fs)
    else:
        h = build_kernel(spec, length, 1.0 / fs)
    return h.detach().cpu().numpy().astype(float)


def _template(h: np.ndarray, fs: float, pre: int, post: int, ba) -> np.ndarray:
    """The kernel as the band-passed data would show it, aligned on its maximum.

    Padded with silence on both sides long enough for the high-pass's response to die
    out, then cut to the transients' window around its own maximum and scaled to one.
    """
    pad = int(15.0 * fs) + pre + post
    x = np.zeros(pad + h.size + pad)
    x[pad:pad + h.size] = h
    g = filtfilt(*ba, x)
    m = int(np.argmax(g))
    out = g[m - pre:m + post]
    return out / g[m]


def _rms(target: np.ndarray, g: np.ndarray) -> float:
    """Residual of the best amplitude-scaled template, in units of the peak."""
    a = float(g @ target / (g @ g))
    return float(np.sqrt(np.mean((target - a * g) ** 2)))


def _loss(spec: dict, target: np.ndarray, fs: float, pre: int, post: int, ba) -> float:
    try:
        h = _kernel(spec, _length(spec, fs, final=False), fs)
    except (ValueError, ZeroDivisionError, OverflowError):
        return 1e3
    if not np.isfinite(h).all():
        return 1e3
    return _rms(target, _template(h, fs, pre, post, ba))


def _fit_family(family: str, target: np.ndarray, starts: list[dict], fs: float,
                pre: int, post: int, ba, *, maxiter: int) -> tuple[dict, float]:
    to_spec, to_theta = FAMILIES[family]
    f = lambda th: _loss(to_spec(th), target, fs, pre, post, ba)  # noqa: E731
    # Score every start, optimize from the best two. Nelder-Mead finds the basin it starts
    # in, and the starts span sensors whose kinetics differ tenfold.
    thetas = sorted((np.asarray(to_theta(s), float) for s in starts), key=f)[:2]
    best = None
    for th in thetas:
        r = minimize(f, th, method="Nelder-Mead",
                     options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    return to_spec(best.x), float(best.fun)


# ------------------------------------------------------------------------------ result
def _shape(h: np.ndarray, fs: float) -> dict[str, float]:
    pk = int(np.argmax(h))
    return dict(peak_s=round(pk / fs, 4),
                fwhm_s=round(float((h >= 0.5 * h.max()).sum()) / fs, 4),
                undershoot=round(float(h.min()), 4),
                support_s=round(h.size / fs, 3),
                sum=float(h.sum()))


@dataclass
class KernelCandidate:
    """One kernel the user can choose, with what it scored on held-out transients.

    `spec` and `kernel_length` are exactly what `api.FitSettings(kernel=,
    kernel_length=)` takes. `template` is the kernel as the band-passed data would show
    it, on the transients' time axis, for drawing over their average.
    """

    key: str
    label: str
    family: str
    spec: dict[str, Any]
    kernel_length: int
    fitted: bool
    heldout_rms: float
    train_rms: float
    shape: dict[str, float]
    kernel: np.ndarray
    template: np.ndarray

    def settings(self) -> dict[str, Any]:
        """The two `FitSettings` fields that select this kernel."""
        return dict(kernel=dict(self.spec), kernel_length=int(self.kernel_length))

    def to_dict(self) -> dict[str, Any]:
        return dict(key=self.key, label=self.label, family=self.family,
                    spec=dict(self.spec), kernel_length=int(self.kernel_length),
                    fitted=self.fitted, heldout_rms=self.heldout_rms,
                    train_rms=self.train_rms, shape=dict(self.shape),
                    kernel=self.kernel.tolist(), template=self.template.tolist())


@dataclass
class KernelCalibration:
    """Every candidate kernel, scored on the same held-out transients, and a suggestion."""

    transients: IsolatedTransients
    candidates: list[KernelCandidate]
    recommended: str
    sensor: str
    folds: int
    warnings: list[str] = field(default_factory=list)

    def candidate(self, key: str | None = None) -> KernelCandidate:
        key = key or self.recommended
        for c in self.candidates:
            if c.key == key:
                return c
        raise KeyError(f"no candidate {key!r}; have {[c.key for c in self.candidates]}")

    def settings(self, key: str | None = None) -> dict[str, Any]:
        """`FitSettings` fields for a candidate, the suggested one by default."""
        return self.candidate(key).settings()

    def config(self, key: str | None = None) -> dict[str, Any]:
        """The `kernel:` block of a config file, as `configs/toy.yaml` writes it."""
        c = self.candidate(key)
        return dict(kernel=dict(length=int(c.kernel_length), fit=dict(c.spec)))

    def to_dict(self, max_points: int | None = None) -> dict[str, Any]:
        """Everything the interface draws, as plain JSON types."""
        tr = self.transients
        return dict(
            sensor=self.sensor, recommended=self.recommended, folds=self.folds,
            warnings=list(self.warnings), fs=tr.fs, n_transients=tr.n,
            n_recordings=int(np.unique(tr.source).size) if tr.n else 0,
            t_s=tr.t_s.tolist(), mean=tr.mean.tolist(), sd=tr.sd.tolist(),
            candidates=[c.to_dict() for c in self.candidates])

    def plot(self, path: Path | str) -> None:
        """The average isolated transient with every candidate drawn over it."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tr = self.transients
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.8))
        ax.fill_between(tr.t_s, tr.mean - tr.sd, tr.mean + tr.sd, color="#d9d9d9", lw=0)
        ax.plot(tr.t_s, tr.mean, color="#2b2b2b", lw=1.8,
                label=f"isolated transients (n = {tr.n})")
        for c in self.candidates:
            a = float(c.template @ tr.mean / (c.template @ c.template)) if tr.n else 1.0
            lw = 2.2 if c.key == self.recommended else 1.1
            ax.plot(tr.t_s, a * c.template, lw=lw,
                    label=f"{c.label}  held-out {c.heldout_rms:.3f}"
                          + ("  (suggested)" if c.key == self.recommended else ""))
            bx.plot(np.arange(c.kernel.size) / tr.fs, c.kernel, lw=lw)
        ax.set_xlabel("time from peak (s)")
        ax.set_ylabel("band-passed signal (peak = 1)")
        ax.legend(fontsize=7, frameon=False)
        bx.axhline(0, color="#8a8a8a", lw=0.8)
        bx.set_xlabel("time (s)")
        bx.set_ylabel("kernel $h$")
        bx.set_xlim(0, min(MAX_KERNEL_S, max(c.kernel.size for c in self.candidates) / tr.fs))
        fig.tight_layout()
        fig.savefig(path, facecolor="white")
        plt.close(fig)


def _default_kernel(fs: float) -> tuple[dict, int]:
    """The kernel an unconfigured fit at `fs` uses, via the API."""
    from photosvi.api import FitSettings
    r = FitSettings().resolved(fs=fs)
    return dict(r["kernel"]), int(r["kernel_length"])


def calibrate_kernel(
    sessions: Any,
    *,
    sensor: str = "dLight1.3b",
    families: Sequence[str] = ("biphasic", "bigamma", "biexp"),
    folds: int = 5,
    min_transients: int = 20,
    maxiter: int = 600,
    seed: int = 0,
    **extract_kw: Any,
) -> KernelCalibration:
    """Fit candidate kernels to a set of recordings and suggest one.

    `sessions` is an `api.Session`, a list of them, or an `api.Study`; all must share a
    sampling rate, since a kernel is a shape in bins once it is built. `sensor` chooses the
    starting values (see `SENSORS`) and labels the result; every family is also started
    from the other sensors' values. `extract_kw` go to `extract_isolated_transients`.

    Held-out error is the RMS residual, in units of the transient's peak, of a candidate
    against the average of the transients it was not fitted to, over `folds` folds. The
    two shipped kernels are scored on the same folds without fitting.
    """
    sess = list(getattr(sessions, "sessions", None) or
                (sessions if isinstance(sessions, (list, tuple)) else [sessions]))
    if not sess:
        raise ValueError("no recordings to calibrate on")
    rates = {float(s.fs) for s in sess}
    if len(rates) > 1:
        raise ValueError(f"recordings at different sampling rates {sorted(rates)}; "
                         "a kernel is calibrated for one rate")
    fs = rates.pop()
    if sensor not in SENSORS:
        raise ValueError(f"unknown sensor {sensor!r}; known: {sorted(SENSORS)}")
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families {sorted(unknown)}; known: {sorted(FAMILIES)}")

    tr = extract_isolated_transients([s.trace for s in sess], fs,
                                     [s.mask for s in sess], **extract_kw)
    ba = _bandpass(fs)
    pre, post = tr.pre, tr.post
    notes: list[str] = []

    dspec, dlen = _default_kernel(fs)
    fixed = [("default", "shipped default: dLight1.3b, fitted to the paper's recordings (K10)",
              dspec, dlen)]

    enough = tr.n >= min_transients
    if not enough:
        notes.append(
            f"only {tr.n} isolated transients (fewer than {min_transients}): too few to "
            "fit a shape, so the shipped kernel is suggested. Add recordings, or loosen "
            "the isolation if the transients are frequent.")
        warnings.warn(notes[-1], stacklevel=2)
    elif tr.n < 3 * min_transients:
        notes.append(f"{tr.n} isolated transients: enough to fit, few enough that the "
                     "held-out errors are noisy; prefer a shipped kernel unless a fitted "
                     "one is clearly better.")

    k = max(2, min(folds, tr.n // max(1, min_transients // 4))) if enough else 0
    rng = np.random.default_rng(seed)
    order = rng.permutation(tr.n)
    split = np.array_split(order, k) if k else []
    target = tr.mean

    def heldout(g_of_train) -> float:
        errs = []
        for test in split:
            train = np.setdiff1d(order, test)
            g = g_of_train(tr.snippets[train].mean(axis=0))
            errs.append(_rms(tr.snippets[test].mean(axis=0), g))
        return float(np.mean(errs)) if errs else float("nan")

    cands: list[KernelCandidate] = []
    for key, label, spec, n in fixed:
        h = build_kernel(spec, n, 1.0 / fs).detach().cpu().numpy().astype(float)
        g = _template(h, fs, pre, post, ba)
        cands.append(KernelCandidate(
            key=key, label=label, family=spec["family"], spec=dict(spec), kernel_length=n,
            fitted=False, heldout_rms=heldout(lambda _m, g=g: g) if enough else float("nan"),
            train_rms=_rms(target, g) if tr.n else float("nan"),
            shape=_shape(h, fs), kernel=h, template=g))

    if enough:
        sensor_first = [sensor] + [s for s in SENSORS if s != sensor]
        for fam in families:
            starts = [_start(fam, **SENSORS[s]) for s in sensor_first]
            if fam == "bigamma":
                starts.append(dict(K10))
            spec, train = _fit_family(fam, target, starts, fs, pre, post, ba,
                                      maxiter=maxiter)
            # Refitting per fold from the full-data optimum: the start is shared, the
            # fit is not, so each fold's error is of a kernel that never saw its test set.
            def g_of(m, fam=fam, spec=spec):
                s, _ = _fit_family(fam, m, [spec], fs, pre, post, ba,
                                   maxiter=maxiter // 3)
                return _template(_kernel(s, _length(s, fs, final=False), fs),
                                 fs, pre, post, ba)
            # Rounded before the kernel is built, so the spec written to a config builds
            # exactly the kernel that was scored.
            spec = {kk: (round(v, 4) if isinstance(v, float) else v)
                    for kk, v in spec.items()}
            n = _length(spec, fs, final=True)
            try:
                # Through build_kernel, with its checks, because that is how the fit
                # will build it; a kernel the fit would refuse is not a candidate.
                h = build_kernel(spec, n, 1.0 / fs).detach().cpu().numpy().astype(float)
            except ValueError as exc:
                notes.append(f"{fam}: the fitted kernel is not usable ({exc}); dropped.")
                continue
            cands.append(KernelCandidate(
                key=f"fit-{fam}", label=f"{fam}, fitted to these recordings",
                family=fam, spec=spec, kernel_length=n, fitted=True,
                heldout_rms=heldout(g_of), train_rms=train, shape=_shape(h, fs),
                kernel=h, template=_template(h, fs, pre, post, ba)))
        best = min(cands, key=lambda c: c.heldout_rms)
        recommended = best.key
    else:
        recommended = "default"

    return KernelCalibration(transients=tr, candidates=cands, recommended=recommended,
                             sensor=sensor, folds=k, warnings=notes)
