"""An estimate of the mark distribution from the visible peaks of the recordings.

The mark distribution is not identified by the likelihood alone: the signal depends on
the event indicator times the mark, and at low SNR the fit trades one for the other
(measured 2026-10-09: with mu_a and sigma_a free, mu_a fell 0.6-1.2 below its simulated
value at every SNR from 1.5 to 25, and beta0 rose 0.7-2 above it). The prior that holds
them comes from here: an estimate made from the trace before any fit, by a procedure that
does not involve the model.

The estimate uses only the pre-injection segment, so that the drug cannot enter it, and
only peaks that stand clear of the noise, so it describes the transients that can be seen.
Both make it biased -- small transients are missed, so the median is too high and the
spread too low -- which is why it anchors a prior with a stated width rather than fixing
the parameters.

This is the procedure scripts/cohort_bonaventura3.py used inline for the real recordings
from 2026-10-02; it is a function now so the simulations and the recordings get the same.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend, filtfilt, find_peaks

BAND_HZ = (0.1, 10.0)
PROMINENCE_SD = 3.0
MIN_WIDTH_S = 0.04


def peak_prominences(y: np.ndarray, fs: float) -> np.ndarray:
    """Prominences of the peaks of one segment that stand clear of its noise.

    Linear detrend, zero-phase third-order Butterworth band-pass, then local maxima with a
    prominence above PROMINENCE_SD robust standard deviations and a width of at least
    MIN_WIDTH_S.
    """
    z = filtfilt(*butter(3, [BAND_HZ[0] / (fs / 2), BAND_HZ[1] / (fs / 2)], "band"),
                 detrend(np.asarray(y, float), type="linear"))
    sd = np.median(np.abs(z)) / 0.6745
    _, props = find_peaks(z, prominence=PROMINENCE_SD * sd,
                          width=max(1, int(MIN_WIDTH_S * fs)))
    return np.asarray(props["prominences"], float)


def estimate_mark_distribution(segments: list[np.ndarray], fs: float) -> tuple[float, float]:
    """(mu_a, sigma_a) from one or more pre-injection segments.

    mu_a is the log of the median over segments of each segment's median prominence;
    sigma_a the median over segments of the standard deviation of its log prominences.
    With one segment these are that segment's own.
    """
    meds, sds = [], []
    for seg in segments:
        p = peak_prominences(seg, fs)
        if p.size < 3:
            continue
        meds.append(float(np.median(p)))
        sds.append(float(np.std(np.log(p), ddof=1)))
    if not meds:
        raise ValueError("no segment had three peaks clear of the noise")
    return float(np.log(np.median(meds))), float(np.median(sds))


# --------------------------------------------------------------- noise-unit marks
# An alternative to the estimate above, opt-in: the mark distribution is
# fixed a priori in units of the emission noise, a_t = sigma_y * r_t with
# r_t ~ LogNormal(mu_r, sigma_r^2), and neither parameter is fitted. The defaults put the
# 2.5th percentile at about 1 sigma_y (just above the noise) and the 97.5th at about 12
# sigma_y (the largest prominences seen in the recordings), median 3.5 sigma_y. Being in
# noise units it carries over between sensors and rigs, and only sigma_y -- which every
# fit already estimates -- is read off the data. Selected by `fit.mark_mode: noise_units`
# in a config or MARK_MODE=noise_units in the environment; absent, nothing changes.
#
# Defaults since 2026-10-21 (the paper's final model): median 1.95 sigma_y -- the
# calibrated medium SNR, read off the recordings by the same peak measurement on simulated
# and experimental traces -- and sigma_r 0.75, wider than the visible peaks' spread
# because the posterior predictive check showed the recordings' large-transient tail
# heavier than 0.63 allows. The first pilot's 1.24 / 0.63 came from detected peaks only,
# which the detector truncates from below, and put every simulation at SNR 3.5.
NOISE_UNITS_MU = 0.668    # ln(1.95)
NOISE_UNITS_SIGMA = 0.75


def noise_unit_mode(fit_cfg: dict | None = None) -> tuple[float, float] | None:
    """(mu_r, sigma_r) when the noise-unit mark mode is selected, else None."""
    import os
    fit_cfg = fit_cfg or {}
    mode = os.environ.get("MARK_MODE") or fit_cfg.get("mark_mode")
    if mode in (None, "", "default"):
        return None
    if mode != "noise_units":
        raise ValueError(f"unknown mark mode {mode!r}")
    p = fit_cfg.get("mark_noise_units") or {}
    return (float(p.get("mu", NOISE_UNITS_MU)), float(p.get("sigma", NOISE_UNITS_SIGMA)))


def noise_unit_generation(fit_cfg: dict | None, sigma_y: float, mu_a: float,
                          sigma_a: float) -> tuple[float, float]:
    """(mu_a, sigma_a) the SIMULATOR draws marks from under the noise-unit mode.

    By default the simulator draws from the same distribution the fit assumes, in units of
    the simulated sigma_y. `mark_noise_units.gen_mu_a: spec` keeps the config's own mu_a
    (absolute units) instead, with width `gen_sigma` (default the fit's sigma_r), so that
    the SNR of a cell stays e^{mu_a} / sigma_y and a sweep over sigma_y sweeps the SNR,
    while the fit keeps its fixed mu_r at every level -- the paper's final design
    (2026-10-21): well specified at the medium level, deliberately misspecified at the
    high and low ones.
    """
    nu = noise_unit_mode(fit_cfg)
    if nu is None:
        return float(mu_a), float(sigma_a)
    p = (fit_cfg or {}).get("mark_noise_units") or {}
    gen = p.get("gen_mu_a")
    if gen is None:
        return noise_unit_marks(sigma_y, *nu)
    if gen != "spec":
        mu_a = float(gen)
    return float(mu_a), float(p.get("gen_sigma", nu[1]))


def noise_unit_marks(sigma_y: float, mu_r: float = NOISE_UNITS_MU,
                     sigma_r: float = NOISE_UNITS_SIGMA) -> tuple[float, float]:
    """(mu_a, sigma_a) of a_t = sigma_y r_t, r_t ~ LogNormal(mu_r, sigma_r^2)."""
    import math
    return math.log(float(sigma_y)) + float(mu_r), float(sigma_r)


def fix_marks_at(phasic, mu_a: float, sigma_a: float) -> None:
    """Set a PhasicEventProcess's mark distribution and hold it fixed (no prior, no fit)."""
    import math

    import torch
    with torch.no_grad():
        phasic.mu_a.fill_(float(mu_a))
        phasic.log_sigma_a.fill_(math.log(float(sigma_a)))
    phasic.mark_prior = None
    phasic.mu_a.requires_grad_(False)
    phasic.log_sigma_a.requires_grad_(False)


# ------------------------------------------------------- per-session noise scale
# A cohort model with one sigma_y^(j) per session instead of one for the cohort
# (2026-10-20): each estimated as in the single-trace model, from that session's own
# pre-injection segment, and fixed. Under the noise-unit mode the marks follow it,
# mu_a^(j) = log sigma_y^(j) + mu_r. Selected by `fit.sigma_y_per_session: true` or
# SIGMA_Y_PER_SESSION=1; absent, the cohort keeps its single sigma_y.
def per_session_sigma(fit_cfg: dict | None = None) -> bool:
    """Whether the cohort model takes one sigma_y per session."""
    import os
    env = os.environ.get("SIGMA_Y_PER_SESSION")
    if env is not None:
        return env not in ("", "0", "false")
    return bool((fit_cfg or {}).get("sigma_y_per_session", False))


def fix_marks_per_session(phasic, mu_a, sigma_a: float) -> None:
    """Hold the marks fixed at a per-session location `mu_a` ([S]) and a common width."""
    import math

    import torch
    from torch import nn
    mu = torch.as_tensor(mu_a, dtype=phasic.mu_a.dtype, device=phasic.mu_a.device)
    phasic.mu_a = nn.Parameter(mu.reshape(-1, 1), requires_grad=False)
    with torch.no_grad():
        phasic.log_sigma_a.fill_(math.log(float(sigma_a)))
    phasic.mark_prior = None
    phasic.log_sigma_a.requires_grad_(False)
