"""The tonic pre-fit must find a small bump, not hand it to the drift.

`fit_tonic_baseline` least-squares a Bateman plus a linear drift to the smoothed trace, as
the initialization SVI starts from. Over a recording of this length a SMALL Bateman is
close enough to a ramp that the optimizer can give the whole post-injection rise to the
drift and walk the amplitude onto its lower bound, returning a peak of exactly zero. The
session then enters SVI with no slow component at all and never recovers: its residual
ratio is pinned from the first probe.

It was invisible while the simulated bump was 303 noise standard deviations, where the
Bateman's curvature has no competitor. It appeared the moment the configuration was
corrected to the 15 the recordings carry, and cost three of four fits.

The fix fits the bump first with the drift pinned at zero and only then releases it, so
these cases assert what that is for: a small bump is still found, and a bump that is
genuinely absent still comes back as something the model can shrink rather than as the
bound itself.
"""
from __future__ import annotations

import math

import pytest
import torch

from photosvi.models.points.tonic import fit_tonic_baseline

FS = 48.0
DT = 1.0 / FS
# The same geometry as the half-duration study config: the Bateman peaks about 418 s
# after the injection, so the window has to run well past that or the decay is never
# observed and the fitter trades k_decay against the amplitude. A first version of this
# file ended 41 s after the peak and "failed" by recovering 1.7x the simulated bump,
# which was the test's geometry and not the fit.
T = 58295
INJ = 14400


def _trace(peak: float, drift: float = 0.0, noise: float = 0.013, seed: int = 0):
    """A Bateman bump after `INJ`, plus a linear drift and white noise."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(T, dtype=torch.float32) * DT
    tau = torch.clamp(t - INJ * DT, min=0.0)
    k_decay, k_rise = 9e-4, 5.9e-3
    f = torch.exp(-k_decay * tau) * (-torch.expm1(-(k_rise - k_decay) * tau))
    t_pk = math.log(k_rise / k_decay) / (k_rise - k_decay)
    f_pk = math.exp(-k_decay * t_pk) * (-math.expm1(-(k_rise - k_decay) * t_pk))
    y = peak * f / f_pk + drift * tau / 3600.0
    return (y + noise * torch.randn(T, generator=g))[None, :]


@pytest.mark.parametrize("seed", range(3))
def test_a_small_bump_is_found_and_not_given_to_the_drift(seed):
    """15 noise standard deviations: what the recordings carry."""
    peak = 0.1962
    fitted = float(fit_tonic_baseline(_trace(peak, seed=seed), INJ, dt=DT).peak)
    assert fitted > 0.3 * peak, (
        f"peak collapsed to {fitted:.5f} against a simulated {peak}; the drift has taken "
        f"the bump (seed {seed})")
    assert fitted == pytest.approx(peak, rel=0.4)


@pytest.mark.parametrize("seed", range(2))
def test_a_small_bump_survives_a_real_drift(seed):
    """The drift is there to be fitted; it must not swallow the bump on the way."""
    peak = 0.1962
    fitted = float(fit_tonic_baseline(_trace(peak, drift=-0.5, seed=seed), INJ, dt=DT).peak)
    assert fitted > 0.3 * peak, f"peak collapsed to {fitted:.5f} with a drift present"


def test_a_large_bump_is_still_recovered():
    """The regime everything was fitted in before, which must not regress."""
    peak = 3.9394
    fitted = float(fit_tonic_baseline(_trace(peak, noise=0.013), INJ, dt=DT).peak)
    assert fitted == pytest.approx(peak, rel=0.05)


def test_no_bump_returns_something_the_model_can_shrink():
    """A flat trace has no bump; the pre-fit should say so without returning the bound."""
    fitted = float(fit_tonic_baseline(_trace(0.0), INJ, dt=DT).peak)
    assert fitted >= 0.0
    assert fitted < 0.05, f"invented a bump of {fitted:.4f} where there is none"
