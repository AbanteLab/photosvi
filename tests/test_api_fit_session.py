"""The public entry point runs.

Not what it returns -- the benchmark establishes that over thirty recordings with known
truth -- but that calling it works at all. The suite covers the pieces thoroughly and the
assembled thing not at all, so a change that keeps every piece correct and the wiring wrong
passes: adding a field to `FitSettings.resolved()` without adding it to `PointsSetup` broke
every call to `fit_session` and the suite went green. Another repository's test caught it.
"""
from __future__ import annotations

import numpy as np
import pytest

from photosvi.api import FitSettings, Session, fit_session


@pytest.fixture
def short_recording():
    """Two minutes at 25 Hz with an injection in the middle. Small enough to fit in a test."""
    fs, n = 25.0, 3000
    rng = np.random.default_rng(0)
    t = np.arange(n) / fs
    trace = 0.5 * np.exp(-((t - 60.0) / 40.0) ** 2) + rng.normal(0, 0.1, n)
    return Session(trace=trace.astype(np.float32), fs=fs, inj_t=n // 2, name="t")


def test_fit_session_runs_with_default_settings(short_recording):
    r = fit_session(short_recording, FitSettings(steps=30, max_steps=30, threads=1))
    rr = r.rate_ratio
    assert np.isfinite(rr["estimate"]) and rr["estimate"] > 0
    assert rr["lo"] <= rr["estimate"] <= rr["hi"]
    assert r.event_prob.shape == short_recording.trace.shape
    assert r.stopped_at > 0


def test_fit_session_accepts_every_setting_it_advertises(short_recording):
    """Each field of FitSettings has to survive the trip into the model.

    This is the specific failure: a setting can be added to `resolved()`, documented, and
    accepted by the dataclass while nothing carries it the last step into the fit.
    """
    s = FitSettings(steps=30, max_steps=30, threads=1, seed=1, tau=0.3, window=4,
                    sigma_y=0.1, mu_a=1.5, sigma_a=0.3, lr=0.02,
                    event_threshold=0.4, refractory=6, mark_gate_floor=0.01)
    resolved = s.resolved()
    for name in vars(s):
        assert name in resolved, f"{name} is a setting that resolved() drops"
    r = fit_session(short_recording, s)
    assert np.isfinite(r.rate_ratio["estimate"])
