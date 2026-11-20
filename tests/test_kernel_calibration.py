"""Calibrating the transient kernel from recordings, before any fit.

Simulated recordings are drawn with a known kernel from each family, under AR(1) noise and
a slow drift, and the calibration has to recover the shape it was given. "Recover" means
the shape -- time to peak and width -- rather than the four constants, which were found
to lie on flat ridges; and it means the true family's fit, since the suggestion is allowed
to be a different family that draws the same curve.
"""

from __future__ import annotations

import numpy as np
import pytest

from photosvi.api import FitSettings, Session, calibrate_kernel
from photosvi.kernel_calibration import (
    K10,
    K10_SUPPORT_S,
    _shape,
    extract_isolated_transients,
)
from photosvi.models.points.events import build_kernel

FS = 48.0

TRUTHS = {
    "bigamma": (K10, 384),
    "biphasic": (dict(family="biphasic", k_rise_fast=12.0, k_decay_fast=5.0,
                      k_rise_slow=5.0, k_decay_slow=2.0), 300),
    "biexp": (dict(family="biexp", k_rise=15.0, k_decay=4.0, normalize="zerosum"), 60),
}


def _recording(spec, length, *, minutes=20.0, rate_per_min=6.0, noise=0.1, seed=0,
               mask=None):
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * FS)
    s = np.zeros(n)
    ev = rng.random(n) < rate_per_min / 60 / FS
    s[ev] = np.exp(rng.normal(0.0, 0.3, ev.sum()))
    h = build_kernel(spec, length, 1 / FS).numpy()
    y = np.convolve(s, h)[:n]
    w = rng.normal(0.0, noise, n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.6 * e[t - 1] + w[t]
    y = y + e + 0.3 * np.sin(np.arange(n) / FS / 300.0)
    return Session(trace=y, fs=FS, inj_t=n // 2, mask=mask)


@pytest.fixture(scope="module", params=sorted(TRUTHS))
def calibrated(request):
    spec, length = TRUTHS[request.param]
    sess = [_recording(spec, length, seed=s) for s in (1, 2)]
    return request.param, calibrate_kernel(sess, folds=3)


def test_the_true_familys_fit_recovers_the_shape(calibrated):
    fam, cal = calibrated
    spec, length = TRUTHS[fam]
    true = _shape(build_kernel(spec, length, 1 / FS).numpy(), FS)
    got = cal.candidate(f"fit-{fam}").shape
    # A tenth of a second: the data are band-passed at 10 Hz before anything is fitted, so
    # a rise faster than that is smoothed in the data and the fit cannot place the peak
    # more finely than the filter lets it.
    assert got["peak_s"] == pytest.approx(true["peak_s"], abs=0.1)
    assert got["fwhm_s"] == pytest.approx(true["fwhm_s"], rel=0.2, abs=2 / FS)


def test_a_fitted_kernel_is_suggested_and_beats_the_wrong_shipped_one(calibrated):
    fam, cal = calibrated
    best = cal.candidate()
    assert best.fitted
    # K10 is the truth for the bigamma case, so only there may a shipped kernel tie.
    if fam != "bigamma":
        assert best.heldout_rms < cal.candidate("default").heldout_rms


def test_every_candidate_is_mean_free_and_peak_one(calibrated):
    _, cal = calibrated
    for c in cal.candidates:
        h = build_kernel(c.spec, c.kernel_length, 1 / FS).numpy()
        assert abs(h.sum()) < 1e-4, c.key
        assert h.max() == pytest.approx(1.0, abs=1e-5), c.key
        assert np.allclose(h, c.kernel, atol=1e-6), c.key


def test_the_suggestion_is_what_a_fit_takes(calibrated):
    _, cal = calibrated
    st = FitSettings(**cal.settings())
    r = st.resolved()
    assert r["kernel"] == cal.candidate().spec
    assert r["kernel_length"] == cal.candidate().kernel_length
    block = cal.config()["kernel"]
    assert block["fit"] == cal.candidate().spec and block["length"] > 0


def test_the_result_is_plain_json(calibrated):
    import json
    _, cal = calibrated
    d = json.loads(json.dumps(cal.to_dict()))
    assert d["recommended"] in [c["key"] for c in d["candidates"]]
    assert len(d["mean"]) == len(d["t_s"]) == len(d["candidates"][0]["template"])


def test_too_few_transients_falls_back_to_the_shipped_kernel():
    spec, length = TRUTHS["bigamma"]
    sess = _recording(spec, length, minutes=1.5, seed=3)
    with pytest.warns(UserWarning, match="too few"):
        cal = calibrate_kernel(sess)
    assert cal.recommended == "default"
    assert cal.warnings
    assert not any(c.fitted for c in cal.candidates)


def test_masked_bins_are_never_inside_a_transient():
    spec, length = TRUTHS["bigamma"]
    sess = _recording(spec, length, minutes=10, seed=4)
    keep = np.ones(sess.n_samples, bool)
    keep[int(200 * FS):int(400 * FS)] = False
    tr = extract_isolated_transients(sess.trace, FS, [keep])
    assert tr.n > 0
    assert not ((tr.times_s > 200 - 6) & (tr.times_s < 400 + 6)).any()


def test_recordings_at_different_rates_are_refused():
    spec, length = TRUTHS["bigamma"]
    a = _recording(spec, length, minutes=2, seed=5)
    b = Session(trace=a.trace, fs=25.0, inj_t=0)
    with pytest.raises(ValueError, match="sampling rates"):
        calibrate_kernel([a, b])


def test_the_calibration_k10_is_the_api_default():
    """The shipped candidate and an unconfigured fit must be the same kernel."""
    r = FitSettings().resolved(fs=48.0)
    assert K10 == r["kernel"]
    assert abs(K10_SUPPORT_S * 48 - r["kernel_length"]) <= 0.05 * r["kernel_length"]
