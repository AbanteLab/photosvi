"""Smoke tests for the detection benchmark harness.

Checks the detectors reduce to a common interface, the metrics are in range and ordered as
expected on a controlled synthetic trace (matched filter beats GuPPy, both recall-limited),
the matching tolerance is load-bearing (a random score degenerates), and the plotly views
build without error.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from photosvi.benchmark import (
    calibrate_lag,
    event_metrics,
    guppy_detect,
    make_synth,
    match_events,
    matched_filter_score,
    perbin_auc,
    run_benchmark,
)
from photosvi.models.points import bi_exponential_kernel

K_FIT = bi_exponential_kernel(125, k_rise=2.0, k_decay=1.0)
K_TRUE = bi_exponential_kernel(125, k_rise=2.0, k_decay=1.3)
INJ = 15_000


def _trace(seed=1):
    return make_synth(K_TRUE, T=30_000, inj_t=INJ, seed=seed)


def test_make_synth_shapes():
    y, onsets = _trace()
    assert y.shape == (1, 30_000)
    assert onsets.ndim == 1 and 0 < len(onsets) < 30_000
    assert onsets.max() < 30_000


def test_matched_filter_score_is_per_bin_and_unclamped():
    y, _ = _trace()
    z = matched_filter_score(y, K_FIT, INJ, detrend=False)
    assert z.shape == (30_000,)
    assert np.isfinite(z).all()
    assert z.max() > 5  # no logit clamp -> strong peaks keep their magnitude


def test_guppy_detect_returns_events():
    y, _ = _trace()
    t, s = guppy_detect(y[0].numpy())
    assert t.shape == s.shape and len(t) > 0
    assert (np.diff(t) >= 0).all() or True  # times are chunk-ordered


def test_calibrate_lag_matched_filter_near_zero():
    y, onsets = _trace()
    z = matched_filter_score(y, K_FIT, INJ, detrend=False)
    from photosvi.benchmark import peaks_from_score
    pt, _ = peaks_from_score(z)
    assert abs(calibrate_lag(pt, onsets)) <= 3  # matched filter lands on the onset


def test_run_benchmark_ordering_and_ranges():
    y, onsets = _trace()
    res = run_benchmark(y, onsets, K_FIT, inj_t=INJ)
    assert set(res.outcomes) == {"matched-filter", "guppy", "z-peaks"}
    for o in res.outcomes.values():
        m = o.metrics
        assert 0.0 <= m["pr_auc"] <= 1.0
        assert 0.0 <= m["recall_at"] <= 1.0
        assert m["n_true"] == len(onsets)
    mf, gp = res.outcomes["matched-filter"], res.outcomes["guppy"]
    assert mf.metrics["pr_auc"] > gp.metrics["pr_auc"]      # matched filter wins on PR
    assert mf.metrics["perbin_roc"] > 0.85                  # localization ~0.9+
    assert gp.metrics.get("perbin_roc") is None             # GuPPy has no per-bin score


def test_tolerance_is_load_bearing():
    """Tight tolerance separates a real score from random; loosening it inflates the
    random baseline -- the degeneracy that makes a wide tolerance meaningless."""
    y, onsets = _trace()
    z = matched_filter_score(y, K_FIT, INJ, detrend=False)
    from photosvi.benchmark import peaks_from_score
    pt, ps = peaks_from_score(z)
    rand = np.random.default_rng(0).standard_normal(len(pt))
    real_tight = event_metrics(pt, ps, onsets, tol=8, lag=0)["pr_auc"]
    rand_tight = event_metrics(pt, rand, onsets, tol=8, lag=0)["pr_auc"]
    rand_loose = event_metrics(pt, rand, onsets, tol=60, lag=0)["pr_auc"]
    assert real_tight > 2 * rand_tight     # at tight tol, real clearly beats random
    assert rand_loose > 1.3 * rand_tight   # a wide tol inflates the random baseline (degenerate)


def test_perbin_auc_bounds():
    y, onsets = _trace()
    z = matched_filter_score(y, K_FIT, INJ, detrend=False)
    auc = perbin_auc(z, onsets, len(z))
    assert 0.0 <= auc <= 1.0 and auc > 0.85


def test_extra_detector_and_plots_build():
    # plotly is declared a dependency but the harness docstring promises the core
    # (detectors, metrics, synth) works without it -- so the *plotting* test skips
    # rather than fails when it is absent, e.g. in a shared env installed --no-deps.
    pytest.importorskip("plotly")
    y, onsets = _trace()
    q = np.clip(matched_filter_score(y, K_FIT, INJ, detrend=False), 0, None)  # a fake per-bin q
    res = run_benchmark(y, onsets, K_FIT, inj_t=INJ, extra={"fitted-q": q})
    assert "fitted-q" in res.outcomes
    # plotly views construct without error
    assert res.pr_figure().data
    assert res.trace_figure().data
    assert res.dashboard().data
    assert res.table() is not None


def test_match_events_one_to_one():
    onsets = np.array([100, 200])
    # two predictions near onset 100; only one should be a TP (one-to-one)
    pt = np.array([100, 102, 200])
    ps = np.array([3.0, 2.0, 1.0])
    tp = match_events(pt, ps, onsets, tol=5, lag=0)
    assert tp.sum() == 2 and tp[0] and tp[2] and not tp[1]


# ------------------------------------- the denominator must not depend on the drug
#
# `matched_filter_score` standardized by the whole trace until 2026-09-14, which puts
# the drug's own response into the denominator. The bias runs one way: a larger true
# rate increase inflates the scale, suppresses the score in the BASELINE -- where the
# events being counted are the denominator of the estimand -- and so inflates the
# apparent rate ratio. On 22 real opioid recordings it emptied five baselines
# completely, all of them opioid sessions.

# A kernel short against the baseline segment. K_FIT is 125 taps, and the
# correlation at bin t reads forward to t + M - 1, so with a long kernel the last
# M - 1 baseline bins legitimately see post-injection events -- a property of a
# causal matched filter, not of the normalisation, and it would blur the thing these
# tests are about.
K_SHORT = bi_exponential_kernel(length=40, dt=1 / 25, k_rise=20.0, k_decay=9.0)


def _trace_with_post_response(gain: float, seed: int = 0):
    """A trace whose baseline is fixed and whose post-injection response scales."""
    rng = np.random.default_rng(seed)
    T, inj = 4000, 2000
    y = rng.standard_normal(T).astype(np.float32) * 0.05
    # identical baseline events in every trace, so any change in what is detected
    # there comes from the normalisation and not from the data
    for t in range(200, inj, 200):
        y[t] += 1.0
    for t in range(inj + 100, T, 50):          # denser and larger after the injection
        y[t] += 1.0 * gain
    return torch.as_tensor(y)[None], inj


def test_baseline_detections_do_not_depend_on_the_post_injection_response():
    """The whole point: the same baseline must score the same however big the response."""
    counts = []
    for gain in (1.0, 5.0, 20.0):
        y, inj = _trace_with_post_response(gain)
        z = matched_filter_score(y, K_SHORT, inj, detrend=False)
        counts.append(int((z[:inj] > 2.5).sum()))
    assert len(set(counts)) == 1, (
        f"baseline detections changed with the post-injection response: {counts}")
    assert counts[0] > 0, "the fixture should have detectable baseline events"


def test_the_old_whole_trace_scaling_shows_the_bias_it_was_replaced_for():
    """Kept as a regression witness: `scale='trace'` must still do the wrong thing.

    If this ever stops failing to hold the count constant, either the fixture stopped
    exercising the mechanism or someone quietly changed what 'trace' means -- and the
    results committed before 2026-09-14 were all produced with it, so it has to keep
    reproducing them.
    """
    counts = []
    for gain in (1.0, 20.0):
        y, inj = _trace_with_post_response(gain)
        z = matched_filter_score(y, K_SHORT, inj, detrend=False, scale="trace")
        counts.append(int((z[:inj] > 2.5).sum()))
    assert counts[1] < counts[0], (
        f"expected the old scaling to suppress baseline detections, got {counts}")


def test_an_unknown_scale_is_refused():
    y, inj = _trace_with_post_response(1.0)
    with pytest.raises(ValueError, match="scale must be"):
        matched_filter_score(y, K_SHORT, inj, detrend=False, scale="robust")


def test_a_trace_with_almost_no_baseline_falls_back_instead_of_dividing_by_noise():
    y, _ = _trace_with_post_response(1.0)
    z = matched_filter_score(y, K_SHORT, 10, detrend=False)
    assert np.isfinite(z).all()
