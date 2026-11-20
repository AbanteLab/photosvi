"""Tests for the pluggable event-guide blocks (models/points/event_guides.py).

Covers the seam itself (the default block keeps SingleTraceModel' old behaviour) and
the amortized `MatchedFilterEventGuide`: its constructor guards, that a fit tunes
two scalars instead of B*T free logits, that it recovers planted events, and --
the property the family exists for -- that its event ranking is pinned to the
matched filter's and therefore cannot erode as SVI runs.

Also covers `WarmFreeEventGuide` (same family as the free block, matched-filter
init) and the `couple_events` flag on SingleTraceModel, whose readout under coupling
is not the block's logits alone.
"""

from __future__ import annotations

import math

import numpy as np
import pyro
import pytest
import torch

from photosvi.benchmark import matched_filter_score
from photosvi.inference.points import train_points_svi
from photosvi.models.points import (
    AdditiveEmission,
    FreeEventGuide,
    MatchedFilterEventGuide,
    PhasicEventProcess,
    SingleTraceModel,
    TonicBaseline,
    WarmFreeEventGuide,
    bi_exponential_kernel,
)

EVENT_BINS = (120, 200, 260)


def _planted_data(event_bins=EVENT_BINS, T=300, inj_t=100, noise=0.05, seed=0):
    """A [1, T] trace with events at known bins (mirrors test_points_inference)."""
    h = bi_exponential_kernel(40, k_rise=3.0, k_decay=0.8)
    with torch.no_grad():
        proc = PhasicEventProcess(h, mu_a=0.0, sigma_a=0.2)
        tb = TonicBaseline(b=0.0, peak=0.42799379518491015,  # was A=0.8; same curve
                          k_rise=0.05, k_decay=0.01)
        s = torch.zeros(1, T)
        for b in event_bins:
            s[0, b] = 1.0
        c = proc.convolve(s * torch.ones(1, T))
        g = torch.Generator().manual_seed(seed)
        y = tb.baseline(T, inj_t) + c + noise * torch.randn(1, T, generator=g)
    return y, inj_t


def _model(event_guide=None, tau=0.5, couple_events=False):
    h = bi_exponential_kernel(40, k_rise=3.0, k_decay=0.8)
    return SingleTraceModel(
        PhasicEventProcess(h, beta0=-2.2, beta_inj=1.2, mu_a=0.0, sigma_a=0.2),
        TonicBaseline(b=0.0, peak=0.42799379518491015,  # was A=0.8; same curve
                          k_rise=0.05, k_decay=0.01),
        AdditiveEmission(sigma_y=0.08),
        tau=tau,
        event_guide=event_guide,
        couple_events=couple_events,
    )


def _mf(y, inj_t):
    h = bi_exponential_kernel(40, k_rise=3.0, k_decay=0.8)
    return matched_filter_score(y, h, inj_t, detrend=False)


def test_default_event_guide_is_free_and_keeps_param_name():
    """The seam is backwards compatible: no event_guide -> the old free per-bin block."""
    pyro.clear_param_store()
    y, inj_t = _planted_data()
    m = _model()
    assert isinstance(m.event_guide, FreeEventGuide)
    train_points_svi(m, y, inj_t, lr=1.5e-2, steps=20, verbose=False, seed=0)
    assert "q_s_logits" in pyro.get_param_store()  # the pre-seam param name survives
    assert m.event_posterior(y).shape == y.shape


def test_matched_filter_guide_guards():
    mf = np.zeros(300, dtype=np.float32)
    for bad_alpha in (0.0, -1.0):
        try:
            MatchedFilterEventGuide(mf, alpha_init=bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for alpha_init={bad_alpha}")
    try:
        MatchedFilterEventGuide(np.zeros((3, 300), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a [B, T] mf with B > 1")
    # mf shorter than the trace is a scoring mismatch, not a broadcast
    pyro.clear_param_store()
    y, inj_t = _planted_data()
    m = _model(event_guide=MatchedFilterEventGuide(np.zeros(100, dtype=np.float32)))
    try:
        m.guide(y, inj_t)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when mf is shorter than the trace")


def test_matched_filter_guide_accepts_1xt_and_is_deterministic_at_init():
    """[1, T] mf is squeezed, and the block has no RNG-drawn init to lottery on."""
    y, inj_t = _planted_data()
    mf = _mf(y, inj_t)
    g1 = MatchedFilterEventGuide(mf[None])
    g2 = MatchedFilterEventGuide(mf)
    assert g1.mf.shape == (300,)
    assert torch.allclose(g1.mf, g2.mf)

    # identical logits from two independently constructed blocks: no weight init
    pyro.clear_param_store()
    ell1 = g1.forward(y, inj_t, prior_logits=torch.zeros(300))
    pyro.clear_param_store()
    ell2 = g2.forward(y, inj_t, prior_logits=torch.zeros(300))
    assert torch.allclose(ell1, ell2)


def test_matched_filter_guide_fits_two_scalars_not_per_bin_logits():
    pyro.clear_param_store()
    y, inj_t = _planted_data()
    m = _model(event_guide=MatchedFilterEventGuide(_mf(y, inj_t)))
    hist = train_points_svi(m, y, inj_t, lr=1.5e-2, steps=60, verbose=False, seed=0)

    assert all(math.isfinite(v) for v in hist["elbo"])
    store = pyro.get_param_store()
    assert "q_s_alpha" in store and "q_s_bias" in store
    assert "q_s_logits" not in store  # the B*T free block is not installed

    gains = m.gain_posterior() if hasattr(m, "gain_posterior") else m.event_guide.gain_posterior()
    assert gains["alpha"] > 0  # positivity constraint holds through the fit
    assert m.event_posterior(y).shape == y.shape


def test_matched_filter_guide_ranking_is_pinned_to_the_filter():
    """The reason this family exists: alpha > 0 makes q a monotone map of the filter.

    So q's bin ranking IS the matched filter's ranking, exactly, for any fitted
    (alpha, b) -- detection cannot erode as the fit runs. This is the structural
    guarantee the free per-bin block does not have.
    """
    pyro.clear_param_store()
    y, inj_t = _planted_data()
    mf = _mf(y, inj_t)
    m = _model(event_guide=MatchedFilterEventGuide(mf))
    train_points_svi(m, y, inj_t, lr=1.5e-2, steps=60, verbose=False, seed=0)

    q = m.event_posterior(y)[0].numpy()
    assert np.array_equal(np.argsort(q), np.argsort(mf))  # identical ordering, post-fit


def test_matched_filter_guide_recovers_planted_events():
    """q lights up at the planted bins, and lands *on* the onset rather than near it.

    "Off-event" excludes a full kernel support (40 bins) either side: bins closer
    than that still sit inside a transient's matched-filter response, so scoring
    them as false positives would test the kernel's width, not the guide.
    """
    pyro.clear_param_store()
    y, inj_t = _planted_data()
    m = _model(event_guide=MatchedFilterEventGuide(_mf(y, inj_t)))
    train_points_svi(m, y, inj_t, lr=1.5e-2, steps=60, verbose=False, seed=0)

    q = m.event_posterior(y)[0]
    planted = torch.tensor(EVENT_BINS)
    off = torch.tensor([i for i in range(q.numel()) if min(abs(i - b) for b in EVENT_BINS) > 40])
    assert q[planted].min() > q[off].max()  # planted bins clear the off-event floor

    # onset-aligned: the matched filter is a forward correlation, so its single-event
    # response peaks at the onset bin, not at the later fluorescence peak
    for b in EVENT_BINS:
        assert int(torch.argmax(q[b - 15 : b + 16])) == 15


# --------------------------------------------------------------- warm free block


def test_warm_free_guide_guards():
    mf = np.zeros(300, dtype=np.float32)
    for bad in (0.0, -1.0):
        try:
            WarmFreeEventGuide(mf, gain=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for gain={bad}")
    try:
        WarmFreeEventGuide(mf, floor=1.0, hi=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for floor >= hi")
    try:
        WarmFreeEventGuide(np.zeros((3, 300), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a [B, T] mf with B > 1")

    # mf shorter than the trace is a scoring mismatch, same contract as the amortized block
    pyro.clear_param_store()
    y, inj_t = _planted_data()
    m = _model(event_guide=WarmFreeEventGuide(np.zeros(100, dtype=np.float32)))
    try:
        m.guide(y, inj_t)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when mf is shorter than the trace")


def test_warm_free_guide_is_the_free_family_with_a_different_start():
    """Same param, same shape, same readout -- only the initial value differs."""
    y, inj_t = _planted_data()
    mf = _mf(y, inj_t)

    pyro.clear_param_store()
    m_cold = _model(event_guide=FreeEventGuide())
    m_cold.guide(y, inj_t)
    cold0 = pyro.param("q_s_logits").detach().clone()

    pyro.clear_param_store()
    m_warm = _model(event_guide=WarmFreeEventGuide(mf))
    m_warm.guide(y, inj_t)
    warm0 = pyro.param("q_s_logits").detach().clone()

    assert warm0.shape == cold0.shape == y.shape        # identical family
    assert not torch.allclose(warm0, cold0)             # different basin
    # the cold block starts at the prior intensity: piecewise constant in phase, so it
    # carries no per-bin structure at all. The warm one varies bin to bin *within* a
    # phase, which is exactly the trace information the cold start does not have.
    assert cold0[0, :inj_t].std() < 1e-6
    assert cold0[0, inj_t:].std() < 1e-6
    assert warm0[0, :inj_t].std() > 0.1
    # and it is exactly the documented affine-and-clamp of the filter
    expected = (2.0 * (torch.as_tensor(mf) - 2.5)).clamp(-6.0, 6.0)
    assert torch.allclose(warm0[0], expected, atol=1e-5)


def test_warm_free_guide_trains_and_ranks_events():
    pyro.clear_param_store()
    y, inj_t = _planted_data()
    m = _model(event_guide=WarmFreeEventGuide(_mf(y, inj_t)))
    hist = train_points_svi(m, y, inj_t, lr=1.5e-2, steps=60, verbose=False, seed=0)

    assert all(math.isfinite(v) for v in hist["elbo"])
    assert "q_s_logits" in pyro.get_param_store()
    q = m.event_posterior(y)[0]
    planted = torch.tensor(EVENT_BINS)
    off = torch.tensor([i for i in range(q.numel()) if min(abs(i - b) for b in EVENT_BINS) > 40])
    assert q[planted].min() > q[off].max()


# ------------------------------------------------------------------- coupling flag


def test_couple_events_defaults_off_and_leaves_the_guide_unchanged():
    """The flag is additive: default False reproduces the shipped logits exactly."""
    y, inj_t = _planted_data()
    mf = _mf(y, inj_t)

    pyro.clear_param_store()
    m = _model(event_guide=MatchedFilterEventGuide(mf))
    assert m.couple_events is False
    m.guide(y, inj_t)
    ell_default = m.event_guide.fitted_logits(y).clone()

    # the block's own logits are alpha * mf + b at init -- no prior anchor added
    expected = 2.0 * torch.as_tensor(mf, dtype=y.dtype) - 5.0
    assert torch.allclose(ell_default[0], expected, atol=1e-5)


def test_couple_events_adds_a_live_two_level_anchor():
    """Coupled: q(s) picks up the prior log-odds, which is piecewise constant in phase."""
    y, inj_t = _planted_data()
    mf = _mf(y, inj_t)

    pyro.clear_param_store()
    m = _model(event_guide=MatchedFilterEventGuide(mf), couple_events=True)
    assert m.couple_events is True
    m.guide(y, inj_t)                       # materialize the guide params before reading
    q = m.event_posterior(y, inj_t)[0]

    pyro.clear_param_store()
    m_un = _model(event_guide=MatchedFilterEventGuide(mf), couple_events=False)
    m_un.guide(y, inj_t)
    q_un = m_un.event_posterior(y)[0]

    # coupling shifts q down by the prior log-odds (a sparse prior => large negative)
    assert q.max() < q_un.max()
    # the shift is constant within each phase: two levels, split at inj_t
    d = torch.logit(q.clamp(1e-9, 1 - 1e-9)) - torch.logit(q_un.clamp(1e-9, 1 - 1e-9))
    assert d[:inj_t].std() < 1e-4
    assert d[inj_t:].std() < 1e-4
    assert (d[inj_t:].mean() - d[:inj_t].mean()).abs() > 1e-3   # beta_inj is nonzero here


def test_event_posterior_requires_inj_t_under_coupling():
    """Reading the block alone under coupling over-states q -- so it is refused, not guessed."""
    pyro.clear_param_store()
    y, inj_t = _planted_data()
    m = _model(event_guide=MatchedFilterEventGuide(_mf(y, inj_t)), couple_events=True)
    m.guide(y, inj_t)
    try:
        m.event_posterior(y)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when inj_t is omitted under coupling")


def test_coupled_fit_runs_for_both_families():
    pyro.clear_param_store()
    y, inj_t = _planted_data()
    for eg in (FreeEventGuide(), MatchedFilterEventGuide(_mf(y, inj_t))):
        pyro.clear_param_store()
        m = _model(event_guide=eg, couple_events=True)
        hist = train_points_svi(m, y, inj_t, lr=1.5e-2, steps=40, verbose=False, seed=0)
        assert all(math.isfinite(v) for v in hist["elbo"])
        assert m.event_posterior(y, inj_t).shape == y.shape


# ------------------------------------------------- windowed categorical block


def _cat_guide(T=60, window=5, seed=0):
    from photosvi.models.points import WindowedCategoricalEventGuide

    torch.manual_seed(seed)
    return WindowedCategoricalEventGuide(torch.randn(T) * 2.0, window=window)


def test_windowed_categorical_marginals_sum_to_at_most_one_per_window():
    """The whole point of the block: timing alternatives compete, so they cannot
    accumulate past one event's worth of mass inside a window."""
    pyro.clear_param_store()
    T, W = 60, 5
    g = _cat_guide(T, W)
    q = torch.sigmoid(g(torch.zeros(2, T), 0, prior_logits=torch.zeros(T)))
    assert q.shape == (2, T)
    for b in range(2):
        for w in range(0, T, W):
            assert q[b, w : w + W].sum() <= 1.0 + 1e-5, (b, w)


def test_windowed_categorical_bounds_total_mass_by_window_count():
    """sum(q) <= number of windows -- which is what makes the mass read as a count.

    The independent per-bin families have no such bound: measured on the frozen GT,
    sum(q) reached 787 against 318 true onsets, and the intensity coefficients are
    fit against exactly that number.
    """
    pyro.clear_param_store()
    T, W = 100, 5
    g = _cat_guide(T, W)
    q = torch.sigmoid(g(torch.zeros(1, T), 0, prior_logits=torch.zeros(T)))
    assert float(q.sum()) <= T // W + 1e-5


def test_windowed_categorical_handles_a_ragged_tail_window():
    pyro.clear_param_store()
    T, W = 23, 5                       # 23 = 4 full windows + a 3-bin tail
    g = _cat_guide(T, W)
    q = torch.sigmoid(g(torch.zeros(1, T), 0, prior_logits=torch.zeros(T)))
    assert q.shape == (1, T)
    assert float(q[0, 20:23].sum()) <= 1.0 + 1e-5


def test_windowed_categorical_fitted_logits_match_forward():
    pyro.clear_param_store()
    T = 40
    g = _cat_guide(T)
    y = torch.zeros(1, T)
    assert torch.allclose(g(y, 0, prior_logits=torch.zeros(T)), g.fitted_logits(y))


def test_windowed_categorical_rejects_a_degenerate_window():
    from photosvi.models.points import WindowedCategoricalEventGuide

    with pytest.raises(ValueError, match="window"):
        WindowedCategoricalEventGuide(torch.randn(20), window=1)
