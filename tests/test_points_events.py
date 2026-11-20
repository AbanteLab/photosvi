"""Smoke tests for the points phasic event block.

Covers the forward-generative side only: kernel shape invariants, the causal
convolution, the timing log-likelihood, and the `sample` forward pass.
Posterior inference is not built here.
"""

from __future__ import annotations

import math

import pytest
import torch

from photosvi.models.points import (
    PhasicEventProcess,
    bi_exponential_kernel,
)


def test_bi_exponential_kernel_shape():
    h = bi_exponential_kernel(200, k_rise=2.0, k_decay=0.5, dt=1 / 25)
    assert h.shape == (200,)
    assert h[0].item() == 0.0            # causal: zero at lag 0
    assert bool((h >= 0).all())          # nonnegative
    assert abs(h.max().item() - 1.0) < 1e-6   # peak-normalized
    assert h.argmax().item() > 0         # rise then decay: peak at a positive lag


def test_convolve_reproduces_kernel_on_unit_impulse():
    """A single unit impulse should stamp one (flipped-back) copy of h."""
    h = bi_exponential_kernel(50, k_rise=2.0, k_decay=0.5)
    proc = PhasicEventProcess(h)
    T = 120
    u = torch.zeros(1, T)
    u[0, 30] = 1.0
    c = proc.convolve(u)
    assert c.shape == (1, T)
    # event at t=30 deposits h_k at t=30+k
    assert torch.allclose(c[0, 30:30 + 50], h, atol=1e-5)
    assert torch.allclose(c[0, :30], torch.zeros(30), atol=1e-6)  # causal: nothing before


def test_convolve_is_linear_superposition():
    h = bi_exponential_kernel(40, k_rise=2.0, k_decay=0.5)
    proc = PhasicEventProcess(h)
    T = 100
    u = torch.zeros(1, T)
    u[0, 10] = 1.0
    u[0, 20] = 2.0
    c = proc.convolve(u)
    expected = torch.zeros(T)
    expected[10:10 + 40] += h
    expected[20:20 + 40] += 2.0 * h
    assert torch.allclose(c[0], expected, atol=1e-5)


def test_event_logprob_matches_manual_bernoulli():
    """With no history, the timing LL equals a hand-rolled Bernoulli sum."""
    h = bi_exponential_kernel(20, k_rise=2.0, k_decay=0.5)
    proc = PhasicEventProcess(h, beta0=-2.0, beta_inj=0.0, dt=1 / 25)
    s = torch.tensor([[0.0, 1.0, 0.0, 1.0, 1.0]])
    post = torch.zeros_like(s)

    ll = proc.event_logprob(s, post)
    assert ll.shape == (1,)
    assert math.isfinite(ll.item())

    # manual: constant rate, p = 1 - exp(-lambda dt)
    lam = math.exp(-2.0)
    p = 1.0 - math.exp(-lam * (1 / 25))
    manual = sum(
        math.log(p) if si == 1 else math.log(1 - p)
        for si in s[0].tolist()
    )
    assert abs(ll.item() - manual) < 1e-4


def test_event_logprob_post_injection_shift():
    """beta_inj > 0 raises the rate after injection -> events likelier there."""
    h = bi_exponential_kernel(20, k_rise=2.0, k_decay=0.5)
    proc = PhasicEventProcess(h, beta0=-3.0, beta_inj=2.0)
    # one event before injection vs the same event after
    s_pre = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    s_post = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    post = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    assert proc.event_logprob(s_post, post).item() > proc.event_logprob(s_pre, post).item()


def test_history_term_refractory_lowers_logprob():
    """A back-to-back pair of events is less likely under a refractory FIR."""
    h = bi_exponential_kernel(20, k_rise=2.0, k_decay=0.5)
    proc = PhasicEventProcess(h, beta0=-1.0, history=[-5.0])  # w_1 strongly negative
    post = torch.zeros(1, 4)
    adjacent = torch.tensor([[1.0, 1.0, 0.0, 0.0]])   # second event right after first
    spaced = torch.tensor([[1.0, 0.0, 0.0, 1.0]])     # same count, well separated
    assert proc.event_logprob(adjacent, post).item() < proc.event_logprob(spaced, post).item()


def test_sample_shapes_and_binary_events():
    h = bi_exponential_kernel(50, k_rise=2.0, k_decay=0.5)
    proc = PhasicEventProcess(h, beta0=-2.0, beta_inj=1.0)
    g = torch.Generator().manual_seed(0)
    out = proc.sample(T=500, inj_t=250, batch=3, generator=g)
    for key in ("s", "a", "u", "c"):
        assert out[key].shape == (3, 500)
    assert set(out["s"].unique().tolist()).issubset({0.0, 1.0})
    assert bool((out["a"] > 0).all())                 # log-normal marks are positive
    assert torch.allclose(out["u"], out["s"] * out["a"])  # drive masks marks by events
    assert torch.isfinite(out["c"]).all()


def test_sample_rate_responds_to_beta0():
    """A near-silent baseline fires far fewer events than a hot one."""
    h = bi_exponential_kernel(50, k_rise=2.0, k_decay=0.5)
    g = torch.Generator().manual_seed(0)
    cold = PhasicEventProcess(h, beta0=-12.0).sample(T=2000, inj_t=0, batch=1, generator=g)
    g = torch.Generator().manual_seed(0)
    hot = PhasicEventProcess(h, beta0=0.0).sample(T=2000, inj_t=0, batch=1, generator=g)
    assert hot["s"].sum().item() > cold["s"].sum().item()
    assert cold["s"].sum().item() < 5  # essentially silent


def test_the_biphasic_basis_is_mean_free_at_any_length():
    """Truncating the support does not cost the property the paper's argument rests on.

    The opposite looks obvious -- cut off a kernel whose undershoot runs for six
    seconds and surely the areas stop cancelling -- and it is wrong, because the
    builder sets the slow lobe's scale from the sums over whatever length it is given.
    Asserted because the belief that it needs a longer support was written into a
    config comment before anyone measured it.

    Checked on K7, the slow candidate added 2026-09-30, which is where it matters:
    K5 decays inside 1.5 s and never comes close to its own truncation.
    """
    import numpy as np

    from photosvi.models.points import build_kernel

    K7 = dict(family="biphasic", k_rise_fast=5.0, k_decay_fast=2.25,
              k_rise_slow=2.0, k_decay_slow=0.875)
    ref = None
    for taps in (125, 250, 500, 700):
        k = build_kernel(K7, taps, 1 / 25).numpy()
        assert abs(k.sum()) < 1e-4, f"{taps} taps: sum(h) = {k.sum()}"
        assert abs(k.max() - 1.0) < 1e-6, "peak-normalised at every length"
        half = np.where(k >= k.max() / 2)[0]
        fwhm = (half[-1] - half[0] + 1) / 25
        if ref is None:
            ref = fwhm
        else:
            assert abs(fwhm - ref) < 1e-9, f"{taps} taps: width moved to {fwhm}"
    # The undershoot is the one thing that does move, and only at the shortest.
    deep = build_kernel(K7, 125, 1 / 25).numpy().min()
    long = build_kernel(K7, 700, 1 / 25).numpy().min()
    assert abs(deep - long) < 0.02, f"undershoot moved from {long} to {deep}"


def test_masking_an_artefact_window_has_to_widen_under_ar1():
    """One masked bin is not enough when the residual is correlated.

    The AR(1) likelihood factorises as p(y_1) + sum_t p(y_t | y_{t-1}), so dropping
    the term AT a bad bin leaves the term AFTER it conditioning on the bad value. This
    is not academic: it is the injection handling window, where the fibre is moved and
    the trace is unusable, and it sits exactly where the rate contrast is computed.
    """
    import torch

    from photosvi.models.points.emission import AdditiveEmission
    from photosvi.models.points.points import observed_bins

    T, BAD = 12, 5
    y = torch.zeros(1, T)
    y[0, BAD] = 50.0                      # the artefact, 50 sigma
    mean = torch.zeros(1, T)
    naive = torch.ones(1, T, dtype=torch.bool)
    naive[0, BAD] = False

    def per_bin(em, obs):
        """Pyro's em.mask() zeroes per-bin terms, so score them the same way."""
        if not em.ar_rho:
            return em._dist(mean).log_prob(obs)
        r = obs - mean
        first = em._dist(mean[..., :1]).log_prob(obs[..., :1])
        cond = mean[..., 1:] + em.ar_rho * r[..., :-1]
        tail = em._dist(cond, scale_mul=(1 - em.ar_rho**2) ** 0.5).log_prob(obs[..., 1:])
        return torch.cat([first, tail], dim=-1)

    # what a trace with no artefact in it scores, as the reference for "clean"
    spotless = torch.zeros(1, T)

    # white: the naive mask is already the right one, and widening is not applied
    white = AdditiveEmission(sigma_y=1.0, ar_rho=0.0)
    assert torch.equal(observed_bins(naive, y.shape, y.device, len(white.ar_coeffs)), naive)
    ref = (per_bin(white, spotless) * naive).sum()
    assert abs((per_bin(white, y) * naive).sum() - ref) < 1e-4

    # AR(1): the naive mask leaves most of the damage, the widened one removes it
    ar = AdditiveEmission(sigma_y=1.0, ar_rho=0.9)
    lp = per_bin(ar, y)
    wide = observed_bins(naive, y.shape, y.device, len(ar.ar_coeffs))
    assert not wide[0, BAD] and not wide[0, BAD + 1]   # the successor goes too
    assert wide.sum() == T - 2                         # and nothing else does
    leaked = (lp * naive).sum()
    fixed = (lp * wide).sum()
    ref_ar = (per_bin(ar, spotless) * wide).sum()
    assert leaked < -1000, f"expected the naive mask to leak, got {leaked}"
    assert abs(fixed - ref_ar) < 1e-4, f"expected {ref_ar} once widened, got {fixed}"


# ----------------------------------------------------------------- gamma family

K9 = dict(family="bigamma", shape_fast=20.34, rate_fast=15.65,
          shape_slow=15.40, rate_slow=3.95)


def test_the_gamma_basis_is_mean_free_and_peaked():
    import torch

    from photosvi.models.points import build_kernel

    h = build_kernel(K9, 400, 1 / 48)
    assert abs(h.sum() / h.abs().sum()) < 1e-5      # the property beta_inj needs
    assert h.max() == pytest.approx(1.0)            # peak-normalized
    assert h[0] == 0.0                              # smooth onset, unlike a biexp
    assert h.argmax().item() / 48 == pytest.approx(1.23, abs=0.02)
    assert -0.35 < h.min() < -0.25                  # one undershoot, not a cliff
    assert torch.isfinite(h).all()


def test_the_gamma_basis_refuses_a_support_that_truncates_it():
    """The biphasic family tolerates any length; this one does not, and must say so.

    Truncating before the slow lobe has decayed forces a larger subtraction, and the
    undershoot grows to swallow the transient -- 2.7x the peak at 2.1 s -- while
    sum(h) = 0 holds the whole way, so the mean-free property gives no warning at all.
    """
    from photosvi.models.points import build_kernel
    from photosvi.models.points.events import bigamma_kernel

    for taps in (100, 150, 250):
        with pytest.raises(ValueError, match="truncates this kernel"):
            build_kernel(K9, taps, 1 / 48)
    for taps in (300, 400, 600):
        build_kernel(K9, taps, 1 / 48)               # these are fine

    # the failure the guard is for, reachable only on purpose
    bad = bigamma_kernel(100, 20.34, 15.65, 15.40, 3.95, dt=1 / 48, undershoot_tol=0)
    assert bad.min() < -2.0
    assert abs(bad.sum() / bad.abs().sum()) < 1e-5   # mean-free, and still useless

    # and the advice it gives is actually sufficient
    msg = str(pytest.raises(ValueError, build_kernel, K9, 250, 1 / 48).value)
    need = int(msg.split("Use at least ")[1].split(" taps")[0])
    assert build_kernel(K9, need, 1 / 48) is not None


def test_the_gamma_lobes_have_to_be_ordered_and_finite():
    from photosvi.models.points.events import bigamma_kernel, gamma_lobe

    with pytest.raises(ValueError, match="shape > 1"):
        gamma_lobe(100, 1.0, 5.0)
    with pytest.raises(ValueError, match="rate > 0"):
        gamma_lobe(100, 3.0, 0.0)
    with pytest.raises(ValueError, match="slower tail"):
        bigamma_kernel(400, 20.0, 4.0, 15.0, 4.0, dt=1 / 48)
    # a large shape must not overflow before the exponential pulls it down
    assert gamma_lobe(400, 60.0, 20.0, dt=1 / 48).max() == pytest.approx(1.0)


def test_observed_bins_widens_by_the_ar_order():
    """Under AR(p) a masked bin takes the p following ones with it.

    The widening used to key on `ar_rho`, which `AdditiveEmission` sets to zero for any
    order above one, so an AR(2) emission got the naive mask.
    """
    import torch

    from photosvi.models.points.emission import AdditiveEmission
    from photosvi.models.points.points import observed_bins

    T, BAD = 12, 5
    naive = torch.ones(1, T, dtype=torch.bool)
    naive[0, BAD] = False
    ar2 = AdditiveEmission(sigma_y=1.0, ar_coeffs=(1.432, -0.4464))
    assert ar2.ar_rho == 0.0                     # the trap the old key fell into
    wide = observed_bins(naive, (1, T), "cpu", len(ar2.ar_coeffs))
    assert not wide[0, BAD:BAD + 3].any()        # the bin and its two successors
    assert wide.sum() == T - 3
    # a mask reaching the end of the trace does not wrap round
    tail = torch.ones(1, T, dtype=torch.bool)
    tail[0, -1] = False
    assert observed_bins(tail, (1, T), "cpu", 2).sum() == T - 1
