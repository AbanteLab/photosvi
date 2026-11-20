"""Smoke tests for the points additive-mean emission.

Covers the emission node only: the Gaussian and Student-t per-bin
densities on the additive mean m = x_hat + c, the emission block shape
(summed over time, [B]), the robustness ordering (a far outlier costs less NLL
under Student-t than Gaussian), the nu -> inf Gaussian limit, gradient flow to
sigma_y/nu, the scale-mixture forward sample, and assembly against the
PhasicEventProcess + TonicBaseline carriers. The auxiliary precision w_t is only
materialized by `sample`; SVI over it is not built here.
"""

from __future__ import annotations

import math

import torch

from photosvi.models.points import (
    AdditiveEmission,
    PhasicEventProcess,
    TonicBaseline,
    bi_exponential_kernel,
    estimate_log_noise_scale,
)


def test_constructor_rejects_bad_scale_and_dof():
    for kw in (dict(sigma_y=0.0), dict(sigma_y=-1.0), dict(nu=0.0), dict(nu=-3.0)):
        try:
            AdditiveEmission(**kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kw}")


def test_constrained_properties_round_trip():
    em = AdditiveEmission(sigma_y=0.3, nu=5.0)
    assert abs(em.sigma_y.item() - 0.3) < 1e-6
    assert em.robust is True
    assert abs(em.nu.item() - 5.0) < 1e-5
    g = AdditiveEmission(sigma_y=0.2)
    assert g.robust is False and g.nu is None


def test_gaussian_log_prob_matches_closed_form():
    """log_prob is the eq-2 Gaussian block summed over t, shape [B]."""
    em = AdditiveEmission(sigma_y=0.5)
    mean = torch.zeros(2, 100)
    y = 0.3 * torch.ones(2, 100)
    lp = em.log_prob(y, mean)
    assert lp.shape == (2,)
    sig = 0.5
    per_bin = -0.5 * math.log(2 * math.pi * sig**2) - (0.3**2) / (2 * sig**2)
    assert torch.allclose(lp, torch.full((2,), 100 * per_bin), atol=1e-4)


def test_studentt_downweights_outlier_vs_gaussian():
    """A far outlier costs far less NLL under Student-t than Gaussian."""
    mean = torch.zeros(1, 1)
    y = torch.tensor([[8.0]])  # ~16 sigma out for sigma_y=0.5
    gauss = AdditiveEmission(sigma_y=0.5)
    robust = AdditiveEmission(sigma_y=0.5, nu=4.0)
    # negative log-likelihood (per the single outlier bin)
    nll_g = -gauss.log_prob(y, mean).item()
    nll_t = -robust.log_prob(y, mean).item()
    assert nll_t < nll_g            # heavy tails are cheaper for the outlier
    assert nll_g - nll_t > 50       # Gaussian grows quadratically, t logarithmically


def test_studentt_approaches_gaussian_as_nu_grows():
    """As nu grows the Student-t log-density approaches the Gaussian.

    Asserted as a *monotone* approach (larger nu lands closer to the Gaussian)
    rather than equality at a huge nu: the StudentT normalizer differences two
    large lgamma terms, which lose precision in float32 well before nu = inf, so
    the clean numerical statement is convergence, not exact coincidence.
    """
    mean = torch.zeros(1, 200)
    y = 0.4 * torch.randn(1, 200, generator=torch.Generator().manual_seed(0))
    lp_g = AdditiveEmission(sigma_y=0.5).log_prob(y, mean).item()
    lp_small = AdditiveEmission(sigma_y=0.5, nu=4.0).log_prob(y, mean).item()
    lp_large = AdditiveEmission(sigma_y=0.5, nu=200.0).log_prob(y, mean).item()
    assert abs(lp_large - lp_g) < abs(lp_small - lp_g)   # larger nu -> nearer Gaussian
    assert abs(lp_large - lp_g) < 0.5                    # and already close by nu=200


def test_log_prob_is_differentiable_in_sigma_and_nu():
    em = AdditiveEmission(sigma_y=0.5, nu=4.0)
    mean = torch.zeros(1, 50)
    y = 0.3 * torch.ones(1, 50)
    em.log_prob(y, mean).sum().backward()
    for name in ("log_sigma_y", "log_nu"):
        g = getattr(em, name).grad
        assert g is not None and torch.isfinite(g).all()


def test_sample_shapes_and_scale_mixture():
    """Gaussian draw has no w; robust draw materializes per-bin precisions."""
    mean = torch.zeros(3, 500)
    g = AdditiveEmission(sigma_y=0.2)
    out_g = g.sample(mean, generator=torch.Generator().manual_seed(0))
    assert out_g["y"].shape == (3, 500) and out_g["w"] is None
    assert torch.isfinite(out_g["y"]).all()
    # empirical residual SD is ~ sigma_y for the Gaussian draw
    assert abs(out_g["y"].std().item() - 0.2) < 0.03

    torch.manual_seed(0)
    r = AdditiveEmission(sigma_y=0.2, nu=6.0)
    out_r = r.sample(mean)
    assert out_r["y"].shape == (3, 500)
    assert out_r["w"].shape == (3, 500)
    assert (out_r["w"] > 0).all() and torch.isfinite(out_r["y"]).all()


def test_assembles_against_carriers():
    """The emission consumes the additive mean x_hat + c from the two carriers."""
    T, inj_t, batch = 1500, 750, 3
    tb = TonicBaseline(b=0.0, peak=1.0, k_rise=0.01, k_decay=0.002)
    proc = PhasicEventProcess(bi_exponential_kernel(200, 2.0, 0.5), beta0=-2.0, beta_inj=1.0)
    g = torch.Generator().manual_seed(0)
    c = proc.sample(T=T, inj_t=inj_t, batch=batch, generator=g)["c"]
    mean = tb.baseline(T=T, inj_t=inj_t) + c          # [batch, T]

    em = AdditiveEmission(sigma_y=0.1)
    out = em.sample(mean, generator=g)
    assert out["y"].shape == (batch, T)
    lp = em.log_prob(out["y"], mean)
    assert lp.shape == (batch,) and torch.isfinite(lp).all()
    # the realized trace differs from the mean by emission noise only
    assert torch.allclose(em.residual(out["y"], mean), out["y"] - mean)


def test_estimate_log_noise_scale_recovers_sigma():
    """The high-frequency anchor recovers the true noise std from a noisy trace."""
    T = 4000
    true_sigma = 0.3
    g = torch.Generator().manual_seed(0)
    y = true_sigma * torch.randn(T, generator=g)
    sigma_hat = math.exp(estimate_log_noise_scale(y))
    # MAD-of-differences estimator is unbiased for Gaussian noise; loose tol on 4k samples
    assert abs(sigma_hat - true_sigma) < 0.05, f"sigma_hat={sigma_hat:.3f}"


def test_estimate_log_noise_scale_ignores_slow_trend():
    """The estimate is (near-)invariant to slow structure -- the high-pass property.

    This is the whole point: sigma_y must anchor on genuine high-frequency noise,
    not on the slow tonic structure whose misfit is what inflates a free sigma_y.
    """
    T = 4000
    true_sigma = 0.3
    g = torch.Generator().manual_seed(1)
    noise = true_sigma * torch.randn(T, generator=g)
    t = torch.linspace(0.0, 1.0, T)
    flat = math.exp(estimate_log_noise_scale(noise))
    # add a large slow trend (amplitude 10x the noise) -- a smooth ramp + bump
    trend = 3.0 * t + 2.0 * torch.sin(2 * math.pi * t)
    with_trend = math.exp(estimate_log_noise_scale(noise + trend))
    assert abs(with_trend - flat) < 0.02, f"trend leaked: flat={flat:.3f} with_trend={with_trend:.3f}"


def test_estimate_log_noise_scale_floors_and_shapes():
    """Degenerate input floors at log(min_scale); [T] and [B, T] both accepted."""
    floored = math.exp(estimate_log_noise_scale(torch.zeros(500), min_scale=1e-3))
    assert abs(floored - 1e-3) < 1e-9

    g = torch.Generator().manual_seed(2)
    batched = 0.2 * torch.randn(4, 1000, generator=g)
    val = estimate_log_noise_scale(batched, inj_t=500)  # pre-injection window only
    assert math.isfinite(val) and abs(math.exp(val) - 0.2) < 0.05
