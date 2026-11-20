"""The opt-in noise-unit mark mode: a_t = sigma_y r_t, r_t ~ LogNormal(mu_r, sigma_r^2), fixed."""
from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from photosvi.experiments.marks import (
    NOISE_UNITS_MU,
    NOISE_UNITS_SIGMA,
    fix_marks_at,
    noise_unit_marks,
    noise_unit_mode,
)


def test_the_mode_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("MARK_MODE", raising=False)
    assert noise_unit_mode({}) is None
    assert noise_unit_mode({"mark_mode": "noise_units"}) == (NOISE_UNITS_MU, NOISE_UNITS_SIGMA)
    monkeypatch.setenv("MARK_MODE", "noise_units")
    assert noise_unit_mode({"mark_noise_units": {"mu": 1.0, "sigma": 0.5}}) == (1.0, 0.5)
    monkeypatch.setenv("MARK_MODE", "nonsense")
    with pytest.raises(ValueError):
        noise_unit_mode({})


def test_marks_are_in_units_of_the_noise():
    mu, sig = noise_unit_marks(0.02)
    assert mu == pytest.approx(math.log(0.02) + NOISE_UNITS_MU) and sig == NOISE_UNITS_SIGMA


def test_the_simulated_cohort_draws_marks_in_noise_units(monkeypatch):
    from photosvi.experiments.cohort import spec_from_config
    cfg = yaml.safe_load(open("configs/cohort_power_48.yaml"))
    monkeypatch.delenv("MARK_MODE", raising=False)
    assert spec_from_config(cfg, 0.0).mu_a == cfg["cohort"]["mu_a"]
    monkeypatch.setenv("MARK_MODE", "noise_units")
    sp = spec_from_config(cfg, 0.0)
    assert sp.mu_a == pytest.approx(math.log(cfg["cohort"]["sigma_y"]) + NOISE_UNITS_MU)
    assert sp.sigma_a == NOISE_UNITS_SIGMA


def test_fixed_marks_are_not_fitted():
    import torch

    from photosvi.models.points import PhasicEventProcess, build_kernel
    k = build_kernel({"family": "biexp", "k_rise": 20.0, "k_decay": 5.0}, 50, 0.04)
    ph = PhasicEventProcess(k, beta0=-2.0, beta_inj=0.0, mu_a=0.0, sigma_a=1.0, dt=0.04,
                            history=None)
    ph.set_mark_prior(0.0, 1.0)
    fix_marks_at(ph, -3.0, 0.63)
    assert float(ph.mu_a) == pytest.approx(-3.0)
    assert float(ph.sigma_a) == pytest.approx(0.63)
    assert not ph.mu_a.requires_grad and not ph.log_sigma_a.requires_grad
    assert ph.mark_prior is None
    assert isinstance(ph.mu_a, torch.nn.Parameter)


def test_the_api_fixes_the_marks_at_its_noise_estimate():
    from photosvi.api import FitSettings, Session, fit_session
    rng = np.random.default_rng(0)
    y = 0.05 * rng.standard_normal(1200)
    y[::97] += 0.4
    r = fit_session(Session(trace=y, fs=25.0, inj_t=600),
                    FitSettings(steps=3, max_steps=3, threads=1,
                                mark_noise_units=[NOISE_UNITS_MU, NOISE_UNITS_SIGMA]))
    assert r.mu_a == pytest.approx(math.log(r.sigma_y) + NOISE_UNITS_MU, abs=1e-5)
    assert r.sigma_a == pytest.approx(NOISE_UNITS_SIGMA, abs=1e-5)


# ------------------------------------------------------- per-session noise scale
def _small_cohort(monkeypatch, per_session: bool):
    from photosvi.experiments.cohort import build_cohort_model, simulate_cohort, spec_from_config
    cfg = yaml.safe_load(open("configs/cohort_power_48.yaml"))
    cfg["cohort"] |= dict(n_animals=2, T=4800, inj_t=2400, inj_jitter=0)
    monkeypatch.setenv("MARK_MODE", "noise_units")
    monkeypatch.setenv("SIGMA_Y_PER_SESSION", "1" if per_session else "0")
    spec = spec_from_config(cfg, 0.5)
    coh = simulate_cohort(spec, 0)
    coh["y"][1] = coh["y"][1] * 3.0          # one session three times as noisy
    model, Y, _ = build_cohort_model(cfg, spec, coh)
    return model, Y, spec


def test_each_session_gets_its_own_noise_scale_and_marks(monkeypatch):
    model, Y, spec = _small_cohort(monkeypatch, per_session=True)
    sig = model.emission.sigma_y.detach().reshape(-1)
    assert sig.numel() == Y.shape[0]
    assert float(sig[1] / sig[0]) == pytest.approx(3.0, rel=0.25)
    mu = model.phasic.mu_a.detach().reshape(-1)
    assert mu.numel() == Y.shape[0] and not model.phasic.mu_a.requires_grad
    assert torch_allclose(mu, sig.log() + NOISE_UNITS_MU)
    assert float(model.phasic.sigma_a) == pytest.approx(NOISE_UNITS_SIGMA)


def test_per_session_scales_reach_the_likelihood(monkeypatch):
    """Scaling one session's sigma_y changes that session's emission density, not another's."""
    import torch
    model, Y, _ = _small_cohort(monkeypatch, per_session=True)
    mean = torch.zeros_like(Y)
    lp0 = model._emission_dist(mean, obs=Y).log_prob(Y).sum(-1)
    with torch.no_grad():
        model.emission.log_sigma_y[1] += 0.5
    lp1 = model._emission_dist(mean, obs=Y).log_prob(Y).sum(-1)
    assert torch.allclose(lp0[0], lp1[0]) and not torch.allclose(lp0[1], lp1[1])


def test_without_the_switch_the_cohort_keeps_one_scale(monkeypatch):
    model, Y, spec = _small_cohort(monkeypatch, per_session=False)
    assert model.emission.log_sigma_y.dim() == 0
    assert float(model.emission.sigma_y) == pytest.approx(spec.sigma_y)


def torch_allclose(a, b):
    import torch
    return bool(torch.allclose(a, b, atol=1e-5))


def test_a_per_session_cohort_fit_takes_steps(monkeypatch):
    import pyro
    import torch
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import ClippedAdam
    model, Y, _ = _small_cohort(monkeypatch, per_session=True)
    pyro.clear_param_store()
    svi = SVI(model.model, model.guide, ClippedAdam({"lr": 0.01}), Trace_ELBO(max_plate_nesting=2))
    for _ in range(3):
        assert math.isfinite(svi.step(Y))
    loc = pyro.param("q_a_loc").detach()
    assert loc.shape[0] == Y.shape[0]
    assert not torch.allclose(loc[0, 0], loc[1, 0])     # each session starts at its own scale
