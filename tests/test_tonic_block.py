"""The opt-in joint block over the per-drug tonic means (fit.tonic_block / TONIC_BLOCK)."""
from __future__ import annotations

import math

import pyro
import pytest
import torch
import yaml
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam


def _fit(monkeypatch, block: bool, steps: int = 5):
    from photosvi.experiments.cohort import build_cohort_model, simulate_cohort, spec_from_config
    cfg = yaml.safe_load(open("configs/cohort_power_48.yaml"))
    cfg["cohort"] |= dict(n_animals=2, T=4800, inj_t=2400, inj_jitter=0)
    monkeypatch.setenv("TONIC_BLOCK", block if isinstance(block, str) else ("1" if block else "0"))
    spec = spec_from_config(cfg, 0.5)
    model, Y, _ = build_cohort_model(cfg, spec, simulate_cohort(spec, 0))
    pyro.clear_param_store()
    svi = SVI(model.model, model.guide, ClippedAdam({"lr": 0.01}), Trace_ELBO(max_plate_nesting=2))
    for _ in range(steps):
        assert math.isfinite(svi.step(Y))
    return model


def test_the_block_replaces_the_factorised_tonic_means(monkeypatch):
    model = _fit(monkeypatch, block=True)
    store = pyro.get_param_store()
    assert model.tonic_block
    assert "q_tonic_block_loc" in store and "q_tonic_mu_alpha_loc" not in store
    D = model.n_drugs
    assert store["q_tonic_block_tril"].shape == (3 * D, 3 * D)
    loc, sd = model.tonic_contrast(1, 0, "alpha")
    assert math.isfinite(loc) and sd > 0
    post = model.tonic_posterior()
    assert set(post) == set(range(D)) and all(v["peak"] > 0 for v in post.values())
    assert len(model.session_tonic()["peak"]) == model.n_sessions


def test_off_by_default_the_tonic_means_stay_factorised(monkeypatch):
    model = _fit(monkeypatch, block=False)
    store = pyro.get_param_store()
    assert not model.tonic_block
    assert "q_tonic_mu_alpha_loc" in store and "q_tonic_block_loc" not in store
    loc, sd = model.tonic_contrast(1, 0, "alpha")
    s = store["q_tonic_mu_alpha_scale"].detach()
    assert sd == pytest.approx(float(torch.sqrt(s[0] ** 2 + s[1] ** 2)), rel=1e-5)


def test_the_block_starts_where_the_factorised_guide_does(monkeypatch):
    """Same anchors and the same 0.1 width, so switching it on changes the family only."""
    model = _fit(monkeypatch, block=True, steps=1)
    loc0 = torch.cat([model.tonic_anchor[n] for n in ("alpha", "rho", "gamma")])
    assert torch.allclose(pyro.param("q_tonic_block_loc").detach().cpu(), loc0, atol=0.05)


@pytest.mark.parametrize("mode", ["with_sessions", "with_sessions_level_drift"])
def test_the_extended_blocks_carry_their_latents(monkeypatch, mode):
    model = _fit(monkeypatch, block=mode)
    store = pyro.get_param_store()
    D, S = model.n_drugs, model.n_sessions
    n = 3 * D + 3 * S + (2 * S if mode.endswith("drift") else 0)
    assert model.tonic_block_mode == mode
    assert store["q_tonic_block_tril"].shape == (n, n)
    assert "q_z_tonic_alpha_loc" not in store
    if mode.endswith("drift"):
        assert model.bayesian_offsets and "q_tonic_b_loc" not in store
    loc, sd = model.tonic_alpha_contrast()
    assert math.isfinite(loc) and sd > 0
    assert len(model.session_tonic()["peak"]) == S
    model.tonic_guide_widths()
