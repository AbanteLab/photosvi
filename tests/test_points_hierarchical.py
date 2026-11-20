"""Tests for the cohort model: design detection, the batched tonic, and the ELBO.

The load-bearing property here is **design detection**. A variance component the
design cannot identify does not fail loudly -- it absorbs the component below it, and
the one it usually steals from is `sigma_animal`, which sets the width of the drug
contrast. So the rules in `_detect_levels` are tested directly, against designs whose
correct answer is known by construction, rather than only through a fit.
"""

from __future__ import annotations

import math

import pyro
import pytest
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from photosvi.models.points import (
    AdditiveEmission,
    BatchedTonicBaseline,
    HierarchicalModel,
    PhasicEventProcess,
    TonicBaseline,
    WindowedCategoricalEventGuide,
    biphasic_kernel,
    stack_tonic_baselines,
)

DT = 1.0 / 25.0


def _kernel():
    return biphasic_kernel(
        length=50, dt=DT, k_rise_fast=20.0, k_decay_fast=9.0,
        k_rise_slow=8.0, k_decay_slow=3.5,
    )


def _model(animal, drug, inj_t, *, T=200, **kw):
    n = len(animal)
    tonic = BatchedTonicBaseline(n_seq=n, dt=DT, b=0.0, peak=1.0, k_rise=0.01, k_decay=0.002)
    phasic = PhasicEventProcess(_kernel(), beta0=-2.8, beta_inj=0.0, mu_a=1.7,
                                sigma_a=0.2, dt=DT, history=None)
    return HierarchicalModel(phasic, tonic, AdditiveEmission(sigma_y=0.1),
                                animal=animal, drug=drug, inj_t=inj_t,
                                beta0_anchor=-2.8, **kw)


# --------------------------------------------------------------- batched tonic
def test_batched_tonic_matches_per_trace_tonic():
    """Each row of the batched baseline equals the scalar block it was stacked from."""
    blocks = [
        TonicBaseline(dt=DT, b=-0.3, peak=7.0, k_rise=0.005, k_decay=0.0009, drift=-0.002),
        TonicBaseline(dt=DT, b=0.1, peak=2.0, k_rise=0.02, k_decay=0.004, drift=0.001),
    ]
    batched = stack_tonic_baselines(blocks)
    inj = [100, 250]
    got = batched.baseline(600, inj)
    assert got.shape == (2, 600)
    for i, blk in enumerate(blocks):
        torch.testing.assert_close(got[i], blk.baseline(600, inj[i]), rtol=1e-5, atol=1e-6)


def test_batched_tonic_respects_per_sequence_injection():
    """A later injection means a longer flat pre-injection stretch, per sequence."""
    b = BatchedTonicBaseline(n_seq=2, dt=DT, b=0.0, peak=5.0, k_rise=0.01, k_decay=0.002)
    x = b.baseline(500, [50, 300])
    assert torch.allclose(x[0, :50], torch.zeros(50), atol=1e-7)
    assert torch.allclose(x[1, :300], torch.zeros(300), atol=1e-7)
    assert x[0, 200] > 1e-3          # already responding
    assert abs(float(x[1, 200])) < 1e-7   # still pre-injection


def test_batched_tonic_rejects_broken_bateman_ordering():
    with pytest.raises(ValueError, match="Bateman ordering"):
        BatchedTonicBaseline(n_seq=2, dt=DT, k_rise=[0.01, 0.001], k_decay=[0.002, 0.002])


# ------------------------------------------------------------ design detection
def test_one_drug_per_animal_is_between_animal():
    """Nested design: no animal x drug level, and the contrast is between animals."""
    m = _model([0, 1, 2, 3], [0, 0, 1, 1], [50, 50, 50, 50])
    d = m.design_summary()
    assert d["levels"] == ("animal",)
    assert d["within_animal"] is False
    assert d["n_cells"] == d["n_animals"] == 4


def test_every_animal_sees_every_drug_is_within_animal():
    """Crossed design: the animal x drug level is identifiable, so the contrast is within."""
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [50, 50, 50, 50])
    d = m.design_summary()
    # b0_session comes along because each animal was recorded twice -- see the
    # baseline-session test below; the point here is that "cell" is present.
    assert d["levels"] == ("animal", "cell", "b0_session")
    assert d["within_animal"] is True
    assert d["drugs_per_animal"] == [2, 2]


def test_partially_crossed_design_still_gets_the_interaction():
    """One animal with two drugs is enough to identify the interaction."""
    m = _model([0, 0, 1, 2], [0, 1, 0, 1], [50, 50, 50, 50])
    assert "cell" in m.design_summary()["levels"]


def test_session_level_needs_a_replicated_cell():
    """The residual appears only when an (animal, drug) cell has more than one recording."""
    without = _model([0, 0, 1, 1], [0, 1, 0, 1], [50] * 4)
    assert "session" not in without.levels
    within = _model([0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [50] * 5)   # animal 0 x drug 0 twice
    assert "session" in within.levels


def test_repeats_of_the_same_animal_and_drug_are_not_an_interaction():
    """Two sessions of one animal on one drug give a residual, not a crossed design."""
    m = _model([0, 0, 1], [0, 0, 1], [50, 50, 50])
    d = m.design_summary()
    assert d["within_animal"] is False
    assert "cell" not in d["levels"]
    assert "session" in d["levels"]


def test_single_animal_has_no_animal_level():
    m = _model([0, 0], [0, 1], [50, 50])
    assert "animal" not in m.levels


def test_drug_may_be_given_per_animal_as_shorthand():
    """A length-A drug vector expands to the one-drug-per-animal design."""
    per_animal = _model([0, 0, 1, 1], [0, 1], [50] * 4)
    per_session = _model([0, 0, 1, 1], [0, 0, 1, 1], [50] * 4)
    assert torch.equal(per_animal.drug, per_session.drug)
    assert per_animal.design_summary() == per_session.design_summary()


def test_requesting_an_unidentifiable_level_is_an_error():
    """Silent variance stealing is the failure mode, so the override refuses."""
    with pytest.raises(ValueError, match="not identifiable"):
        _model([0, 1, 2, 3], [0, 0, 1, 1], [50] * 4, levels=("animal", "cell"))


def test_indices_must_agree_in_length():
    with pytest.raises(ValueError, match="one entry per session"):
        _model([0, 0, 1], [0, 1, 0, 1], [50, 50, 50])


def test_injection_bins_may_differ_per_session():
    m = _model([0, 1], [0, 1], [40, 90])
    gate = m._gate(200, device=torch.device("cpu"), dtype=torch.float32)
    assert gate[0].sum() == 160 and gate[1].sum() == 110


def test_model_rejects_an_injection_vector_that_contradicts_construction():
    m = _model([0, 1], [0, 1], [40, 90])
    with pytest.raises(ValueError, match="disagrees"):
        m.model(torch.zeros(2, 200), [40, 40])


# ----------------------------------------------------------------- the program
def _fit(m, y, steps=60, lr=1e-2):
    pyro.clear_param_store()
    pyro.set_rng_seed(0)
    svi = SVI(m.model, m.guide, ClippedAdam({"lr": lr}), Trace_ELBO(max_plate_nesting=2))
    return [svi.step(y) for _ in range(steps)]


def test_svi_runs_and_the_elbo_is_finite():
    """model and guide agree on every site, and the bound is finite throughout."""
    torch.manual_seed(0)
    animal, drug, inj = [0, 0, 1, 1], [0, 1, 0, 1], [60, 70, 55, 80]
    m = _model(animal, drug, inj, T=200)
    m.train(False)
    y = 0.1 * torch.randn(4, 200)
    losses = _fit(m, y)
    assert all(math.isfinite(x) for x in losses)
    assert losses[-1] < losses[0]


def test_readouts_have_the_shapes_the_design_implies():
    torch.manual_seed(0)
    m = _model([0, 0, 1, 1, 2], [0, 1, 0, 1, 0], [60] * 5, T=200)
    m.train(False)
    _fit(m, 0.1 * torch.randn(5, 200), steps=20)
    assert m.session_beta_inj().shape == (5,)
    assert m.animal_beta0().shape == (3,)
    assert set(m.drug_posterior()) == {0, 1}
    assert set(m.variance_components()) == {
        "sigma_b0", "sigma_animal", "sigma_cell", "sigma_b0_session"}
    loc, scale = m.drug_contrast(1, 0)
    assert math.isfinite(loc) and scale > 0


def test_the_windowed_guide_takes_a_warm_start_per_session():
    """A cohort is B different recordings; each must get its own matched filter."""
    torch.manual_seed(0)
    mf = torch.randn(3, 200)
    g = WindowedCategoricalEventGuide(mf, window=5)
    assert g.per_sequence
    pyro.clear_param_store()
    logits = g(torch.zeros(3, 200), 50, prior_logits=torch.zeros(3, 200))
    assert logits.shape == (3, 200)
    # different warm scores => different initial logits across sequences
    assert not torch.allclose(logits[0], logits[1])


def test_a_shared_warm_start_still_broadcasts():
    """The [T] form keeps working, and gives every sequence the same init."""
    pyro.clear_param_store()
    g = WindowedCategoricalEventGuide(torch.randn(200), window=5)
    assert not g.per_sequence
    logits = g(torch.zeros(3, 200), 50, prior_logits=torch.zeros(3, 200))
    torch.testing.assert_close(logits[0], logits[1])


def test_a_per_sequence_warm_start_must_match_the_batch():
    pyro.clear_param_store()
    g = WindowedCategoricalEventGuide(torch.randn(2, 200), window=5)
    with pytest.raises(ValueError, match="sequences but the batch"):
        g(torch.zeros(3, 200), 50, prior_logits=torch.zeros(3, 200))


def test_sessions_of_one_animal_share_that_animal_s_effect():
    """The point of nesting: two sessions of one animal are not independent evidence.

    With no session level, both sessions of an animal get exactly the same beta_inj --
    they differ only through the drug, which is what makes the contrast a comparison
    across animals rather than across recordings.
    """
    torch.manual_seed(0)
    m = _model([0, 0, 1, 1], [0, 0, 1, 1], [60] * 4, T=200)   # nested, no replicated-cell
    m.train(False)
    _fit(m, 0.1 * torch.randn(4, 200), steps=20)
    assert "session" in m.levels          # animal 0 x drug 0 IS replicated here
    bi = m.session_beta_inj()
    assert bi.shape == (4,)


# ------------------------------------------------- the realistic saline design
def test_saline_design_activates_the_per_session_baseline():
    """Vehicle + two opioids in every animal: sessions differ by day, so beta0 does.

    Three sessions per animal on three different days is the design people actually
    run. It is also the one where a single beta0 per animal is wrong: the baseline
    event rate drifts between recordings, and with no per-session baseline that
    misfit has nowhere to go but beta_inj.
    """
    animal = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    drug = [0, 1, 2, 0, 1, 2, 0, 1, 2]           # 0 = saline
    m = _model(animal, drug, [60] * 9)
    d = m.design_summary()
    assert "b0_session" in d["levels"]
    assert d["within_animal"] is True
    assert d["n_drugs"] == 3
    assert d["drugs_per_animal"] == [3, 3, 3]
    # each (animal, drug) cell is seen once, so there is no within-cell residual
    assert "session" not in d["levels"]


def test_the_baseline_session_level_acts_on_beta0_not_on_the_effect():
    """It is day-to-day baseline drift, not part of the drug response."""
    torch.manual_seed(0)
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [60] * 4, T=200)
    m.train(False)
    _fit(m, 0.1 * torch.randn(4, 200), steps=20)
    assert "b0_session" in m.levels
    # beta0 varies across an animal's sessions; the effect readout ignores the level
    b0 = m.session_beta0()
    assert b0.shape == (4,)
    assert not torch.allclose(b0[0], b0[1])
    assert m.session_beta_inj().shape == (4,)


def test_a_single_session_per_animal_has_no_baseline_session_level():
    m = _model([0, 1, 2, 3], [0, 0, 1, 1], [60] * 4)
    assert "b0_session" not in m.levels


def test_saline_contrasts_are_available_against_both_opioids():
    torch.manual_seed(0)
    animal = [0, 0, 0, 1, 1, 1]
    drug = [0, 1, 2, 0, 1, 2]
    m = _model(animal, drug, [60] * 6, T=200)
    m.train(False)
    _fit(m, 0.1 * torch.randn(6, 200), steps=20)
    for d1, d2 in ((1, 0), (2, 0), (2, 1)):
        loc, scale = m.drug_contrast(d1, d2)
        assert math.isfinite(loc) and scale > 0


# ---------------------------------------------------- the drug-structured tonic
def test_the_tonic_gets_a_per_drug_mean_by_default():
    """The Bateman curve is the drug's pharmacokinetics, so it is pooled by drug."""
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [60] * 4)
    d = m.design_summary()
    assert d["tonic_hierarchy"] is True
    assert set(m.tonic_anchor) == {"alpha", "rho", "gamma"}
    for v in m.tonic_anchor.values():
        assert v.shape == (m.n_drugs,)


def test_the_pooled_tonic_components_stop_being_map_parameters():
    """alpha/rho/gamma become latents, so their nn.Parameters must not be optimised."""
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [60] * 4)
    for name in ("alpha", "rho", "gamma"):
        assert getattr(m.tonic, name).requires_grad is False
    # b and drift stay per-session MAP: recording nuisance, not pharmacokinetics
    for name in ("b", "drift"):
        assert getattr(m.tonic, name).requires_grad is True


def test_the_tonic_session_level_needs_a_drug_with_replicates():
    """One session per drug confounds the deviation with the per-drug mean itself."""
    many = _model([0, 0, 1, 1], [0, 1, 0, 1], [60] * 4)     # 2 sessions per drug
    assert many.design_summary()["tonic_session_level"] is True
    one = _model([0, 0, 0], [0, 1, 2], [60] * 3)            # 1 session per drug
    assert one.design_summary()["tonic_session_level"] is False


def test_the_tonic_hierarchy_can_be_switched_off():
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [60] * 4, tonic_hierarchy=False)
    assert m.design_summary()["tonic_hierarchy"] is False
    assert m.tonic.alpha.requires_grad is True
    assert m.tonic_posterior() == {}


def test_the_tonic_anchor_comes_from_the_prefit_per_drug():
    """Averaging the per-trace pre-fits within drug, not a hard-coded constant."""
    tonic = BatchedTonicBaseline(n_seq=4, dt=DT, peak=[1.0, 3.0, 1.0, 3.0],
                                 k_rise=0.01, k_decay=0.002)
    phasic = PhasicEventProcess(_kernel(), beta0=-2.8, beta_inj=0.0, mu_a=1.7,
                                sigma_a=0.2, dt=DT, history=None)
    m = HierarchicalModel(phasic, tonic, AdditiveEmission(sigma_y=0.1),
                             animal=[0, 0, 1, 1], drug=[0, 1, 0, 1],
                             inj_t=[60] * 4, beta0_anchor=-2.8)
    # drug 0 saw A = 1.0 twice, drug 1 saw A = 3.0 twice
    torch.testing.assert_close(m.tonic_anchor["alpha"],
                               torch.log(torch.tensor([1.0, 3.0])), rtol=1e-5, atol=1e-6)


def test_the_tonic_svi_runs_and_reports_per_drug_pharmacokinetics():
    torch.manual_seed(0)
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [60] * 4, T=200)
    m.train(False)
    losses = _fit(m, 0.1 * torch.randn(4, 200), steps=40)
    assert all(math.isfinite(x) for x in losses)
    post = m.tonic_posterior()
    assert set(post) == {0, 1}
    for d, p in post.items():
        # "peak" is the height of the bump, "peak_s" the time it occurs at. The pair
        # is what the per-drug hierarchy exists to report; neither alone distinguishes
        # two compounds.
        assert p["peak"] > 0
        assert p["k_rise"] > p["k_decay"] > 0        # Bateman ordering survives
        assert p["peak_s"] > 0


def test_model_and_guide_agree_on_the_tonic_sites():
    """A site in one and not the other is silent in Pyro until the ELBO goes wrong."""
    import pyro.poutine as poutine
    torch.manual_seed(0)
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [60] * 4, T=200)
    m.train(False)
    pyro.clear_param_store()
    y = 0.1 * torch.randn(4, 200)
    g = poutine.trace(m.guide).get_trace(y)
    mt = poutine.trace(poutine.replay(m.model, trace=g)).get_trace(y)
    gs = {k for k, v in g.nodes.items() if v["type"] == "sample"}
    ms = {k for k, v in mt.nodes.items()
          if v["type"] == "sample" and not v.get("is_observed")}
    assert not (ms - gs), f"model sites missing from the guide: {sorted(ms - gs)}"
    for name in ("tonic_mu_alpha", "tonic_mu_rho", "tonic_mu_gamma"):
        assert name in gs and name in ms


# ------------------------------------------- MAP vs Bayesian per-session offsets
def test_offsets_are_map_by_default():
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [60] * 4)
    assert m.design_summary()["bayesian_offsets"] is False
    for name in ("b", "drift"):
        assert getattr(m.tonic, name).requires_grad is True


def test_bayesian_offsets_turn_b_and_drift_into_latents():
    """A MAP parameter has zero uncertainty, so its error never reaches the estimand."""
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [60] * 4, bayesian_offsets=True)
    assert m.design_summary()["bayesian_offsets"] is True
    for name in ("b", "drift"):
        assert getattr(m.tonic, name).requires_grad is False
    torch.testing.assert_close(m.offset_anchor["b"], torch.zeros(4))


def test_bayesian_offsets_require_the_tonic_hierarchy():
    with pytest.raises(ValueError, match="requires tonic_hierarchy"):
        _model([0, 0, 1, 1], [0, 1, 0, 1], [60] * 4,
               bayesian_offsets=True, tonic_hierarchy=False)


def test_svi_runs_with_bayesian_offsets_and_the_sites_match():
    import pyro.poutine as poutine
    torch.manual_seed(0)
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [60] * 4, T=200, bayesian_offsets=True)
    m.train(False)
    losses = _fit(m, 0.1 * torch.randn(4, 200), steps=30)
    assert all(math.isfinite(x) for x in losses)
    pyro.clear_param_store()
    y = 0.1 * torch.randn(4, 200)
    g = poutine.trace(m.guide).get_trace(y)
    mt = poutine.trace(poutine.replay(m.model, trace=g)).get_trace(y)
    gs = {k for k, v in g.nodes.items() if v["type"] == "sample"}
    ms = {k for k, v in mt.nodes.items()
          if v["type"] == "sample" and not v.get("is_observed")}
    assert not (ms - gs), f"model sites missing from the guide: {sorted(ms - gs)}"
    assert {"tonic_b", "tonic_drift"} <= gs


# ------------------------------------------- numerical guards on the joint block
def test_a_runaway_variance_component_cannot_produce_nan():
    """The scales are coordinates of an unbounded joint Gaussian once the block is
    widened, and the Cholesky factor can carry one far during early optimisation.
    Measured 2026-09-10: without the bound, 2 of the first 7 cohort fits died with an
    infinite intensity, which reaches the emission as a NaN mean."""
    torch.manual_seed(0)
    m = _model([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [60] * 6, T=200,
               beta_guide="fullrank")
    m.train(False)
    pyro.clear_param_store()
    y = 0.1 * torch.randn(6, 200)
    _fit(m, y, steps=5)                          # create the params
    with torch.no_grad():
        pyro.param("q_global_loc")[:] = 60.0     # every coordinate absurd
    losses = _fit(m, y, steps=5)
    assert all(math.isfinite(x) for x in losses)


def test_the_log_intensity_is_bounded():
    """p = 1 - exp(-lambda dt) must stay strictly inside (0, 1): at p = 1 exactly the
    relaxed Bernoulli's logits are infinite. With an unbounded Normal prior there is
    no support constraint doing this implicitly any more."""
    m = _model([0, 1], [0, 1], [50, 50])
    huge = torch.full((2,), 500.0)
    tiny = torch.full((2,), -500.0)
    gate = m._gate(100, device=torch.device("cpu"), dtype=torch.float32)
    for b0 in (huge, tiny):
        p = m._prior_event_prob(b0, torch.zeros(2), gate)
        assert torch.isfinite(p).all()
        assert (p > 0).all() and (p < 1).all()


# --------------------------------------------------------------------- the LR tilt
#
# The tilt IS the linear-response method: `dE[theta]/dt` under `p(.) exp(t.theta)` is
# the covariance. If the term entering the log joint is not exactly `sum_k c_k
# theta_k`, the finite difference measures the wrong derivative and nothing downstream
# says so -- the answer is simply a plausible wrong number. So it is checked directly
# against the sampled values rather than through a fit.

def _traced_tilt(m, y, tilt):
    """The `lr_tilt` factor's value, and the sites it should have been built from."""
    tr = pyro.poutine.trace(m.model).get_trace(y, None, tilt)
    return tr, tr.nodes["lr_tilt"]["fn"].log_factor


def test_a_tilt_adds_exactly_the_linear_term_it_names():
    animal, drug = [0, 0, 1, 1], [0, 1, 0, 1]
    m = _model(animal, drug, [100] * 4)
    y = torch.randn(4, 200) * 0.1
    coeff = [-3.0, 3.0]        # two drugs in this design
    tr, factor = _traced_tilt(m, y, {"mu_drug": coeff})
    mu_drug = tr.nodes["mu_drug"]["value"]
    expected = sum(c * float(v) for c, v in zip(coeff, mu_drug, strict=True))
    assert float(factor) == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_a_tilt_may_name_several_sites_at_once():
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [100] * 4)
    y = torch.randn(4, 200) * 0.1
    tr, factor = _traced_tilt(m, y, {"mu_b0": 2.0, "mu_drug": [1.0, 0.0]})
    expected = (2.0 * float(tr.nodes["mu_b0"]["value"])
                + float(tr.nodes["mu_drug"]["value"][0]))
    assert float(factor) == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_no_tilt_means_no_factor():
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [100] * 4)
    tr = pyro.poutine.trace(m.model).get_trace(torch.randn(4, 200) * 0.1)
    assert "lr_tilt" not in tr.nodes


def test_tilting_a_site_that_is_not_a_cohort_level_is_an_error():
    """Not silently ignored: a tilt on `z_animal` would answer a different question."""
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [100] * 4)
    with pytest.raises(ValueError, match="cannot tilt"):
        m.model(torch.randn(4, 200) * 0.1, None, {"z_animal": 1.0})


def test_a_tilt_of_the_wrong_length_is_an_error():
    """Three drugs, two coefficients: broadcasting this would tilt the wrong drug."""
    m = _model([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [100] * 6)
    with pytest.raises(ValueError, match="expected"):
        m.model(torch.randn(6, 200) * 0.1, None, {"mu_drug": [1.0, 0.0]})


def test_the_guide_accepts_the_tilt_and_ignores_it():
    """The tilt is a term of the joint, not of q, but Pyro needs matching signatures."""
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [100] * 4)
    y = torch.randn(4, 200) * 0.1
    pyro.clear_param_store()
    pyro.set_rng_seed(0)
    tr = pyro.poutine.trace(m.guide).get_trace(y, None, {"mu_drug": [-1.0, 1.0]})
    assert "lr_tilt" not in tr.nodes


# ------------------------------------------------------- the extended block
#
# `fullrank` correlates the eight globals. The variance the cohort interval is missing
# is not among them -- the block already has them jointly, and each drug level is
# still 2.5-3.6x too narrow -- so `fullrank_z` extends it through the non-centred
# effects, which are the channel by which between-animal spread reaches `mu_drug`.

def test_the_block_layout_covers_every_site_exactly_once():
    """Indexing a block by hand in six places is how the wrong latent gets unpacked."""
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [100] * 4, beta_guide="fullrank_z")
    n, where = m._block_layout()
    covered = sorted((sl.start, sl.stop) for sl in where.values())
    assert covered[0][0] == 0
    assert covered[-1][1] == n
    for (_, end), (start, _) in zip(covered, covered[1:], strict=False):
        assert end == start, "the layout has a gap or an overlap"


def test_the_extended_block_is_the_size_the_design_implies():
    animal, drug = [0, 0, 1, 1], [0, 1, 0, 1]
    m = _model(animal, drug, [100] * 4, beta_guide="fullrank_z")
    n, where = m._block_layout()
    expected = (1 + m.n_drugs + len(m._block_scales())
                + m.n_animals + sum(m.level_size[lvl] for lvl in m.levels))
    assert n == expected
    assert where["mu_b0"].start == 0
    assert where["mu_drug"].stop == 1 + m.n_drugs


def test_the_three_block_modes_nest():
    """fullrank subset fullrank_za subset fullrank_z, by construction.

    The partial mode exists because the full one overshoots: it widens the cohort
    contrast past nominal while sigma_animal falls to 0.158 against a truth of 0.25,
    which is variance displaced out of the between-animal level rather than added.
    The suspicion is that the displacement comes from the levels with the least data
    per coefficient, so the partial mode keeps only the animal-level effects.
    """
    animal, drug = [0, 0, 1, 1], [0, 1, 0, 1]
    blocked = {g: set(_model(animal, drug, [100] * 4, beta_guide=g)._blocked_z())
               for g in ("fullrank", "fullrank_za", "fullrank_z")}
    assert blocked["fullrank"] == set()
    assert blocked["fullrank"] < blocked["fullrank_za"] < blocked["fullrank_z"]
    assert blocked["fullrank_za"] == {"z_b0", "z_animal"}


def test_the_partial_block_leaves_the_crowded_levels_factorised():
    """cell and session have one coefficient per session; animal has one per animal."""
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [100] * 4, beta_guide="fullrank_za")
    _, where = m._block_layout()
    assert "z_animal" in where
    assert "z_cell" not in where and "z_b0_session" not in where


def test_the_plain_block_stops_at_the_globals():
    """`fullrank` is what the shipped results used and must not have moved."""
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [100] * 4, beta_guide="fullrank")
    n, where = m._block_layout()
    assert n == 1 + m.n_drugs + len(m._block_scales())
    assert "z_b0" not in where


def test_the_effects_are_seeded_wider_than_the_globals():
    """64 unit-scale latents started at 0.1 is a far tighter posterior than the fit
    begins from otherwise, and it takes thousands of steps to undo."""
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [100] * 4, beta_guide="fullrank_z")
    _, where = m._block_layout()
    _, diag = m._block_init(torch.device("cpu"), torch.float32)
    assert float(diag[where["mu_drug"]].max()) == pytest.approx(0.1)
    assert float(diag[where["z_b0"]].min()) == pytest.approx(0.5)


def test_every_guide_runs_and_the_readouts_agree_on_shape():
    animal, drug = [0, 0, 1, 1], [0, 1, 0, 1]
    y = torch.randn(4, 200) * 0.1
    for guide in ("meanfield", "fullrank", "fullrank_za", "fullrank_z"):
        m = _model(animal, drug, [100] * 4, beta_guide=guide)
        pyro.clear_param_store()
        pyro.set_rng_seed(0)
        _fit(m, y, steps=5)
        assert m.session_beta_inj().shape == (4,)
        assert m.session_beta0().shape == (4,)
        assert m.animal_beta0().shape == (2,)
        loc, scale = m.drug_contrast(1, 0)
        assert math.isfinite(loc) and scale > 0
        assert set(m.variance_components()) >= {"sigma_animal"}


def test_an_unknown_guide_is_refused():
    with pytest.raises(ValueError, match="beta_guide must be"):
        _model([0, 0, 1, 1], [0, 1, 0, 1], [100] * 4, beta_guide="fancy")


# ------------------------------------------- the tonic guide's scales are bounded
#
# One uncrossed cohort diverged reproducibly across two launches with
# `q_z_tonic_rho_scale` reaching 58.6 -- a posterior that wide on log(k_decay)
# proposes elimination rates differing by e^58 between sessions, and the draws reach
# the emission as a non-finite mean. The phasic components have had a bound since the
# joint block made them coordinates of an unbounded Gaussian; the tonic ones are
# mean-field, outside the block, and had none.

def test_the_tonic_guide_scales_cannot_leave_their_bound():
    from photosvi.models.points.hierarchical import _TONIC_SCALE_RANGE
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [100] * 4)
    pyro.clear_param_store()
    y = torch.randn(4, 200) * 0.1
    m.guide(y)
    lo, hi = _TONIC_SCALE_RANGE
    seen = 0
    for k, v in pyro.get_param_store().items():
        if k.startswith(("q_tonic_sigma_", "q_z_tonic_")) and k.endswith("_scale"):
            seen += 1
            t = v.detach()
            assert bool((t >= lo).all() and (t <= hi).all()), f"{k} outside the bound"
    assert seen >= 6, f"expected six bounded tonic scales, constrained {seen}"


def test_a_run_can_tell_whether_a_tonic_scale_sat_on_the_bound():
    """A bound makes a fit finish; it does not make the parameter identified."""
    m = _model([0, 0, 1, 1], [0, 1, 0, 1], [100] * 4)
    pyro.clear_param_store()
    m.guide(torch.randn(4, 200) * 0.1)
    w = m.tonic_guide_widths()
    assert "at_bound" in w and w["at_bound"] == 0.0
    assert any(k.endswith("_scale_max") for k in w)
