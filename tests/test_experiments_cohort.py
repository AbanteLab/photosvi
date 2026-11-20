"""The cohort simulator, and the property the comparison study depends on.

The load-bearing test here is `test_the_two_arms_see_identical_recordings`. Both arms
regenerate the cohort from its seed rather than sharing a cache, so if their spec
construction ever diverges they fit *different data* while every other test still
passes, and the comparison becomes meaningless in a way no downstream check could
detect. That is why the spec has one definition and why this test exercises it.
"""

from __future__ import annotations

import math
import pathlib

import pytest
import torch
import yaml

from photosvi.experiments.cohort import (
    OPIOID1,
    OPIOID2,
    SALINE,
    CohortSpec,
    build_cohort_model,
    simulate_cohort,
    spec_from_config,
)

CFG = "configs/cohort_comparison.yaml"


def _small(**kw) -> CohortSpec:
    """A cohort small enough to simulate in a test: the sampler walks the time axis."""
    base = dict(n_animals=2, T=400, inj_t=150, inj_jitter=20)
    return CohortSpec(**(base | kw))


def test_simulation_is_deterministic_in_its_seed():
    a = simulate_cohort(_small(), 7)
    b = simulate_cohort(_small(), 7)
    torch.testing.assert_close(a["y"], b["y"])
    assert a["beta_inj_true"] == b["beta_inj_true"]
    assert a["inj_t"] == b["inj_t"]


def test_different_seeds_give_different_cohorts():
    a = simulate_cohort(_small(), 0)
    b = simulate_cohort(_small(), 1)
    assert not torch.allclose(a["y"], b["y"])


def test_the_two_arms_see_identical_recordings():
    """Both workers build their spec through one function, so they cannot diverge."""
    cfg = yaml.safe_load(open(CFG))
    cfg["cohort"] |= dict(n_animals=2, T=400, inj_t=150, inj_jitter=20)
    s1 = spec_from_config(cfg, 0.40)
    s2 = spec_from_config(cfg, 0.40)
    assert s1 == s2
    torch.testing.assert_close(simulate_cohort(s1, 3)["y"], simulate_cohort(s2, 3)["y"])


def test_delta_moves_only_the_second_opioid():
    cfg = yaml.safe_load(open(CFG))
    base, shifted = spec_from_config(cfg, 0.0), spec_from_config(cfg, 0.7)
    assert base.mu_drug[SALINE] == shifted.mu_drug[SALINE]
    assert base.mu_drug[OPIOID1] == shifted.mu_drug[OPIOID1]
    assert shifted.mu_drug[OPIOID2] - base.mu_drug[OPIOID2] == pytest.approx(0.7)
    assert base.true_contrast() == pytest.approx(0.0)


def test_saline_is_not_a_null_condition():
    """Handling and the injection raise the rate on their own; vehicle controls for it."""
    cfg = yaml.safe_load(open(CFG))
    assert spec_from_config(cfg, 0.0).mu_drug[SALINE] > 0.0


def test_every_animal_receives_every_drug():
    spec = _small(n_animals=3)
    animal, drug = spec.indices()
    assert len(animal) == len(drug) == spec.n_sessions == 9
    for a in range(3):
        assert sorted(d for i, d in zip(animal, drug, strict=True) if i == a) == [0, 1, 2]


def test_injection_bins_are_jittered_per_session():
    """A fixed injection bin would let a method key off the bin instead of the trace."""
    coh = simulate_cohort(_small(n_animals=3), 0)
    assert len(set(coh["inj_t"])) > 1
    spec = _small(n_animals=3)
    for t in coh["inj_t"]:
        assert abs(t - spec.inj_t) <= spec.inj_jitter


def test_sessions_of_one_animal_share_its_responsiveness():
    """The animal effect is drawn once per animal, not once per session.

    With no interaction and no session residual, an animal's three beta_inj differ
    only by the drug means -- which is exactly what makes sessions of one animal
    non-independent evidence about a drug.
    """
    spec = _small(n_animals=2, sigma_cell=0.0, sigma_b0_session=0.0)
    coh = simulate_cohort(spec, 0)
    animal, drug = spec.indices()
    for a in (0, 1):
        rows = [(d, b) for i, d, b in
                zip(animal, drug, coh["beta_inj_true"], strict=True) if i == a]
        offsets = [b - spec.mu_drug[d] for d, b in rows]
        assert max(offsets) - min(offsets) < 1e-9


def test_saline_sessions_get_a_smaller_tonic_carrier():
    """Vehicle produces a small carrier, not none: a flat one would be a giveaway."""
    spec = _small()
    assert 0 < spec.tonic_peak[SALINE] < min(spec.tonic_peak[OPIOID1], spec.tonic_peak[OPIOID2])


def test_the_two_opioids_have_different_pharmacokinetics():
    """Without this there is nothing for a drug-structured tonic to find.

    The tonic carrier IS the drug's absorption and elimination. An earlier version of
    the simulator gave both opioids one curve, which would have made the whole
    question untestable while producing a clean-looking null.
    """
    spec = _small()
    assert spec.tonic_k_rise[OPIOID1] != spec.tonic_k_rise[OPIOID2]
    assert spec.tonic_k_decay[OPIOID1] != spec.tonic_k_decay[OPIOID2]
    # and the difference is large enough to be visible: peak times differ >2x
    p1, p2 = spec.tonic_peak_s(OPIOID1), spec.tonic_peak_s(OPIOID2)
    assert max(p1, p2) / min(p1, p2) > 2.0


def test_every_drug_keeps_the_bateman_ordering():
    """k_rise > k_decay > 0 for every drug, or TonicBaseline refuses to build."""
    spec = _small()
    for d in range(spec.n_drugs):
        assert spec.tonic_k_rise[d] > spec.tonic_k_decay[d] > 0


def test_the_config_carries_one_curve_per_drug():
    cfg = yaml.safe_load(open(CFG))
    spec = spec_from_config(cfg, 0.0)
    for f in ("tonic_peak", "tonic_k_rise", "tonic_k_decay"):
        assert len(getattr(spec, f)) == spec.n_drugs


def test_the_count_mle_is_reported_per_session():
    """The best any method could do on that recording, and the right per-session
    reference for a fitted estimate."""
    coh = simulate_cohort(_small(), 0)
    assert len(coh["mle"]) == 6
    for m, pre, post in zip(coh["mle"], coh["n_pre"], coh["n_post"], strict=True):
        if pre and post:
            assert math.isfinite(m)


# ------------------------------------------------------------- the shared builder
#
# Every arm of the cohort study has to fit the SAME model; the study's whole claim is
# a comparison between analyses of identical data. An arm that quietly built a
# different warm start or a different prior scale would be measuring something else,
# and nothing in the outputs would say so -- hence one builder, tested here.

def _small_cfg():
    """The shipped config, shrunk to a cohort a test can afford to simulate."""
    cfg = yaml.safe_load(open(CFG))
    cfg["cohort"] |= dict(n_animals=2, T=400, inj_t=150, inj_jitter=20)
    return cfg


def test_the_builder_returns_a_model_matching_the_simulated_design():
    cfg = _small_cfg()
    spec = spec_from_config(cfg, 0.0)
    coh = simulate_cohort(spec, 0)
    model, y, mf = build_cohort_model(cfg, spec, coh, guide="fullrank")
    assert y.shape == (spec.n_sessions, spec.T)
    assert mf.shape == (spec.n_sessions, spec.T)
    assert model.n_sessions == spec.n_sessions
    assert model.n_drugs == spec.n_drugs
    assert model.beta_guide == "fullrank"
    assert not model.training


def test_the_builder_takes_its_settings_from_the_config_not_from_defaults():
    """A silently-defaulted prior scale would change every interval the study reports."""
    cfg = _small_cfg()
    spec = spec_from_config(cfg, 0.0)
    coh = simulate_cohort(spec, 0)
    model, _, _ = build_cohort_model(cfg, spec, coh)
    assert model.tau == cfg["fit"]["tau"]
    assert model.mu_b0_prior_scale == cfg["fit"]["mu_b0_prior_scale"]
    assert model.mu_drug_prior_scale == cfg["fit"]["mu_drug_prior_scale"]
    assert model.beta_guide == cfg["fit"]["hier_beta_guide"]


def test_the_builder_has_no_guide_default_of_its_own():
    """The omitted argument must reach the config, not a literal in the signature.

    This is the shape of the bug that ran the whole two-stage arm: a constructor
    default of `meanfield` meant a caller who forgot the keyword got the study's
    worst-calibrated family, and nothing anywhere said so.
    """
    cfg = _small_cfg()
    cfg["fit"]["hier_beta_guide"] = "fullrank_za"
    spec = spec_from_config(cfg, 0.0)
    model, _, _ = build_cohort_model(cfg, spec, simulate_cohort(spec, 0))
    assert model.beta_guide == "fullrank_za"


def test_the_shipped_cohort_config_names_a_guide_for_every_arm():
    """Neither arm may fall through to a constructor default.

    `SingleTraceModel` and `HierarchicalModel` both default to mean-field, and both
    accept a different set of families, so the config carries one key per arm. A
    config that drops either key has to fail the suite rather than the study.
    """
    cfg = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parents[1]
         / "configs" / "cohort_comparison.yaml").read_text())
    assert cfg["fit"]["beta_guide"] in ("meanfield", "fullrank")
    assert cfg["fit"]["hier_beta_guide"] in (
        "meanfield", "fullrank", "fullrank_za", "fullrank_z")


def test_the_builder_is_deterministic_given_the_seed():
    cfg = _small_cfg()
    spec = spec_from_config(cfg, 0.0)
    a = build_cohort_model(cfg, spec, simulate_cohort(spec, 3))[2]
    b = build_cohort_model(cfg, spec, simulate_cohort(spec, 3))[2]
    assert torch.equal(a, b)


# ------------------------------------------------------------------ the designs
#
# The two designs are the extremes an experimenter chooses between, and they differ
# in what the opioid-versus-opioid contrast IS: within-animal under crossing, between
# -animal without it. The tests below pin the properties the paper's design section
# rests on, because a simulator that quietly generated the wrong assignment would
# produce a comparison that looks fine and measures nothing.

def test_crossed_is_the_default_and_unchanged():
    """A config predating the `design` field must produce exactly its old cohorts."""
    cfg = _small_cfg()
    cfg["cohort"].pop("design", None)
    spec = spec_from_config(cfg, 0.0)
    assert spec.design == "crossed"
    animal, drug = spec.indices()
    assert spec.n_sessions == spec.n_animals * spec.n_drugs == len(animal)
    for a in range(spec.n_animals):
        assert sorted(d for i, d in zip(animal, drug, strict=True) if i == a) == [0, 1, 2]


def test_uncrossed_gives_every_animal_vehicle_and_one_opioid():
    spec = CohortSpec(design="uncrossed", n_animals=6, T=400, inj_t=150, inj_jitter=20)
    animal, drug = spec.indices()
    assert spec.n_sessions == 12 == len(animal)
    for a in range(6):
        got = sorted(d for i, d in zip(animal, drug, strict=True) if i == a)
        assert got[0] == SALINE
        assert got[1] in (OPIOID1, OPIOID2)
        assert len(got) == 2
    # and the two opioid groups are the same size, so neither is confounded with
    # position in the cohort
    assert drug.count(OPIOID1) == drug.count(OPIOID2) == 3


def test_the_designs_can_be_compared_at_equal_recordings():
    """8 x 3 and 12 x 2 are both 24, which is what makes the comparison a design
    comparison rather than a sample-size one."""
    crossed = CohortSpec(n_animals=8, T=400, inj_t=150, inj_jitter=20)
    uncrossed = CohortSpec(design="uncrossed", n_animals=12, T=400, inj_t=150,
                           inj_jitter=20)
    assert crossed.n_sessions == uncrossed.n_sessions == 24


def test_only_the_crossed_design_pairs_the_two_opioids_within_an_animal():
    """The fact the whole design section turns on: under crossing the animal effect
    enters both terms of the opioid contrast and cancels; uncrossed it cannot."""
    for design, n, paired in (("crossed", 8, True), ("uncrossed", 12, False)):
        spec = CohortSpec(design=design, n_animals=n, T=400, inj_t=150, inj_jitter=20)
        animal, drug = spec.indices()
        saw_both = {a for a in range(n)
                    if {OPIOID1, OPIOID2} <= {d for i, d in
                                              zip(animal, drug, strict=True) if i == a}}
        assert bool(saw_both) is paired


def test_an_unknown_design_is_refused():
    spec = CohortSpec(design="latin-square", n_animals=4)
    with pytest.raises(ValueError, match="design must be"):
        spec.indices()


def test_the_uncrossed_design_still_simulates_and_builds():
    cfg = _small_cfg()
    cfg["cohort"] |= dict(design="uncrossed", n_animals=4)
    spec = spec_from_config(cfg, 0.0)
    coh = simulate_cohort(spec, 0)
    model, y, _ = build_cohort_model(cfg, spec, coh)
    assert y.shape == (8, spec.T)
    assert model.n_sessions == 8
    # the interaction stays identifiable -- every animal saw two drugs, so cells
    # outnumber animals -- which is what the paper's comparison needs
    assert "cell" in model.design_summary()["levels"]


def test_the_ou_is_deprecated_and_the_ar1_replaces_it_exactly():
    """An OU sampled at a fixed step IS an AR(1), so keeping both kept one process twice.

    The equality is the point: `ou_sd = s` with `ou_tau = t`, and `sigma_y = s` with
    `ar_rho = exp(-dt/t)`, are the same process at the same amplitude. They differ only in
    WHERE it sits -- the OU on the mean, which the model has no term for, the AR(1) in the
    emission, which the model scores. That is why the AR(1) is the one kept.
    """
    import dataclasses
    import warnings

    import numpy as np

    cfg = yaml.safe_load(open("configs/cohort_comparison.yaml"))
    base = spec_from_config(cfg, 0.20)

    def run(**kw):
        spec = dataclasses.replace(base, **kw)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = simulate_cohort(spec, seed=0)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
        return np.asarray(out["y"]), dep

    y_default, dep_default = run()
    assert not dep_default, "the default must not warn -- ou_sd is zero"

    y_ou, dep_ou = run(ou_sd=3.0)
    assert dep_ou, "asking for the OU has to say it is deprecated"
    assert "ar_rho=0.94446" in str(dep_ou[0].message)   # exp(-0.04/0.7), spelled out
    assert "sigma_y=3" in str(dep_ou[0].message)

    y_ar, dep_ar = run(ar_rho=0.94446, sigma_y=3.0)
    assert not dep_ar, "the replacement must not warn"

    # the same process: same amplitude, and both far above the white default
    assert y_ar.std() == pytest.approx(y_ou.std(), rel=0.05)
    assert y_ou.std() > 1.5 * y_default.std()
