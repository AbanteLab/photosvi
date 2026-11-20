"""Tests for the guide-family sweep machinery (photosvi/experiments).

The properties worth pinning here are the ones that make the *comparison* valid rather
than the ones that make a fit good:

  * the arm registry spans (guide family x coupling) and builds what it claims to;
  * the held-out probe cannot move a global, and leaves the caller's param store intact;
  * the shipped config's arms all exist and its knobs parse;
  * a cell round-trips to JSON with its provenance.

Everything runs at toy scale -- these are contract tests, not experiments.
"""

from __future__ import annotations

import json

import numpy as np
import pyro
import torch

from photosvi.benchmark import matched_filter_score, nms
from photosvi.experiments import (
    ARMS,
    build_arm,
    cell_path,
    detection_row,
    ev_per_min,
    generate,
    heldout_probe,
    load_config,
    null_recon_ll,
    run_cell,
    write_cell,
)
from photosvi.experiments.arms import frozen_local_model
from photosvi.experiments.sweep import gen_spec, setup_from_config
from photosvi.models.points import (
    FreeEventGuide,
    MatchedFilterEventGuide,
    WarmFreeEventGuide,
    WindowedCategoricalEventGuide,
    bi_exponential_kernel,
    build_kernel,
    fit_tonic_baseline,
)

SMOKE_CFG = "configs/points_guide_sweep_smoke.yaml"


def _tiny_cfg():
    """A cell small enough to run in a test: 1200 bins, 20 steps, 5 held-out steps."""
    cfg = load_config(SMOKE_CFG)
    cfg["data"]["T"] = 1200
    cfg["data"]["inj_t"] = 600
    cfg["fit"]["steps"] = 20
    cfg["fit"]["probe"] = 10
    cfg["heldout"]["steps"] = 5
    cfg["heldout"]["every"] = 2
    cfg["kernel"]["length"] = 40
    return cfg


def _setup(cfg=None, beta0=-3.0):
    """The setup the cell itself would build -- derived from the config, not hand-rolled.

    Hand-rolling it silently misspecifies the model relative to the generated data
    (different marks, different noise scale), which makes any likelihood assertion
    measure the mismatch instead of the code.
    """
    return setup_from_config(cfg or _tiny_cfg(), beta0)


# ------------------------------------------------------------------------ arm registry


def test_registry_spans_the_two_axes():
    """Every arm named, and each axis actually varies -- otherwise a contrast is missing."""
    assert set(ARMS) == {"free", "free-warm", "amort", "free+c", "amort+c", "cat", "cat+c"}
    assert {a.guide for a in ARMS.values()} == {"free", "warm", "mf", "cat"}
    assert {a.couple for a in ARMS.values()} == {True, False}
    # the un-confounding arm: same family as `free`, same init as `amort`
    assert ARMS["free-warm"].guide == "warm" and ARMS["free-warm"].couple is False


def test_build_arm_installs_the_declared_block_and_flag():
    cfg = _tiny_cfg()
    y, _ = generate(cfg, 0)
    inj_t = cfg["data"]["inj_t"]
    k = bi_exponential_kernel(40, k_rise=2.0, k_decay=1.0, dt=1 / 25)
    mf = matched_filter_score(y, k, inj_t, dt=1 / 25)
    setup = _setup()

    expected = {
        "free": FreeEventGuide, "free-warm": WarmFreeEventGuide,
        "amort": MatchedFilterEventGuide, "free+c": FreeEventGuide,
        "amort+c": MatchedFilterEventGuide,
        "cat": WindowedCategoricalEventGuide, "cat+c": WindowedCategoricalEventGuide,
    }
    for name, cls in expected.items():
        pyro.clear_param_store()
        m = build_arm(name, y, inj_t, setup, seed=0, mf=mf)
        assert isinstance(m.event_guide, cls), name
        assert m.couple_events is ARMS[name].couple, name
        assert not m.emission.log_sigma_y.requires_grad          # sigma_y stays pinned


def test_build_arm_needs_mf_for_the_filter_families():
    cfg = _tiny_cfg()
    y, _ = generate(cfg, 0)
    for name in ("free-warm", "amort", "amort+c"):
        try:
            build_arm(name, y, cfg["data"]["inj_t"], _setup(), seed=0, mf=None)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {name} without mf")
    # the free families do not need it
    pyro.clear_param_store()
    build_arm("free", y, cfg["data"]["inj_t"], _setup(), seed=0, mf=None)


def test_unknown_arm_raises():
    try:
        build_arm("nope", torch.zeros(1, 100), 50, _setup(), seed=0)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for an unknown arm")


# -------------------------------------------------------------------------- held-out


def test_heldout_probe_cannot_move_a_global_and_restores_the_store():
    """The two guarantees the protocol rests on."""
    cfg = _tiny_cfg()
    y_ho, _ = generate(cfg, 7)
    inj_t = cfg["data"]["inj_t"]
    dt = 1 / 25
    k = bi_exponential_kernel(40, k_rise=2.0, k_decay=1.0, dt=dt)
    mf_ho = matched_filter_score(y_ho, k, inj_t, dt=dt)
    setup = _setup()
    tonic = fit_tonic_baseline(y_ho, inj_t, dt=dt)

    # a sentinel in the caller's store must survive the probe untouched
    pyro.clear_param_store()
    pyro.param("caller_sentinel", torch.tensor(1.234))
    before = dict(pyro.get_param_store().items())
    b0, bi = -3.0, 0.7
    out = heldout_probe(b0, bi, y_ho, inj_t, setup, tonic, mf_ho=mf_ho, steps=5, seed=0)

    assert set(pyro.get_param_store().keys()) == set(before)
    assert torch.allclose(pyro.param("caller_sentinel"), torch.tensor(1.234))
    assert np.isfinite(out["recon"]) and np.isfinite(out["elbo"])

    # the probe is a pure function of the betas it is handed: a different beta_inj
    # must change the score, and the same one must reproduce it
    same = heldout_probe(b0, bi, y_ho, inj_t, setup, tonic, mf_ho=mf_ho, steps=5, seed=0)
    diff = heldout_probe(b0, bi + 2.0, y_ho, inj_t, setup, tonic, mf_ho=mf_ho, steps=5, seed=0)
    assert out["elbo"] == same["elbo"]
    assert out["elbo"] != diff["elbo"]


def test_null_recon_is_exactly_the_zero_drive_score():
    """The baseline must be the tonic alone -- that is the whole point of subtracting it.

    Asserted against an explicit zero-drive computation rather than against "some fit
    beats it": whether a fit beats the null depends on its budget, which would make this
    a test of training length rather than of the baseline.
    """
    cfg = _tiny_cfg()
    y_ho, _ = generate(cfg, 7)
    inj_t, dt = cfg["data"]["inj_t"], 1 / 25
    setup = _setup(cfg)
    tonic = fit_tonic_baseline(y_ho, inj_t, dt=dt)

    null = null_recon_ll(y_ho, inj_t, setup, tonic)
    assert np.isfinite(null)

    m = frozen_local_model(0.0, inj_t, setup, tonic)
    mean = m.tonic.baseline(y_ho.shape[-1], inj_t).unsqueeze(0).expand_as(y_ho)
    expected = float(m.emission.log_prob(y_ho, mean).sum()) / y_ho.numel()
    assert np.isclose(null, expected)

    # and it is a pure function of the trace + tonic: no RNG, no fitting
    assert null == null_recon_ll(y_ho, inj_t, setup, tonic)


# ------------------------------------------------------------------------- scoring


def test_nms_enforces_the_refractory():
    times = np.array([10, 12, 14, 100, 103, 300])
    scores = np.array([1.0, 5.0, 2.0, 1.0, 9.0, 3.0])
    kept_t, kept_s = nms(times, scores, refractory=8)
    assert list(kept_t) == [12, 103, 300]                 # best of each cluster survives
    assert list(kept_s) == [5.0, 9.0, 3.0]
    assert list(kept_t) == sorted(kept_t)                 # returned in time order
    assert len(nms(np.array([]), np.array([]))[0]) == 0   # empty input is not a crash


def test_detection_row_scores_a_planted_signal():
    T = 600
    onsets = np.array([100, 250, 400])
    score = np.zeros(T)
    for o in onsets:
        score[o] = 0.9
    row = detection_row(score, onsets, tol=9, refractory=8, thresh=0.5)
    assert row["pr_auc"] > 0.9
    assert row["n_call"] == 3
    assert row["best_f1"] > 0.9


def test_ev_per_min_is_an_exponential_not_a_sigmoid():
    """beta is a log-rate under the cloglog link -- the readout must be exp, times 60."""
    assert np.isclose(ev_per_min(0.0), 60.0)
    assert np.isclose(ev_per_min(np.log(2)), 120.0)
    # a rate ratio is the exponential of the beta difference, whatever the baseline
    for b0 in (-4.0, -2.0, 0.5):
        assert np.isclose(ev_per_min(b0 + 1.1) / ev_per_min(b0), np.exp(1.1))


# ------------------------------------------------------------------- config & cells


ACTIVE_CFGS = (
    "configs/points_guide_sweep_nonneg.yaml",
    "configs/points_guide_sweep_k5.yaml",
    "configs/points_guide_sweep_k5_misspec.yaml",
)

# The configs that generate the paper's headline numbers, and which therefore have to
# carry the frozen production settings. The kernel-comparison arms above are
# deliberately excluded: they exist to isolate the phasic basis, and changing their
# prior or guide would change what that comparison measures relative to the runs
# already archived against it.
PRODUCTION_CFGS = (
    "configs/points_guide_sweep_k5.yaml",
    "configs/benchmark_guppy.yaml",
)


def test_shipped_configs_name_only_real_arms():
    for path in ACTIVE_CFGS + (SMOKE_CFG, "configs/points_guide_sweep.yaml"):
        cfg = load_config(path)
        assert cfg["grid"]["arms"], path
        for arm in cfg["grid"]["arms"]:
            assert arm in ARMS, f"{path} names unknown arm {arm!r}"


def test_shipped_configs_build_both_kernels():
    """Every config must name a buildable fit kernel, and gen must default to it."""
    for path in ACTIVE_CFGS + (SMOKE_CFG, "configs/points_guide_sweep.yaml"):
        cfg = load_config(path)
        k = cfg["kernel"]
        assert "fit" in k, f"{path}: no kernel.fit block"
        h_fit = build_kernel(dict(k["fit"]), k["length"], 1.0 / cfg["data"]["fs"])
        h_gen = build_kernel(gen_spec(k), k["length"], 1.0 / cfg["data"]["fs"])
        assert h_fit.shape == h_gen.shape == (k["length"],), path
        if "gen" not in k:
            assert torch.equal(h_fit, h_gen), f"{path}: gen must default to fit"


def test_only_misspec_configs_override_gen():
    """A gen/fit mismatch is a deliberate probe, never an accident.

    Matching them is the default because a generative model whose phasic carrier has a
    different mean structure from the inference model's has no representable truth, so
    beta recovery would score misspecification rather than the estimator.
    """
    for path in ACTIVE_CFGS:
        cfg = load_config(path)
        if "gen" in cfg["kernel"]:
            assert "misspec" in path, (
                f"{path} overrides kernel.gen but is not named a misspecification probe"
            )


def test_mean_free_configs_have_zero_area():
    for path in ("configs/points_guide_sweep_k5.yaml", "configs/points_guide_sweep_k5_misspec.yaml"):
        cfg = load_config(path)
        h = build_kernel(dict(cfg["kernel"]["fit"]), cfg["kernel"]["length"],
                         1.0 / cfg["data"]["fs"])
        assert abs(float(h.sum())) < 1e-4, f"{path}: fit kernel is not mean-free"
        assert float(h.min()) < 0 < float(h.max()), f"{path}: fit kernel is not biphasic"


def test_cell_path_is_filesystem_safe():
    """`+` in arm names must not leak into filenames."""
    p = cell_path("runs", "amort+c", 3)
    assert "+" not in p.name
    assert p.name == "amort_plus_c__seed3.json"


def test_run_cell_round_trips_to_json(tmp_path):
    cfg = _tiny_cfg()
    rec = run_cell(cfg, "free-warm", 0, verbose=False)

    assert rec["arm"] == "free-warm"
    assert len(rec["steps"]) == len(rec["b0"]) == len(rec["bi"]) == len(rec["pr_auc"])
    assert len(rec["ho_steps"]) == len(rec["ho_recon"]) == len(rec["ho_elbo"])
    assert rec["ho_steps"][-1] == rec["steps"][-1]        # the final fit is always probed
    assert rec["provenance"]["host"] and rec["provenance"]["torch"]
    assert rec["n_onsets_pre"] + rec["n_onsets_post"] == rec["n_onsets"]

    out = write_cell(rec, tmp_path)
    assert out.exists()
    assert not list(tmp_path.glob("*.tmp"))               # atomic write leaves no debris
    back = json.loads(out.read_text())
    assert back["arm"] == "free-warm"
    assert back["config"]["grid"]["arms"] == cfg["grid"]["arms"]


def test_coupled_cell_reads_q_through_the_anchor(tmp_path):
    """A coupled arm must not report the block's logits as q -- it would be over-confident."""
    cfg = _tiny_cfg()
    rec = run_cell(cfg, "amort+c", 0, verbose=False)
    q = np.asarray(rec["q_final"])
    assert q.shape == (cfg["data"]["T"],)
    assert (q > 0).all() and (q < 1).all()
    # with a sparse prior anchor folded in, almost every bin is far from certain
    assert q.mean() < 0.5


def test_uniform_bounds_are_derived_from_the_anchor():
    """Hard-coded bounds would eventually exclude a real animal's baseline.

    beta0 is a log-rate, so its plausible range moves with the trace; a fixed
    interval in a config puts some animal outside the Uniform's support, where the
    prior scores -inf and the fit fails without saying so. The bounds therefore
    track the matched-filter anchor. beta_inj is a log-ratio and stays on zero.
    """
    cfg = _tiny_cfg()
    cfg["fit"]["beta_prior"] = "uniform"
    for anchor in (-2.7, -5.0, -0.5):
        kw = setup_from_config(cfg, anchor).beta_prior_kwargs()
        assert kw["beta_prior"] == "uniform"
        assert kw["beta_low"] < anchor < kw["beta_high"]
        assert kw["beta_inj_low"] < 0.0 < kw["beta_inj_high"]
        # beta_inj's bounds do NOT move with the anchor -- it is a ratio
        assert kw["beta_inj_low"] == -4.0 and kw["beta_inj_high"] == 4.0


def test_normal_prior_path_carries_the_guide_choice():
    cfg = _tiny_cfg()
    cfg["fit"]["beta_prior"] = "normal"
    kw = setup_from_config(cfg, -3.0).beta_prior_kwargs()
    assert kw == {"beta_prior": "normal", "beta_guide": "fullrank",
                  "beta0_prior_scale": cfg["fit"]["beta0_prior_scale"],
                  "beta_inj_prior_scale": cfg["fit"]["beta_inj_prior_scale"]}


def test_shipped_configs_use_the_unbounded_prior_and_the_joint_block():
    """Frozen 2026-09-10, and both halves were measured before being chosen.

    The bounded Uniform was replaced because it forced the guide onto a transformed
    axis and the scale it read back was a delta-method linearisation: on the same
    recordings it reported 0.1525 where the unbounded fits reported 0.1711, a 12%
    artificial narrowing. Shrinkage from the Normal's widths is 0.77%, matching the
    analytic prediction. The joint block is what makes the interval honest at all --
    0.97x the exact posterior against NUTS, from 2.2x factorised.
    """
    for path in PRODUCTION_CFGS:
        fit = load_config(path)["fit"]
        assert fit["beta_prior"] == "normal", path
        assert fit["beta_guide"] == "fullrank", path
        assert fit["beta_inj_prior_scale"] >= 2.0, path
        assert fit["tau"] <= 0.5, path


def test_diagnostic_config_is_half_length_and_labelled_as_diagnostic():
    """The half-length config must not quietly become a results config.

    Halving the trace halves the events and takes the standard error of a count MLE
    for beta_inj from 0.167 to about 0.25. That is fine for watching an optimiser
    drift and not fine for a number the paper quotes, so the file says so and this
    pins the property the warning is about.
    """
    full = load_config("configs/points_guide_sweep_k5.yaml")
    half = load_config("configs/diagnostic_half.yaml")
    assert half["data"]["T"] * 2 == full["data"]["T"]
    # the pre/post split is preserved, not moved to the midpoint
    def inj_fraction(cfg):
        return cfg["data"]["inj_t"] / cfg["data"]["T"]

    assert abs(inj_fraction(half) - inj_fraction(full)) < 1e-9
    assert half["data"]["fs"] == full["data"]["fs"]        # sampling rate stays realistic
    assert "diagnostics only" in open("configs/diagnostic_half.yaml").read().lower()


def test_diagnostic_config_is_not_in_the_active_set():
    """It must never be swept as if it produced results."""
    assert "configs/diagnostic_half.yaml" not in ACTIVE_CFGS
