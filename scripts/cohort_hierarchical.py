"""One cohort, fit by the hierarchical model. GPU by default.

    python scripts/cohort_hierarchical.py <delta> <seed> [device]

Writes runs_cohort/hier_<delta>_<seed>.json with the drug posteriors, every pairwise
contrast, the variance components and the per-session estimates.

This arm exists to be compared against `cohort_twostage.py` on identical data: both
call `simulate_cohort(spec, seed)`, which is deterministic, so neither can be handed
a different cohort than the other.

Runs on GPU because the per-step cost is flat in batch size (measured: 55 ms/step at
B = 16 as at B = 1), so a 24-session cohort costs what one trace costs. On CPU the
same fit is roughly 24x a single-trace fit.
"""
from __future__ import annotations

import json
import math
import os
import sys

import pyro
import yaml
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from photosvi.experiments.cohort import (
    DRUG_NAMES,
    OPIOID1,
    OPIOID2,
    SALINE,
    apply_cohort_env,
    build_cohort_model,
    simulate_cohort,
    spec_from_config,
)
from photosvi.inference.points import (
    EstimandPlateau,
    FitQuality,
    block_spread,
    prepare,
    residual_ratio,
    resolve_device,
)

DELTA, SEED = float(sys.argv[1]), int(sys.argv[2])
# Empty means the config's fit.hier_beta_guide. The variant arms set GUIDE to
# override it; nothing should have to set it to get the shipped family.
GUIDE = os.environ.get("GUIDE") or None
# Couple the event guide to the sampled coefficients. Off by every arm so far.
COUPLE = os.environ.get("COUPLE", "0") == "1"
# Force the random-effect levels, e.g. LEVELS=animal,b0_session to drop the
# animal x drug interaction. Empty means let the design decide.
LEVELS = tuple(x for x in os.environ.get("LEVELS", "").split(",") if x) or None
DEVICE = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("DEVICE", "auto")
OUT = os.environ.get("OUT", "runs_cohort")
# The config is an environment variable so the three arms cannot be pointed at
# different studies: the grid sets it once and every arm of a cell reads the same
# file, which is the only reason their cohorts are identical.
cfg = apply_cohort_env(
    yaml.safe_load(open(os.environ.get("CONFIG", "configs/cohort_comparison.yaml"))))
# MARK_PRIOR_MU_SD / MARK_PRIOR_LS_SD override the widths of the mark prior (2026-10-10).
if os.environ.get("MARK_PRIOR_MU_SD"):
    cfg["fit"].setdefault("mark_prior", {})["mu_sd"] = float(os.environ["MARK_PRIOR_MU_SD"])
if os.environ.get("MARK_PRIOR_LS_SD"):
    cfg["fit"].setdefault("mark_prior", {})["log_sigma_sd"] = float(os.environ["MARK_PRIOR_LS_SD"])
# MARK_GATE_FLOOR overrides the config's floor on the gated mark divergence (2026-10-10).
if os.environ.get("MARK_GATE_FLOOR") is not None:
    cfg["fit"]["mark_gate_floor"] = float(os.environ["MARK_GATE_FLOOR"])
c, f, w = cfg["cohort"], cfg["fit"], cfg["warm"]

# The overrides live in `apply_cohort_env`, applied above, because every arm has to
# apply the same ones: see its docstring for what happened when only this one did.

# One definition, shared with the other arm: see spec_from_config.
spec = spec_from_config(cfg, DELTA)
coh = simulate_cohort(spec, SEED)
# SWAP_DRUGS=1 exchanges the two condition labels of a two-condition cohort before fitting
# (2026-10-14): a diagnostic for an asymmetry in how the model treats the conditions. The
# recordings are unchanged, so the estimated contrast should change sign and nothing else.
if os.environ.get("SWAP_DRUGS") == "1":
    _d = coh["drug"]
    coh["drug"] = (1 - _d) if hasattr(_d, "__sub__") else [1 - x for x in _d]
S = spec.n_sessions

# The model is built by the library, not here, so that every arm of the study fits
# the same one. See build_cohort_model.
model, Y, _ = build_cohort_model(cfg, spec, coh, guide=GUIDE,
                                couple_events=COUPLE, levels=LEVELS)
device = resolve_device(DEVICE, batch=S)
# FIX_MARKS=1 freezes mu_a and sigma_a at their starting values (the pooled empirical
# estimate when the config has a mark prior), 2026-10-10.
if os.environ.get("FIX_MARKS") == "1":
    model.phasic.mu_a.requires_grad_(False)
    model.phasic.log_sigma_a.requires_grad_(False)
model, Y = prepare(model, Y, device)

pyro.clear_param_store()
pyro.set_rng_seed(SEED)
svi = SVI(model.model, model.guide, ClippedAdam({"lr": f["lr"]}),
          Trace_ELBO(max_plate_nesting=2))
# Watch the two OPIOID LEVELS. Both alternatives were measured and both are wrong:
#
#   * All three levels (max drift): `mu_saline` never settles. Vehicle has a small
#     tonic bump (A = 1.2 against 7.0 and 6.0), so its pharmacokinetics are weakly
#     identified and its level wanders. Five of six cohorts ran past 30000 steps
#     toward the 60000 ceiling -- 64 h of GPU for a coordinate the study does not
#     report.
#   * The contrast alone: it fires at the 18000-step floor, and the recorded probe
#     traces show why that is a trap. At 18000 the levels are still climbing hard
#     (mu_op1 at 0.92 against a true 1.10) but climbing TOGETHER, so the contrast
#     looks flat. Both levels are then shrunk by roughly the same factor, which is
#     invisible at delta = 0 -- zero times anything is zero -- and biases every
#     non-null cell toward zero. Clean nulls and quietly shrunken alternatives is
#     the worst failure available here, because nothing in the null cells shows it.
#
# So: the two opioid levels, whose climb is the actual convergence signal, and not
# saline, which is the one coordinate the data genuinely cannot pin down.
# The drug indices come from the SPEC, not from the three-drug constants, because the
# two-condition study has no second opioid and indexing one is how this script first met it.
# Two conditions: a vehicle arm and a treated arm, the estimand is treated minus vehicle, and
# BOTH levels are watched -- the reasoning above against watching the vehicle level is about
# a vehicle with a small tonic bump and therefore barely identified pharmacokinetics, which
# is true of the three-drug study and false here, where both arms receive the same injection
# and differ only in rate.
if spec.n_drugs >= 3:
    NAMES = DRUG_NAMES
    ESTIMAND = (OPIOID2, OPIOID1)
    WATCH = (OPIOID1, OPIOID2)
    CONTRASTS = ((OPIOID1, SALINE), (OPIOID2, SALINE), (OPIOID2, OPIOID1))
else:
    NAMES = ("vehicle", "treated")
    ESTIMAND = (1, 0)
    WATCH = (0, 1)
    CONTRASTS = ((1, 0),)
plateau = EstimandPlateau(tol=f["stop_tol"], window=f["stop_window"],
                          min_steps=f["stop_min_steps"])
# Watches the FIT, which the plateau rule does not. It stops when the estimand stops
# moving, and on 2026-09-17 a real recording satisfied that at step 39000 with a
# post-injection residual eleven times its emission noise -- the tonic had been
# oscillating for the whole run and nothing was looking. Reported, never enforced.
quality = FitQuality()
rho_per: list[float] = []
stopped_at, trace = None, []
import time  # noqa: E402, I001
_t0 = time.time()
for step in range(1, f["max_steps"] + 1):
    svi.step(Y)
    if step % f["stop_probe"] == 0:
        cs = [model.drug_contrast(x, y)[0] for x, y in CONTRASTS]
        lv = [model.drug_posterior()[d][0] for d in range(spec.n_drugs)]
        # The estimand's own width and location, recorded every probe. Both oscillate
        # around a stable mean with no trend -- measured at 26-35% and about 50%
        # relative -- so reading either at whichever step the rule happened to fire
        # samples that oscillation once. The averages below are what should be
        # reported; the single values are kept so the two can be compared.
        est = model.drug_contrast(*ESTIMAND)
        rho, rho_per = residual_ratio(model, Y)
        quality.update(step, rho)
        trace.append({"step": step, "contrasts": cs, "levels": lv,
                      "estimand_loc": est[0], "estimand_scale": est[1],
                      "residual_ratio": rho})
        if plateau.update(step, [lv[i] for i in WATCH]):
            stopped_at = step
            break

# Averaged over the final probes rather than read at the stopping step. The
# stopping rule watches the two opioid levels; nothing watches the estimand's width,
# and it moves by a quarter of itself between consecutive probes.
TAIL = int(f.get("report_tail", 6))
tail = [t for t in trace if "estimand_scale" in t][-TAIL:]
est_loc_avg = sum(t["estimand_loc"] for t in tail) / len(tail) if tail else float("nan")
est_scale_avg = sum(t["estimand_scale"] for t in tail) / len(tail) if tail else float("nan")

post = model.drug_posterior()
contrasts = {f"{NAMES[a]}-{NAMES[b]}": model.drug_contrast(a, b) for a, b in CONTRASTS}
loc, scale = contrasts[f"{NAMES[ESTIMAND[0]]}-{NAMES[ESTIMAND[1]]}"]
out = dict(
    arm="hierarchical", delta=DELTA, seed=SEED, device=str(device),
    design=model.design_summary(),
    stopped_at=stopped_at, plateau=plateau.plateau_means, drift=plateau.drift,
    fit_quality=quality.summary(), block_spread=block_spread(), residual_ratio_per_session=rho_per,
    probe_trace=trace,
    mu_drug={NAMES[d]: post[d] for d in range(spec.n_drugs)},
    contrasts=contrasts,
    estimand_loc=loc, estimand_scale=scale,
    # What to report: the same quantities averaged over the last `report_tail`
    # probes. Six probes cut the sampling noise on each from about half its value to
    # a fifth, and cost nothing -- they were already being computed.
    estimand_loc_avg=est_loc_avg, estimand_scale_avg=est_scale_avg,
    report_tail=len(tail),
    estimand_z=loc / scale if scale > 0 else float("nan"),
    # Pass the estimand explicitly: true_contrast defaults to the third drug's index,
    # which a two-condition spec does not have.
    true_contrast=spec.true_contrast(*ESTIMAND),
    variance_components=model.variance_components(),
    # The per-drug pharmacokinetics: what a per-session tonic fit cannot produce.
    tonic_posterior={NAMES[d]: v for d, v in model.tonic_posterior().items()},
    tonic_peak_true=coh["tonic_peak_s"],
    session_beta_inj=[float(x) for x in model.session_beta_inj().cpu()],
    session_beta_inj_true=coh["beta_inj_true"],
    session_mle=coh["mle"],
)
os.makedirs(OUT, exist_ok=True)
# The family actually fitted, not the request. With GUIDE unset the request is None
# and the config decides; a run record that said `null` would be unreadable later.
out["guide"] = model.beta_guide
# The tonic guide's own posterior widths, and whether any sat on the bound. A scale
# at the ceiling is the ceiling being reported rather than a measurement, and nothing
# recorded it until a cohort diverged and had to be re-run under instrumentation to
# find out which parameter ran away.
out["tonic_scales"] = model.tonic_scales()
out["tonic_guide_widths"] = model.tonic_guide_widths()
# The posterior of the per-condition tonic amplitude means, alpha_d = log P_d, location and
# scale, for the tonic-response benchmark (2026-10-20). Factorized in the guide, so the
# contrast's width is the two scales in quadrature.
try:
    out["tonic_alpha_q"] = {
        NAMES[d]: {"loc": float(pyro.param("q_tonic_mu_alpha_loc").detach()[d]),
                   "scale": float(pyro.param("q_tonic_mu_alpha_scale").detach()[d])}
        for d in range(spec.n_drugs)}
except KeyError:
    out["tonic_alpha_q"] = None
# With a joint block over the tonic means (fit.tonic_block, opt-in) the two alphas are
# correlated and the contrast's width is not the quadrature sum; a model that can report the
# contrast itself is read first, whichever guide it was fitted with.
_tc = getattr(model, "tonic_alpha_contrast", None)
out["tonic_alpha_contrast"] = _tc() if callable(_tc) else None
out["tonic_block"] = bool(cfg["fit"].get("tonic_block", False))
# Wall time per step over the whole fit (includes probes; shared GPU, so comparable only
# between fits run side by side).
out["ms_per_step"] = 1000 * (time.time() - _t0) / max(1, step)
out["tonic_block_mode"] = getattr(model, "tonic_block_mode", "off")
out["couple_events"] = COUPLE
out["forced_levels"] = list(LEVELS) if LEVELS else None
out["sim_sigma_cell"] = spec.sigma_cell
_mu = model.phasic.mu_a.detach().reshape(-1).cpu()
# A list when the marks are per session (sigma_y per session under noise units).
out["mu_a_fit"] = float(_mu[0]) if _mu.numel() == 1 else [float(v) for v in _mu]
out["sigma_y_sessions"] = list(getattr(model, "sigma_y_sessions", []) or [])
if model.tonic_hierarchy:
    out["tonic_contrast_alpha"] = list(model.tonic_contrast(1, 0, "alpha"))
    out["tonic_block"] = bool(model.tonic_block)
out["sigma_a_fit"] = float(model.phasic.log_sigma_a.detach().exp())
out["mu_a_true"], out["sigma_a_true"] = float(spec.mu_a), float(spec.sigma_a)
out["mark_prior"] = list(model.phasic.mark_prior) if model.phasic.mark_prior else None
out["tau"] = float(model.tau) if hasattr(model, "tau") else None
out["fix_marks"] = os.environ.get("FIX_MARKS") == "1"
json.dump(out, open(f"{OUT}/hier_{DELTA:.2f}_{SEED:04d}.json", "w"))
tp = out["tonic_posterior"]
peaks = "  ".join(f"{n}:{tp[n]['peak_s']:.0f}s" for n in NAMES if n in tp)
true_peaks = "  ".join(f"{n}:{p:.0f}s" for n, p in zip(NAMES, out["tonic_peak_true"]))
print(quality.verdict())
print(f"hier d={DELTA:.2f} s{SEED:04d} | stopped {stopped_at} | "
      f"{NAMES[ESTIMAND[0]]}-{NAMES[ESTIMAND[1]]} {loc:+.3f} +- {scale:.3f} "
      f"(true {spec.true_contrast(*ESTIMAND):+.3f}) "
      f"| z {out['estimand_z']:+.2f} | "
      f"sig_anim {model.variance_components().get('sigma_animal', float('nan')):.3f}\n"
      f"    tonic peak fit  {peaks}\n"
      f"    tonic peak true {true_peaks}")
assert math.isfinite(loc)
