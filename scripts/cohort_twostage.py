"""All sessions of one cohort, fit one at a time by the single-trace model. CPU.

    python scripts/cohort_twostage.py <delta> <seed> [device]

Writes runs_cohort/ts_<delta>_<seed>_<session>.json per session, as each completes,
so an interrupted cohort resumes rather than restarting.

**One process per cohort, not per session**, which is not the obvious choice. Per
session would give 3600 independent units instead of 150, but the cohort has to be
regenerated to get at any one of its traces, and the forward sampler walks the time
axis in Python: 3 s per session, 1.2 min per cohort. Paying that once per session
would add 72 h of pure simulation to the study; paying it once per cohort costs 3 h.
The 150 units still saturate a 24-way fan-out.
The two-stage arms are assembled from these afterwards by `cohort_report.py`:

  * *session-level*: a two-sample test on the per-session estimates, which is
    pseudoreplication -- two sessions of one animal are not two pieces of evidence
    about a drug. It is included because it is what gets done.
  * *animal-averaged*: the same estimates averaged within animal first, then a
    paired test across animals. This is the correct classical analysis and it is the
    comparator the hierarchical arm has to beat.

Note what neither can use: in the opioid1-vs-opioid2 contrast the **saline sessions
drop out entirely**. The hierarchical arm does use them -- they inform every variance
component, which sets the width of all the contrasts. That is a third of the
recordings, and with 8 animals the variance components are exactly what is fragile.

The cohort is regenerated from `(spec, seed)`, which is deterministic, so this and
the hierarchical worker fit the same recordings without sharing a cache.

**The guide is read from the config and not defaulted.** Until 2026-09-14 this call
omitted `beta_guide`, so all 519 per-session fits of the study took SingleTraceModel's
mean-field default -- the family the guide sweep had already rejected -- and every
interval this arm reported was 3.91x narrower than the estimator's actual spread
(90% coverage: 22%). The comparator the hierarchical arm had to beat was therefore
beating it on a handicap that came from a missing keyword argument, not from the
two-stage design. Numbers and method: scripts/session_guide_test.py.
"""
from __future__ import annotations

import json
import math
import os
import sys

import pyro
import torch
import yaml
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from photosvi.benchmark import matched_filter_score
from photosvi.experiments.cohort import (
    apply_cohort_env,
    simulate_cohort,
    spec_from_config,
)
from photosvi.experiments.sweep import mf_anchor
from photosvi.inference.points import (
    EstimandPlateau,
    FitQuality,
    prepare,
    residual_ratio,
    resolve_device,
)
from photosvi.models.points import (
    AdditiveEmission,
    PhasicEventProcess,
    SingleTraceModel,
    WindowedCategoricalEventGuide,
    build_kernel,
    fit_tonic_baseline,
)

DELTA, SEED = float(sys.argv[1]), int(sys.argv[2])
DEVICE = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("DEVICE", "cpu")
OUT = os.environ.get("OUT", "runs_cohort")
# The config is an environment variable so the three arms cannot be pointed at
# different studies: the grid sets it once and every arm of a cell reads the same
# file, which is the only reason their cohorts are identical.
cfg = apply_cohort_env(
    yaml.safe_load(open(os.environ.get("CONFIG", "configs/cohort_comparison.yaml"))))
c, f, w, det = cfg["cohort"], cfg["fit"], cfg["warm"], cfg["detection"]

# One definition, shared with the other arm: see spec_from_config.
spec = spec_from_config(cfg, DELTA)
os.makedirs(OUT, exist_ok=True)
# SESSIONS="0,1,2" restricts the cohort to those sessions (sweeps that need a subset).
_only = os.environ.get("SESSIONS")
todo = [s for s in range(spec.n_sessions) if not _only or s in {int(x) for x in _only.split(",")}]
todo = [s for s in todo
        if not os.path.exists(f"{OUT}/ts_{DELTA:.2f}_{SEED:04d}_{s:02d}.json")]
if not todo:
    print(f"ts d={DELTA:.2f} s{SEED:04d} | all {spec.n_sessions} sessions already done")
    raise SystemExit(0)

coh = simulate_cohort(spec, SEED)
dt, T = spec.dt, spec.T
kern = build_kernel(dict(spec.kernel), spec.kernel_length, dt)
device = resolve_device(DEVICE, batch=1)

for SESSION in todo:
    y = coh["y"][SESSION:SESSION + 1]
    INJ = coh["inj_t"][SESSION]
    mf = matched_filter_score(y, kern, INJ, dt=dt)
    b0a, _ = mf_anchor(mf, INJ, dt, thresh=det["mf_thresh"], refractory=det["refractory"])

    from photosvi.experiments.marks import fix_marks_at, noise_unit_marks, noise_unit_mode
    from photosvi.experiments.single_trace import build_single_trace, is_final
    _nu = noise_unit_mode(f)
    mark_hat = None
    if is_final(cfg):
        # The paper's final single architecture (experiments.single_trace): the same model
        # the detection benchmark and photosvi.api fit, sigma_y read off this recording.
        model, _info = build_single_trace(
            y, INJ, kernel=kern, dt=dt, fit=f, warm=w, detection=det,
            ar_coeffs=list(spec.ar_coeffs) if spec.ar_coeffs else None)
        sig_fit = _info["sigma_y"]
    else:
        # The same emission the hierarchical arm fits (build_cohort_model): the cohort's AR
        # coefficients when the config has them. Until 2026-10-07 this arm was white whatever
        # the config said, so on an AR(2) cohort it compared a misspecified single-trace fit
        # against a correctly specified hierarchy.
        # Noise-unit marks (opt-in, experiments.marks): sigma_y is estimated from this
        # recording, as the single-trace model does, and the marks are fixed in its units.
        # Otherwise the simulated sigma_y, as before (2026-10-20).
        from photosvi.experiments.marks import fix_marks_at, noise_unit_marks, noise_unit_mode
        _nu = noise_unit_mode(f)
        sig_fit = spec.sigma_y
        if _nu is not None:
            from photosvi.models.points import estimate_log_noise_scale
            sig_fit = math.exp(estimate_log_noise_scale(y, INJ, ar_coeffs=spec.ar_coeffs or None))
        em = (AdditiveEmission(sigma_y=sig_fit, ar_coeffs=spec.ar_coeffs)
              if spec.ar_coeffs else AdditiveEmission(sigma_y=sig_fit, ar_rho=spec.ar_rho))
        em.log_sigma_y.requires_grad_(False)
        model = SingleTraceModel(
            PhasicEventProcess(kern, beta0=b0a, beta_inj=0.0, mu_a=spec.mu_a,
                               sigma_a=spec.sigma_a, dt=dt, history=None),
            fit_tonic_baseline(y, INJ, dt=dt), em, tau=float(os.environ.get("TAU", f["tau"])),
            beta0_prior_loc=b0a, beta_inj_prior_loc=0.0,
            # BETA_PRIOR overrides the config for this arm alone. The config carries
            # `uniform`, which only reaches here -- the hierarchical model's globals are
            # Normal by construction and it does not take the setting -- so the
            # comparator is fitted under a prior whose reported width is a delta-method
            # linearisation on a transformed axis, measured 11% narrower than the
            # unbounded Normal on the same recordings (S1 Table). That makes the arm the
            # hierarchy has to beat look more precise than it is, on the axis the paper
            # compares. The override exists so the two can be run and compared rather
            # than the config edited and the difference lost.
            beta_prior=os.environ.get("BETA_PRIOR") or f["beta_prior"],
            beta_low=b0a - f["beta0_halfwidth"], beta_high=b0a + f["beta0_halfwidth"],
            beta_inj_low=-f["beta_inj_halfwidth"], beta_inj_high=f["beta_inj_halfwidth"],
            beta0_prior_scale=f["beta0_prior_scale"],
            event_guide=WindowedCategoricalEventGuide(
                torch.as_tensor(mf, dtype=torch.float32), window=f["window"],
                thresh=w["thresh"], gain=w["gain"]),
            # Indexed, not `.get(..., default)`: a config that forgets this key should
            # stop the run, not quietly fit the whole study with the constructor's
            # mean-field default. That is exactly what happened to the first 519 fits.
            beta_guide=f["beta_guide"],
            couple_events=False)
        # MARK_PRIOR=1 anchors the mark distribution on this recording's own visible peaks
        # (photosvi.experiments.marks), with a prior of the stated widths; MARK_INIT=prior
        # starts q(a) at mu_a rather than at the noise scale (2026-10-09).
        mark_hat = None
        _mp = f.get("mark_prior") or {}
        if _nu is None and (os.environ.get("MARK_PRIOR") == "1"
                            or (_mp and os.environ.get("MARK_PRIOR") != "0")):
            from photosvi.experiments.marks import estimate_mark_distribution
            mark_hat = estimate_mark_distribution([y[0, :INJ].numpy()], 1.0 / dt)
            model.phasic.set_mark_prior(*mark_hat,
                                        mu_sd=float(os.environ.get("MARK_PRIOR_MU_SD", _mp.get("mu_sd", 0.2))),
                                        log_sigma_sd=float(os.environ.get("MARK_PRIOR_LS_SD", _mp.get("log_sigma_sd", 0.3))))
        if _nu is not None:
            fix_marks_at(model.phasic, *noise_unit_marks(sig_fit, *_nu))
    if os.environ.get("MARK_INIT"):
        model.phasic.q_a_init = os.environ["MARK_INIT"]
    # FIX_MARKS=1 holds the mark distribution at its simulated value instead of fitting it
    # (2026-10-09): the test of whether rate and amplitude trade off at low SNR.
    if os.environ.get("FIX_MARKS") == "1":
        for name, prm in model.named_parameters():
            if name.endswith("mu_a") or name.endswith("log_sigma_a"):
                prm.requires_grad_(False)
    model.train(False)
    model_d, y_d = prepare(model, y, device)

    # A fresh param store per session: these are independent fits, and leaving the
    # previous session's params in place would warm-start each from its neighbour.
    pyro.clear_param_store()
    pyro.set_rng_seed(SEED * 1000 + SESSION)
    svi = SVI(model_d.model, model_d.guide, ClippedAdam({"lr": f["lr"]}),
              Trace_ELBO(max_plate_nesting=2))
    plateau = EstimandPlateau(tol=f["stop_tol"], window=f["stop_window"],
                              min_steps=f["stop_min_steps"])
    # Same diagnostic as the hierarchical arm, so the comparison includes whether
    # each arm's fits explained their recordings and not only what they reported.
    quality = FitQuality()
    stopped_at = None
    for step in range(1, f["max_steps"] + 1):
        svi.step(y_d, INJ)
        if step % f["stop_probe"] == 0:
            quality.update(step, residual_ratio(model_d, y_d, INJ, draws=2)[0])
            if plateau.update(step, float(model_d.beta_posterior()["beta_inj"][0])):
                stopped_at = step
                break

    bp = model_d.beta_posterior()
    out = dict(
        arm="two-stage", delta=DELTA, seed=SEED, session=SESSION, device=str(device),
        beta_guide=f["beta_guide"],
        animal=coh["animal"][SESSION], drug=coh["drug"][SESSION], inj_t=INJ,
        stopped_at=stopped_at, plateau=plateau.plateau_means,
        fit_quality=quality.summary(),
        beta_inj=bp["beta_inj"][0], beta_inj_scale=bp["beta_inj"][1],
        beta0=bp["beta0"][0],
        beta0_true=float(coh["beta0_true"][SESSION]),
        mu_a_fit=float(dict(model_d.named_parameters())[[n for n, _ in model_d.named_parameters() if n.endswith("mu_a")][0]]),
        sigma_a_fit=float(dict(model_d.named_parameters())[[n for n, _ in model_d.named_parameters() if n.endswith("log_sigma_a")][0]].exp()),
        mu_a_true=spec.mu_a, sigma_a_true=spec.sigma_a, sigma_y=spec.sigma_y,
        fix_marks=os.environ.get("FIX_MARKS") == "1",
        mark_prior=(list(model_d.phasic.mark_prior) if model_d.phasic.mark_prior else None),
        mark_hat=(list(mark_hat) if mark_hat else None),
        q_a_init=model_d.phasic.q_a_init,
        tau=float(os.environ.get("TAU", f["tau"])),
        beta_inj_true=coh["beta_inj_true"][SESSION], mle=coh["mle"][SESSION],
        n_pre=coh["n_pre"][SESSION], n_post=coh["n_post"][SESSION],
        # The fitted tonic response, for the tonic-response benchmark (2026-10-20).
        sigma_y_fit=float(sig_fit), mark_mode="noise_units" if _nu is not None else None,
        architecture=f.get("architecture", "legacy"),
        tonic_peak=float(model_d.tonic.peak.detach()),
        tonic_k_rise=float(model_d.tonic.k_rise.detach()),
        tonic_k_decay=float(model_d.tonic.k_decay.detach()),
        tonic_peak_true=float(spec.tonic_peak[int(coh["drug"][SESSION])]))
    json.dump(out, open(f"{OUT}/ts_{DELTA:.2f}_{SEED:04d}_{SESSION:02d}.json", "w"))
    print(f"ts d={DELTA:.2f} s{SEED:04d} sess{SESSION:02d} "
          f"(a{out['animal']} d{out['drug']}) | stopped {stopped_at} "
          f"| b_inj {out['beta_inj']:+.3f} (true {out['beta_inj_true']:+.3f}, "
          f"MLE {out['mle']:+.3f})", flush=True)
    assert math.isfinite(out["beta_inj"])
