#!/usr/bin/env python3
"""One cell of the prior comparison: same recording, different prior on the betas.

    python scripts/prior_comparison.py <prior> <seed>
        prior: "uniform" | "normal-wide" | "normal-tight" | "flat"

The bounded Uniform prior we have been using is what forces the guide onto a
transformed axis, and every awkwardness downstream follows from that: the
sigmoid-affine map, reading the estimate back through it, a delta-method scale that
is wrong in the tails, and -- once the coefficients are given a joint block -- a
change-of-variables term that is easy to leave out and produces a silently wrong
objective when you do.

A wide Normal removes all of it. The question this measures is the only one that
matters for the choice: does it move the estimate? The Uniform was adopted precisely
because a Normal(0, 1) was found to shrink beta_inj by about 2 per cent, and a
reviewer asking "did your prior shrink your effect?" deserves a number rather than an
appeal to flatness.

`flat` is an improper uniform on the real line: zero shrinkage by construction, and
the posterior is proper whenever events are observed either side of the injection. It
is included as the limiting case, to show what the Normal's width is costing.
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
from photosvi.experiments.sweep import gen_spec, mf_anchor, setup_from_config
from photosvi.inference.points import EstimandPlateau
from photosvi.models.points import (
    AdditiveEmission,
    PhasicEventProcess,
    SingleTraceModel,
    TonicBaseline,
    WindowedCategoricalEventGuide,
    build_kernel,
    fit_tonic_baseline,
)

PRIOR, SEED = sys.argv[1], int(sys.argv[2])
GUIDE = os.environ.get("GUIDE", "fullrank")
OUT = os.environ.get("OUT", "runs_prior")
cfg = yaml.safe_load(open("configs/points_guide_sweep_k5.yaml"))
d, k, gt, ton, det, f = (cfg["data"], cfg["kernel"], cfg["gt"], cfg["tonic"],
                         cfg["detection"], cfg["fit"])
dt, T, INJ = 1 / d["fs"], d["T"], d["inj_t"]

g = torch.Generator().manual_seed(SEED)
x_true = TonicBaseline(dt=dt, **ton).baseline(T, INJ)
ex = PhasicEventProcess(build_kernel(gen_spec(k), k["length"], dt), beta0=gt["beta0"],
                        beta_inj=gt["beta_inj"], mu_a=gt["mu_a"], sigma_a=gt["sigma_a"],
                        dt=dt, history=None).sample(T, INJ, 1, g)
y = AdditiveEmission(sigma_y=gt["sigma_y"]).sample(x_true + ex["c"], generator=g)["y"]
ons = ex["s"][0].nonzero().reshape(-1)
npre, npost = int((ons < INJ).sum()), int((ons >= INJ).sum())
mle = (math.log((npost / ((T - INJ) * dt)) / (npre / (INJ * dt)))
       if npre and npost else float("nan"))

mf = matched_filter_score(y, build_kernel(dict(k["fit"]), k["length"], dt), INJ, dt=dt)
b0a, _ = mf_anchor(mf, INJ, dt, thresh=det["mf_thresh"], refractory=det["refractory"])
setup = setup_from_config(cfg, b0a)

PRIORS = {
    # what we have been using: bounded, hence the transformed guide
    "uniform": dict(beta_prior="uniform",
                    beta_low=b0a - f["beta0_halfwidth"],
                    beta_high=b0a + f["beta0_halfwidth"],
                    beta_inj_low=-f["beta_inj_halfwidth"],
                    beta_inj_high=f["beta_inj_halfwidth"]),
    # the proposal: proper, unbounded, wide enough that shrinkage is negligible
    "normal-wide": dict(beta_prior="normal", beta0_prior_scale=3.0,
                        beta_inj_prior_scale=2.0),
    # the one that was rejected, kept so the rejection stays checkable
    "normal-tight": dict(beta_prior="normal", beta0_prior_scale=1.0,
                         beta_inj_prior_scale=1.0),
    # the limiting case: no shrinkage at all
    "flat": dict(beta_prior="normal", beta0_prior_scale=50.0,
                 beta_inj_prior_scale=50.0),
}
if PRIOR not in PRIORS:
    raise SystemExit(f"prior must be one of {sorted(PRIORS)} (got {PRIOR!r})")

torch.set_num_threads(1)
pyro.set_rng_seed(SEED)
em = AdditiveEmission(sigma_y=setup.sigma_y)
em.log_sigma_y.requires_grad_(False)
model = SingleTraceModel(
    PhasicEventProcess(setup.kernel(), beta0=b0a, beta_inj=0.0, mu_a=setup.mu_a,
                       sigma_a=setup.sigma_a, dt=dt, history=None),
    fit_tonic_baseline(y, INJ, dt=dt), em, tau=setup.tau,
    beta0_prior_loc=b0a, beta_inj_prior_loc=0.0,
    event_guide=WindowedCategoricalEventGuide(
        torch.as_tensor(mf, dtype=torch.float32), window=setup.window,
        thresh=setup.warm["thresh"], gain=setup.warm["gain"]),
    couple_events=False, beta_guide=GUIDE, **PRIORS[PRIOR])
model.train(False)
pyro.clear_param_store()
pyro.set_rng_seed(SEED)
svi = SVI(model.model, model.guide, ClippedAdam({"lr": f["lr"]}),
          Trace_ELBO(max_plate_nesting=2))
plateau = EstimandPlateau(tol=f["stop_tol"], window=f["stop_window"],
                          min_steps=f["stop_min_steps"])
stopped_at = None
for step in range(1, f["max_steps"] + 1):
    svi.step(y, INJ)
    if step % f["stop_probe"] == 0 and plateau.update(
            step, float(model.beta_posterior()["beta_inj"][0])):
        stopped_at = step
        break

bp = model.beta_posterior()
out = dict(prior=PRIOR, guide=GUIDE, seed=SEED, stopped_at=stopped_at,
           beta_inj=bp["beta_inj"][0], beta_inj_scale=bp["beta_inj"][1],
           beta0=bp["beta0"][0], beta0_scale=bp["beta0"][1], mle=mle,
           true_beta_inj=gt["beta_inj"], n_pre=npre, n_post=npost,
           corr=(model.beta_correlation() if GUIDE == "fullrank" else None))
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(f"{OUT}/{PRIOR}_{SEED:02d}.json", "w"))
print(f"{PRIOR:>12} s{SEED:02d} | b_inj {out['beta_inj']:+.4f} +- "
      f"{out['beta_inj_scale']:.4f} | MLE {mle:+.4f} | err {out['beta_inj']-mle:+.4f}"
      + (f" | corr {out['corr']:+.3f}" if out["corr"] is not None else ""), flush=True)
