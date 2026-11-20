#!/usr/bin/env python3
"""The production variational configuration, on the datasets HMC already sampled.

    python scripts/hmc_vi_production.py <seed>

`hmc_reference.py` had to disable the analytic-KL surgery to make the model
samplable, so its variational arm is the plain relaxed configuration -- not the one
we ship. That leaves the comparison one step short: it shows mean-field VI on the
plain relaxed model is far narrower than exact, and leaves transferring that to the
shipped configuration as an assumption.

This closes it. Same data, same frozen nuisances, same everything except that the
event factor is scored by the analytic Bernoulli divergence and the marks are tied
and gated, as they are in every result in the paper. Its posterior width can then be
compared against the HMC draws already stored.

The caveat that remains, and it is real: the analytic surgery changes the objective,
so the shipped configuration approximates a bound on the DISCRETE model while NUTS
samples the RELAXED one. The two targets are close but not identical, and this
comparison is therefore indicative rather than exact.
"""
from __future__ import annotations

import json
import math
import os
import sys

import pyro
import torch
import yaml
from pyro.infer import SVI, RenyiELBO, Trace_ELBO
from pyro.optim import ClippedAdam

from photosvi.benchmark import matched_filter_score
from photosvi.experiments.sweep import mf_anchor
from photosvi.models.points import (
    AdditiveEmission,
    PhasicEventProcess,
    SingleTraceModel,
    TonicBaseline,
    WindowedCategoricalEventGuide,
    build_kernel,
    fit_tonic_baseline,
)

SEED = int(sys.argv[1])
GUIDE = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("GUIDE", "meanfield")
# alpha < 1 makes the objective mass-covering rather than mode-seeking; alpha = 1 is
# the ordinary ELBO. Rényi needs several particles to mean anything -- the bound is
# built from importance weights ACROSS particles, so with one it collapses back to
# the ELBO and measures nothing.
ALPHA = float(os.environ.get("ALPHA", "1.0"))
PARTICLES = int(os.environ.get("PARTICLES", "1"))
# The analytic-KL surgery replaces sampled log-densities with an exact divergence,
# so part of the objective is no longer a log p / log q ratio at all. A Renyi bound
# reweights exactly those ratios, so with the surgery on it is reweighting something
# the theory does not describe. Turning it off also makes the comparison exact rather
# than indicative: NUTS sampled the plain relaxed model, and this is that model.
SURGERY = os.environ.get("SURGERY", "on") == "on"
T = int(os.environ.get("T", 8000))
BETA0 = float(os.environ.get("BETA0", -2.2))
OUT = os.environ.get("OUT", "runs_hmc_prod")
STEPS = int(os.environ.get("STEPS", 20000))

cfg = yaml.safe_load(open("configs/toy.yaml"))
k, gt, ton, f, w = cfg["kernel"], cfg["gt"], cfg["tonic"], cfg["fit"], cfg["warm"]
gt = dict(gt, beta0=BETA0)
dt, INJ = 1 / cfg["data"]["fs"], T // 2          # identical to hmc_reference.py

kernel = build_kernel(dict(k["fit"]), k["length"], dt)
g = torch.Generator().manual_seed(SEED)
x_true = TonicBaseline(dt=dt, **ton).baseline(T, INJ)
ex = PhasicEventProcess(kernel, beta0=gt["beta0"], beta_inj=gt["beta_inj"],
                        mu_a=gt["mu_a"], sigma_a=gt["sigma_a"], dt=dt,
                        history=None).sample(T, INJ, 1, g)
y = AdditiveEmission(sigma_y=gt["sigma_y"]).sample(x_true + ex["c"], generator=g)["y"]
n_pre = int((ex["s"][0][:INJ]).sum())
n_post = int((ex["s"][0][INJ:]).sum())
mle = (math.log((n_post / ((T - INJ) * dt)) / (n_pre / (INJ * dt)))
       if n_pre and n_post else float("nan"))

mf = matched_filter_score(y, kernel, INJ, dt=dt)
b0a, _ = mf_anchor(mf, INJ, dt, thresh=cfg["detection"]["mf_thresh"],
                   refractory=cfg["detection"]["refractory"])
emission = AdditiveEmission(sigma_y=gt["sigma_y"])
emission.log_sigma_y.requires_grad_(False)
tonic = fit_tonic_baseline(y, INJ, dt=dt)
for prm in tonic.parameters():
    prm.requires_grad_(False)                    # frozen, as in the HMC arm

torch.set_num_threads(2)
model = SingleTraceModel(
    PhasicEventProcess(kernel, beta0=b0a, beta_inj=0.0, mu_a=gt["mu_a"],
                       sigma_a=gt["sigma_a"], dt=dt, history=None),
    tonic, emission, tau=f["tau"],
    beta0_prior_loc=b0a, beta_inj_prior_loc=0.0, beta_prior=f["beta_prior"],
    beta_low=b0a - f["beta0_halfwidth"], beta_high=b0a + f["beta0_halfwidth"],
    beta_inj_low=-f["beta_inj_halfwidth"], beta_inj_high=f["beta_inj_halfwidth"],
    event_guide=WindowedCategoricalEventGuide(
        torch.as_tensor(mf, dtype=torch.float32), window=f["window"],
        thresh=w["thresh"], gain=w["gain"]),
    analytic_s_kl=SURGERY,
    mark_kl=("gated" if SURGERY else "sampled"),
    beta_guide=GUIDE,
)
model.train(False)
pyro.clear_param_store()
pyro.set_rng_seed(SEED)
if ALPHA == 1.0 and PARTICLES == 1:
    loss_fn = Trace_ELBO(max_plate_nesting=2)
    tag = "elbo"

else:
    loss_fn = RenyiELBO(alpha=ALPHA, num_particles=PARTICLES, max_plate_nesting=2)
    tag = f"renyi{ALPHA:g}p{PARTICLES}"
tag += "" if SURGERY else "-plain"
svi = SVI(model.model, model.guide, ClippedAdam({"lr": f["lr"]}), loss_fn)
for _ in range(STEPS):
    svi.step(y, INJ)
loc, scale = model.beta_posterior()["beta_inj"]

os.makedirs(OUT, exist_ok=True)
corr = model.beta_correlation() if GUIDE == "fullrank" else None
out = dict(seed=SEED, T=T, guide=GUIDE, objective=tag, alpha=ALPHA,
           surgery=SURGERY,
           particles=PARTICLES, mle=mle, n_pre=n_pre, n_post=n_post,
           prod_loc=float(loc), prod_scale=float(scale), beta_corr=corr)
json.dump(out, open(f"{OUT}/prod_{GUIDE}_{tag}_{SEED:03d}.json", "w"))
print(f"seed {SEED} [{GUIDE}/{tag}] {loc:+.3f} +- {scale:.4f} (MLE {mle:+.3f})"
      + (f" | corr(beta0, beta_inj) {corr:+.3f}" if corr is not None else ""),
      flush=True)
