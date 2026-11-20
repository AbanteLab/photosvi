#!/usr/bin/env python3
"""Linear-response correction of the variational covariance, by finite differences.

    python scripts/linear_response.py <seed>

The covariance of a distribution is how far its mean moves when you tilt it. Add
`t . theta` to the log joint and, for the exact posterior, `dE[theta]/dt` is exactly
`Cov(theta)` -- the cumulant-generating identity. Linear response claims that a
variational fit's *response* to that tilt approximates the exact one even when its own
covariance does not, which is precisely our situation: the mean is right and the width
is not.

Finite differences rather than Hessian-vector products, deliberately. The exact form
needs `(I - Sigma H)^-1` over ~16000 dimensions and the algebra has to be right;
this needs only the definition, and if it works the exact version is worth building.
The cost is a warm-started refit per tilt, which is cheap next to the original fit.

Three things are swept, because each could break it:

  * **tilt size.** Too small and the response drowns in the stochastic objective's
    own noise; too large and the response stops being linear. Both failure modes look
    like a wrong answer, so the sweep is the diagnostic: a plateau across `t` is what
    says the estimate is real.
  * **refit length.** The tilted optimum has to actually be reached. A short refit
    understates the response and would masquerade as a small covariance.
  * **which covariance entry.** Tilting beta0 and reading beta_inj gives the
    off-diagonal -- the term a factorised guide sets to zero and the one we measured
    at -0.88 with the joint block.

Validated against the NUTS samples in runs_hmc_chains/, on the same recordings.
"""
from __future__ import annotations

import copy
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
T = int(os.environ.get("T", 8000))
BETA0 = float(os.environ.get("BETA0", -2.2))
SURGERY = os.environ.get("SURGERY", "off") == "on"
BASE_STEPS = int(os.environ.get("BASE_STEPS", 20000))
TILTS = [float(x) for x in os.environ.get("TILTS", "0.5,1.0,2.0,4.0").split(",")]
REFITS = [int(x) for x in os.environ.get("REFITS", "1000,3000").split(",")]
REFIT_LR_SCALE = float(os.environ.get("REFIT_LR_SCALE", 1.0))
OUT = os.environ.get("OUT", "runs_lr")

cfg = yaml.safe_load(open("configs/toy.yaml"))
k, gt, ton, f, w = cfg["kernel"], cfg["gt"], cfg["tonic"], cfg["fit"], cfg["warm"]
gt = dict(gt, beta0=BETA0)
dt, INJ = 1 / cfg["data"]["fs"], T // 2

kernel = build_kernel(dict(k["fit"]), k["length"], dt)
g = torch.Generator().manual_seed(SEED)
x_true = TonicBaseline(dt=dt, **ton).baseline(T, INJ)
ex = PhasicEventProcess(kernel, beta0=gt["beta0"], beta_inj=gt["beta_inj"],
                        mu_a=gt["mu_a"], sigma_a=gt["sigma_a"], dt=dt,
                        history=None).sample(T, INJ, 1, g)
y = AdditiveEmission(sigma_y=gt["sigma_y"]).sample(x_true + ex["c"], generator=g)["y"]
n_pre = int(ex["s"][0][:INJ].sum())
n_post = int(ex["s"][0][INJ:].sum())
mle = (math.log((n_post / ((T - INJ) * dt)) / (n_pre / (INJ * dt)))
       if n_pre and n_post else float("nan"))

mf = matched_filter_score(y, kernel, INJ, dt=dt)
b0a, _ = mf_anchor(mf, INJ, dt, thresh=cfg["detection"]["mf_thresh"],
                   refractory=cfg["detection"]["refractory"])
emission = AdditiveEmission(sigma_y=gt["sigma_y"])
emission.log_sigma_y.requires_grad_(False)
tonic = fit_tonic_baseline(y, INJ, dt=dt)
for prm in tonic.parameters():
    prm.requires_grad_(False)

torch.set_num_threads(2)
model = SingleTraceModel(
    PhasicEventProcess(kernel, beta0=b0a, beta_inj=0.0, mu_a=gt["mu_a"],
                       sigma_a=gt["sigma_a"], dt=dt, history=None),
    tonic, emission, tau=f["tau"],
    beta0_prior_loc=b0a, beta_inj_prior_loc=0.0, beta_prior="normal",
    beta0_prior_scale=3.0, beta_inj_prior_scale=2.0,
    event_guide=WindowedCategoricalEventGuide(
        torch.as_tensor(mf, dtype=torch.float32), window=f["window"],
        thresh=w["thresh"], gain=w["gain"]),
    analytic_s_kl=SURGERY, mark_kl=("gated" if SURGERY else "sampled"),
    beta_guide="meanfield",          # the fit LR is asked to correct
)
model.train(False)

opt = ClippedAdam({"lr": f["lr"]})
elbo = Trace_ELBO(max_plate_nesting=2)


RAMP = float(os.environ.get("RAMP", 0.25))


def run(steps, tilt=None, lr=None, fresh=False):
    """`fresh` gives the refit its own optimiser and ramps the tilt in.

    Both matter for the refits and neither for the base fit. ClippedAdam caches
    moment state per parameter, so one optimiser shared across refits carries
    momentum from the +t pass into the -t pass and the central difference absorbs it
    as signal. And a tilt switched on as a step change is a constant gradient that
    Adam walks at nearly the full learning rate until the curvature catches up;
    ramped, the optimiser tracks a moving optimum to the same fixed point.
    """
    svi = SVI(model.model, model.guide,
              ClippedAdam({"lr": lr or f["lr"]}) if fresh else opt, elbo)
    ramp = max(1, int(RAMP * steps)) if fresh else 1
    for i in range(steps):
        if tilt is None:
            svi.step(y, INJ, None)
        else:
            g = min(1.0, (i + 1) / ramp)
            svi.step(y, INJ, (tilt[0] * g, tilt[1] * g))
    # get_state, not the items: the store returns CONSTRAINED values, and writing
    # those back into a cleared store registers them as unconstrained -- a positive
    # scale then comes back negative on the first restore.
    return copy.deepcopy(pyro.get_param_store().get_state())


pyro.clear_param_store()
pyro.set_rng_seed(SEED)
base = run(BASE_STEPS)
bp = model.beta_posterior()
mf_scale = bp["beta_inj"][1]


def mean_after(tilt, steps):
    """Refit warm-started from the converged fit and read the new mean."""
    pyro.get_param_store().set_state(copy.deepcopy(base))
    pyro.set_rng_seed(SEED + 991)
    run(steps, tilt=tilt, lr=f["lr"] * REFIT_LR_SCALE, fresh=True)
    b = model.beta_posterior()
    return b["beta_inj"][0], b["beta0"][0]


res = []
for steps in REFITS:
    for t in TILTS:
        # central difference: the antisymmetric part cancels the fit's own drift
        pi, p0 = mean_after((0.0, t), steps)
        mi, m0 = mean_after((0.0, -t), steps)
        var_inj = (pi - mi) / (2 * t)
        # tilting beta0 and reading beta_inj gives the off-diagonal
        qi, _ = mean_after((t, 0.0), steps)
        qj, _ = mean_after((-t, 0.0), steps)
        cov = (qi - qj) / (2 * t)
        res.append(dict(tilt=t, refit_steps=steps, var_inj=var_inj,
                        sd_inj=math.sqrt(var_inj) if var_inj > 0 else float("nan"),
                        cov_b0_inj=cov))
        print(f"seed {SEED} t={t:<4g} refit={steps:<5d} | LR sd "
              f"{res[-1]['sd_inj']:.4f} | mean-field sd {mf_scale:.4f} | "
              f"cov(b0,binj) {cov:+.5f}", flush=True)

os.makedirs(OUT, exist_ok=True)
json.dump(dict(seed=SEED, T=T, surgery=SURGERY, mle=mle, n_pre=n_pre, n_post=n_post,
               mf_loc=bp["beta_inj"][0], mf_scale=mf_scale, results=res),
          open(f"{OUT}/lr_{SEED:03d}.json", "w"))
