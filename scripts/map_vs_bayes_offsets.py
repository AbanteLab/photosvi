#!/usr/bin/env python3
"""How much of the posterior overconfidence comes from MAP-ing the tonic offsets?

    python scripts/map_vs_bayes_offsets.py <arm> <seed> [device]
        arm: "map" (b, drift are point estimates) or "bayes" (they are latents)

The cohort contrast's credible interval does not match the estimator's spread across
replicates (measured at delta = 0: empirical SD 0.091 against a reported scale of
0.027). Several mechanisms could produce that and they call for different remedies;
this script tests one of them:

  * **mean-field factorization** -- q cannot represent posterior correlation, so
    marginal variances are understated. Fixed by a richer family (a full-rank guide
    over the globals) or a different divergence.
  * **MAP nuisance parameters** -- `b` and `drift` are point estimates, one pair per
    session, so their uncertainty is *zero by construction* and never propagates to
    the estimand. Fixed by making them latents. Nothing about the variational family
    changes this.

This runs the same cohorts both ways at delta = 0 and compares the overconfidence
factor. Deliberately at T/4 with the injection at the equivalent fraction: the
question is the RATIO between two arms at a fixed T, which does not need the full
recording, and a four-hour answer to a sizing question is not worth having. The
absolute factor at T/4 will not match the T = 60000 one -- a quarter of the events
per session is a noisier estimator -- and is not the point.
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
from photosvi.experiments.cohort import OPIOID1, OPIOID2, simulate_cohort, spec_from_config
from photosvi.inference.points import EstimandPlateau, prepare, resolve_device
from photosvi.models.points import (
    AdditiveEmission,
    HierarchicalModel,
    PhasicEventProcess,
    WindowedCategoricalEventGuide,
    build_kernel,
    fit_tonic_baseline,
    stack_tonic_baselines,
)

ARM, SEED = sys.argv[1], int(sys.argv[2])
if ARM not in ("map", "bayes"):
    raise SystemExit(f"arm must be 'map' or 'bayes' (got {ARM!r})")
DEVICE = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("DEVICE", "auto")
OUT = os.environ.get("OUT", "runs_offsets")
cfg = yaml.safe_load(open("configs/map_vs_bayes_offsets.yaml"))
f, w = cfg["fit"], cfg["warm"]

spec = spec_from_config(cfg, 0.0)          # null only: the interval is what is at issue
coh = simulate_cohort(spec, SEED)
Y, animal, drug, inj = coh["y"], coh["animal"], coh["drug"], coh["inj_t"]
S, T, dt = spec.n_sessions, spec.T, spec.dt

kern = build_kernel(dict(spec.kernel), spec.kernel_length, dt)
mf = torch.stack([
    torch.as_tensor(matched_filter_score(Y[s:s + 1], kern, inj[s], dt=dt),
                    dtype=torch.float32).reshape(-1) for s in range(S)])
tonic = stack_tonic_baselines(
    [fit_tonic_baseline(Y[s:s + 1], inj[s], dt=dt) for s in range(S)])

em = AdditiveEmission(sigma_y=spec.sigma_y)
em.log_sigma_y.requires_grad_(False)
model = HierarchicalModel(
    PhasicEventProcess(kern, beta0=spec.beta0_mean, beta_inj=0.0, mu_a=spec.mu_a,
                       sigma_a=spec.sigma_a, dt=dt, history=None),
    tonic, em, animal=animal, drug=drug, inj_t=inj,
    beta0_anchor=spec.beta0_mean, tau=f["tau"],
    bayesian_offsets=(ARM == "bayes"),
    event_guide=WindowedCategoricalEventGuide(
        mf, window=f["window"], thresh=w["thresh"], gain=w["gain"]),
    couple_events=False)
model.train(False)
device = resolve_device(DEVICE, batch=S)
model, Y = prepare(model, Y, device)

pyro.clear_param_store()
pyro.set_rng_seed(SEED)
svi = SVI(model.model, model.guide, ClippedAdam({"lr": f["lr"]}),
          Trace_ELBO(max_plate_nesting=2))
plateau = EstimandPlateau(tol=f["stop_tol"], window=f["stop_window"],
                          min_steps=f["stop_min_steps"])
stopped_at, trace = None, []
for step in range(1, f["max_steps"] + 1):
    svi.step(Y)
    if step % f["stop_probe"] == 0:
        lv = [model.drug_posterior()[d][0] for d in range(spec.n_drugs)]
        trace.append({"step": step, "levels": lv,
                      "contrast": model.drug_contrast(OPIOID2, OPIOID1)[0]})
        if plateau.update(step, [lv[OPIOID1], lv[OPIOID2]]):
            stopped_at = step
            break

loc, scale = model.drug_contrast(OPIOID2, OPIOID1)
out = dict(arm=ARM, seed=SEED, device=str(device), stopped_at=stopped_at,
           estimand_loc=loc, estimand_scale=scale, probe_trace=trace,
           variance_components=model.variance_components(),
           n_sessions=S, T=T)
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(f"{OUT}/{ARM}_{SEED:03d}.json", "w"))
print(f"{ARM:>5} s{SEED:03d} | stopped {stopped_at} | "
      f"est {loc:+.4f} +- {scale:.4f}", flush=True)
assert math.isfinite(loc)
