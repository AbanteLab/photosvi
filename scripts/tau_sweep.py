#!/usr/bin/env python3
"""One cell of the relaxation-temperature sweep.

    python scripts/tau_sweep.py <tau> <kl_mode> <seed>
        kl_mode: "analytic" (the shipped configuration) or "relaxed"

The events are discrete, and the model's central claim is about counting them. To get
a reparameterised gradient they are relaxed with a binary-Concrete at temperature
`tau`, which makes each event a continuous value in (0, 1): the convolution then
mixes fractional events, and `tau` controls how far the thing being fit is from the
thing being claimed. Every result reported so far uses tau = 0.5, which was chosen
rather than measured -- this sweep is what turns it into a measured choice.

The second axis is how the event factor is scored. Under `analytic` the relaxed
densities are cancelled by paired factors in the model and the guide, and an exact
Bernoulli divergence is charged in their place, so the relaxation survives only on the
reconstruction path. Under `relaxed` the bound is the ordinary one. The prediction
worth testing is that `analytic` is the less tau-sensitive of the two, because it
removes the relaxation from the term through which the data reach the coefficients.
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

from photosvi.benchmark import matched_filter_score, run_benchmark
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

TAU, KL, SEED = float(sys.argv[1]), sys.argv[2], int(sys.argv[3])
if KL not in ("analytic", "relaxed"):
    raise SystemExit(f"kl_mode must be 'analytic' or 'relaxed' (got {KL!r})")
OUT = os.environ.get("OUT", "runs_tau")

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
onsets = ex["s"][0].nonzero().reshape(-1).numpy()
npre, npost = int((onsets < INJ).sum()), int((onsets >= INJ).sum())
mle = (math.log((npost / ((T - INJ) * dt)) / (npre / (INJ * dt)))
       if npre and npost else float("nan"))

k_fit = build_kernel(dict(k["fit"]), k["length"], dt)
mf = matched_filter_score(y, k_fit, INJ, dt=dt)
b0a, _ = mf_anchor(mf, INJ, dt, thresh=det["mf_thresh"], refractory=det["refractory"])
setup = setup_from_config(cfg, b0a)

torch.set_num_threads(1)
pyro.set_rng_seed(SEED)
em = AdditiveEmission(sigma_y=setup.sigma_y)
em.log_sigma_y.requires_grad_(False)
model = SingleTraceModel(
    PhasicEventProcess(setup.kernel(), beta0=b0a, beta_inj=0.0, mu_a=setup.mu_a,
                       sigma_a=setup.sigma_a, dt=dt, history=None),
    fit_tonic_baseline(y, INJ, dt=dt), em,
    tau=TAU,                                        # the swept axis
    beta0_prior_loc=b0a, beta_inj_prior_loc=0.0,
    event_guide=WindowedCategoricalEventGuide(
        torch.as_tensor(mf, dtype=torch.float32), window=setup.window,
        thresh=setup.warm["thresh"], gain=setup.warm["gain"]),
    couple_events=False,
    analytic_s_kl=(KL == "analytic"),
    mark_kl=("gated" if KL == "analytic" else "sampled"),
    **setup.beta_prior_kwargs(),
)
model.train(False)
pyro.clear_param_store()
pyro.set_rng_seed(SEED)
svi = SVI(model.model, model.guide, ClippedAdam({"lr": f["lr"]}),
          Trace_ELBO(max_plate_nesting=2))
plateau = EstimandPlateau(tol=f["stop_tol"], window=f["stop_window"],
                          min_steps=f["stop_min_steps"])
stopped_at, diverged = None, False
for step in range(1, f["max_steps"] + 1):
    loss = svi.step(y, INJ)
    if not math.isfinite(loss):
        diverged = True
        break
    if step % f["stop_probe"] == 0 and plateau.update(
            step, float(model.beta_posterior()["beta_inj"][0])):
        stopped_at = step
        break

bp = model.beta_posterior()
q = model.event_posterior(y)[0].detach().numpy()
res = run_benchmark(y, onsets, k_fit, inj_t=INJ, fs=d["fs"], tol_s=det["tol_s"],
                    extra={"photosvi": q})
m = res.outcomes["photosvi"].metrics
out = dict(tau=TAU, kl=KL, seed=SEED, diverged=diverged, stopped_at=stopped_at,
           plateau=plateau.plateau_means, beta_inj=bp["beta_inj"][0],
           beta_inj_scale=bp["beta_inj"][1], beta0=bp["beta0"][0], mle=mle,
           true_beta_inj=gt["beta_inj"], n_pre=npre, n_post=npost,
           q_mass=float(q.sum()), n_true=len(onsets),
           best_f1=m.get("best_f1"), pr_auc=m.get("pr_auc"),
           recall_at=m.get("recall_at"))
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(f"{OUT}/tau_{TAU:.3f}_{KL}_{SEED:02d}.json", "w"))
print(f"tau {TAU:.3f} {KL:>8} s{SEED:02d} | b_inj {out['beta_inj']:+.3f} "
      f"(MLE {mle:+.3f}) | mass {out['q_mass']:.0f}/{len(onsets)} | "
      f"F1 {m.get('best_f1', float('nan')):.3f} | stopped {stopped_at}"
      f"{'  DIVERGED' if diverged else ''}", flush=True)
