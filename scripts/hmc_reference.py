#!/usr/bin/env python3
"""Is the variational posterior narrower than the exact one, on the same data?

    python scripts/hmc_reference.py <seed> [--T 500] [--samples 400]

The calibration failure we measured -- the estimate varying about three times more
across replicates than the posterior claims -- has several possible causes, and the
measurement separates none of them. This is the reference that settles one: NUTS on
the same model and the same data gives an asymptotically exact posterior, so the two
widths can be compared directly instead of a sampling spread being compared to a
posterior width.

Three things make the comparison honest, and each of them is easy to get wrong:

  * **The model must not carry the analytic-KL surgery.** `SingleTraceModel` adds
    `pyro.factor` terms that cancel against matching factors in the guide; they make
    the ELBO an exact Bernoulli divergence, but a sampler run on that model targets a
    density that is not the joint. Both arms here use
    `analytic_s_kl=False, mark_kl="sampled"`, the plain relaxed joint.
  * **The nuisance parameters are frozen in both arms.** In an ordinary fit the tonic
    is MAP-optimised alongside the variational parameters; under NUTS it would simply
    sit at its pre-fit. Freezing it in both makes the two arms target the same
    posterior, over `beta0`, `beta_inj`, and the per-bin events and marks.
  * **The comparison is on one dataset at a time.** Sampling spread does not enter.

This is the single-trace model, not the cohort: NUTS costs a full gradient per
leapfrog step over 2T latents, and the cohort model multiplies that by the number of
sessions. The calibration failure has the same signature in both, so the mechanism is
tested where it can be tested.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import pyro
import torch
import yaml
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
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

ap = argparse.ArgumentParser()
ap.add_argument("seed", type=int)
ap.add_argument("--T", type=int, default=500)
ap.add_argument("--samples", type=int, default=400)
ap.add_argument("--warmup", type=int, default=400)
ap.add_argument("--svi-steps", type=int, default=4000)
ap.add_argument("--max-tree-depth", type=int, default=8)
ap.add_argument("--beta0", type=float, default=None,
                help="override the baseline log-rate; the default is too sparse to "
                     "leave any events in a short recording")
ap.add_argument("--inj-frac", type=float, default=0.5,
                help="injection at this fraction of the recording. Half rather than "
                     "the protocol's quarter: the precision of beta_inj is limited by "
                     "the PRE-injection count, and at a tractable T a quarter leaves "
                     "too few to be informative.")
ap.add_argument("--chain", type=int, default=0,
                help="chain index for this dataset. Independent chains on the SAME "
                     "data are what give a genuine r_hat and pool into a usable "
                     "n_eff; more DATASETS do not, because a badly mixed chain's "
                     "posterior SD is biased rather than merely noisy.")
ap.add_argument("--out", default=os.environ.get("OUT", "runs_hmc"))
a = ap.parse_args()

cfg = yaml.safe_load(open("configs/toy.yaml"))
d, k, gt, ton, f, w = (cfg["data"], cfg["kernel"], cfg["gt"], cfg["tonic"],
                       cfg["fit"], cfg["warm"])
dt, T = 1 / d["fs"], a.T
INJ = int(T * a.inj_frac)
if a.beta0 is not None:
    gt = dict(gt, beta0=a.beta0)

# ---- one simulated recording
kernel = build_kernel(dict(k["fit"]), k["length"], dt)
g = torch.Generator().manual_seed(a.seed)
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


def build():
    """A model with no factor surgery and every nuisance frozen."""
    emission = AdditiveEmission(sigma_y=gt["sigma_y"])
    emission.log_sigma_y.requires_grad_(False)
    tonic = fit_tonic_baseline(y, INJ, dt=dt)
    for prm in tonic.parameters():
        prm.requires_grad_(False)
    phasic = PhasicEventProcess(kernel, beta0=b0a, beta_inj=0.0, mu_a=gt["mu_a"],
                                sigma_a=gt["sigma_a"], dt=dt, history=None)
    m = SingleTraceModel(
        phasic, tonic, emission, tau=f["tau"],
        beta0_prior_loc=b0a, beta_inj_prior_loc=0.0,
        beta_prior=f["beta_prior"],
        beta_low=b0a - f["beta0_halfwidth"], beta_high=b0a + f["beta0_halfwidth"],
        beta_inj_low=-f["beta_inj_halfwidth"], beta_inj_high=f["beta_inj_halfwidth"],
        event_guide=WindowedCategoricalEventGuide(
            torch.as_tensor(mf, dtype=torch.float32), window=f["window"],
            thresh=w["thresh"], gain=w["gain"]),
        analytic_s_kl=False,          # no cancelling factors: the plain relaxed joint
        mark_kl="sampled",
        # PINNED, not defaulted. This script's variational arm is deliberately the
        # FACTORISED family: what it measures is what factorisation costs against an
        # exact reference, and that is the number the paper quotes as the plain
        # relaxed bound. `SingleTraceModel` now defaults to the joint block, so
        # leaving this to the default would silently re-measure a different family
        # under the same label the next time anyone runs it.
        beta_guide="meanfield",
    )
    m.train(False)
    return m

torch.set_num_threads(4)

# ---- variational arm
model = build()
pyro.clear_param_store()
pyro.set_rng_seed(a.seed)
svi = SVI(model.model, model.guide, ClippedAdam({"lr": f["lr"]}),
          Trace_ELBO(max_plate_nesting=2))
t0 = time.time()
for _ in range(a.svi_steps):
    svi.step(y, INJ)
vi_loc, vi_scale = model.beta_posterior()["beta_inj"]
vi_secs = time.time() - t0

# ---- exact arm
model = build()
pyro.clear_param_store()
pyro.set_rng_seed(a.seed * 1000 + a.chain)    # data fixed by `seed`; chain varies here
t0 = time.time()
mcmc = MCMC(NUTS(model.model, max_tree_depth=a.max_tree_depth, jit_compile=False),
            num_samples=a.samples, warmup_steps=a.warmup, disable_progbar=True)
mcmc.run(y, INJ)
post = mcmc.get_samples()["beta_inj"].reshape(-1)
hmc_secs = time.time() - t0
diag = mcmc.diagnostics()["beta_inj"]

out = dict(seed=a.seed, chain=a.chain, T=T, inj_t=INJ, mle=mle,
           n_pre=n_pre, n_post=n_post,
           true_beta_inj=gt["beta_inj"],
           vi_loc=float(vi_loc), vi_scale=float(vi_scale), vi_secs=vi_secs,
           hmc_mean=float(post.mean()), hmc_sd=float(post.std()),
           hmc_q05=float(post.quantile(0.05)), hmc_q95=float(post.quantile(0.95)),
           hmc_secs=hmc_secs,
           n_eff=float(diag["n_eff"]), r_hat=float(diag["r_hat"]),
           ratio=float(post.std()) / float(vi_scale),
           draws=[float(x) for x in post])
os.makedirs(a.out, exist_ok=True)
json.dump(out, open(f"{a.out}/hmc_{a.seed:03d}_c{a.chain}.json", "w"))
print(f"seed {a.seed}.{a.chain} T={T} | VI {vi_loc:+.3f} +- {vi_scale:.4f} ({vi_secs:.0f}s) | "
      f"HMC {out['hmc_mean']:+.3f} +- {out['hmc_sd']:.4f} ({hmc_secs:.0f}s) | "
      f"HMC/VI width {out['ratio']:.2f}x | n_eff {out['n_eff']:.0f} "
      f"r_hat {out['r_hat']:.3f}", flush=True)
