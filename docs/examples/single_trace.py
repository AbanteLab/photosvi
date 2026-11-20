"""Fit one simulated recording and read the injection effect off the posterior.

Runnable as-is; it is executed by the test suite on every push, so if the API moves
this example fails rather than quietly becoming wrong. It uses `configs/toy.yaml`,
which is two minutes of recording instead of forty so that it finishes in seconds --
the numbers are not the paper's.
"""

import math

import pyro
import torch
import yaml
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from photosvi.api import config_path
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

cfg = yaml.safe_load(config_path("toy.yaml").read_text())
d, k, gt, ton, f, w = (cfg["data"], cfg["kernel"], cfg["gt"], cfg["tonic"],
                       cfg["fit"], cfg["warm"])
dt, T, INJ = 1 / d["fs"], d["T"], d["inj_t"]

# --- 1. a recording. Replace this block with your own trace and injection bin.
kernel = build_kernel(dict(k["fit"]), k["length"], dt)
g = torch.Generator().manual_seed(0)
x_true = TonicBaseline(dt=dt, **ton).baseline(T, INJ)
ex = PhasicEventProcess(kernel, beta0=gt["beta0"], beta_inj=gt["beta_inj"],
                        mu_a=gt["mu_a"], sigma_a=gt["sigma_a"], dt=dt,
                        history=None).sample(T, INJ, 1, g)
y = AdditiveEmission(sigma_y=gt["sigma_y"]).sample(x_true + ex["c"], generator=g)["y"]

# --- 2. two data-driven anchors: where the baseline rate is, and where the tonic is.
mf = matched_filter_score(y, kernel, INJ, dt=dt)
beta0_anchor, _ = mf_anchor(mf, INJ, dt, thresh=cfg["detection"]["mf_thresh"],
                            refractory=cfg["detection"]["refractory"])
tonic = fit_tonic_baseline(y, INJ, dt=dt)

# --- 3. assemble the model. The emission scale is estimated from the trace and held
#        fixed, so it cannot inflate to absorb tonic misfit.
emission = AdditiveEmission(sigma_y=gt["sigma_y"])
emission.log_sigma_y.requires_grad_(False)
model = SingleTraceModel(
    PhasicEventProcess(kernel, beta0=beta0_anchor, beta_inj=0.0, mu_a=gt["mu_a"],
                       sigma_a=gt["sigma_a"], dt=dt, history=None),
    tonic, emission, tau=f["tau"],
    beta0_prior_loc=beta0_anchor, beta_inj_prior_loc=0.0,
    beta_prior=f["beta_prior"],
    beta_low=beta0_anchor - f["beta0_halfwidth"],
    beta_high=beta0_anchor + f["beta0_halfwidth"],
    beta_inj_low=-f["beta_inj_halfwidth"], beta_inj_high=f["beta_inj_halfwidth"],
    event_guide=WindowedCategoricalEventGuide(
        torch.as_tensor(mf, dtype=torch.float32), window=f["window"],
        thresh=w["thresh"], gain=w["gain"]),
)
model.train(False)

# --- 4. fit.
pyro.clear_param_store()
pyro.set_rng_seed(0)
svi = SVI(model.model, model.guide, ClippedAdam({"lr": f["lr"]}),
          Trace_ELBO(max_plate_nesting=2))
for _ in range(f["steps"]):
    svi.step(y, INJ)

# --- 5. read the two outputs off the one posterior.
beta_inj, scale = model.beta_posterior()["beta_inj"]
q = model.event_posterior(y)[0]          # per-bin P(event), length T

print(f"rate ratio      {math.exp(beta_inj):.2f}x   (simulated {math.exp(gt['beta_inj']):.2f}x)")
print(f"beta_inj        {beta_inj:+.3f} +- {scale:.3f}")
# Thresholding q at 0.5 counts every bin of a transient's correlation lobe, so it
# over-counts; the guide's total mass is the event-count estimate, and
# `photosvi.benchmark.nms` is what turns q into discrete calls.
print(f"expected events {float(q.sum()):.0f}   (simulated {int(ex['s'].sum())})")

assert math.isfinite(beta_inj)
assert q.shape == (T,)
