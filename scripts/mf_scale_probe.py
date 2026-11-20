#!/usr/bin/env python3
"""Does the matched filter's normalisation change the answer, or only the warm start?

    python scripts/mf_scale_probe.py [n_seeds]

`matched_filter_score` standardized by the whole trace until 2026-09-14, putting the
drug's own response into the denominator and suppressing baseline detections in
proportion to the effect being measured. The default is now the
pre-injection segment. Every result committed before that date used the old scaling
and none has been recomputed, so the question this answers is narrow and specific:
**is the stored set wrong, or merely produced from a worse starting point?**

The matched filter is not in the likelihood. It feeds two things:

  * `mf_anchor`, the beta0 warm start
  * `WindowedCategoricalEventGuide`, the event guide's per-bin anchor

so a fit can in principle walk away from a poor start. Whether it does is an
empirical question, and it is the only one that decides whether hours of refitting
are owed.

Two stages, cheap first.

  1. **The warm starts themselves**, in seconds. If beta0 and the guide's initial
     event mass barely move, the fits cannot move much either and stage 2 is a
     formality. If they move a lot, stage 2 is mandatory.
  2. **Paired fits**, one per scaling on the same trace with the same seed, scored
     against the simulated truth and against the count MLE -- which is the best any
     method could do on that realisation and therefore the right reference.

Run at low priority: this is a diagnostic, and anything else on the machine matters
more.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pyro
import torch
import yaml
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from photosvi.benchmark import matched_filter_score
from photosvi.experiments.sweep import generate, mf_anchor, setup_from_config
from photosvi.inference.points import EstimandPlateau, prepare, resolve_device
from photosvi.models.points import (
    AdditiveEmission,
    PhasicEventProcess,
    SingleTraceModel,
    WindowedCategoricalEventGuide,
    fit_tonic_baseline,
)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CFG = os.environ.get("CFG", "configs/benchmark_guppy.yaml")
OUT = os.environ.get("OUT", "runs_mf_scale")
SCALES = ("trace", "baseline")

cfg = yaml.safe_load(open(CFG))
d, f, det, gt = cfg["data"], cfg["fit"], cfg["detection"], cfg["gt"]
dt, INJ = 1.0 / d["fs"], int(d["inj_t"])
torch.set_num_threads(int(os.environ.get("THREADS", 2)))
os.makedirs(OUT, exist_ok=True)


def count_mle(onsets: np.ndarray, T: int) -> float:
    """log of (post rate / pre rate) from the realised onsets: the achievable target."""
    pre = int((onsets < INJ).sum())
    post = int((onsets >= INJ).sum())
    if not (pre and post):
        return float("nan")
    return math.log((post / ((T - INJ) * dt)) / (pre / (INJ * dt)))


# --------------------------------------------------- stage 1: the warm starts alone
print("STAGE 1 -- warm starts only, no fitting\n", flush=True)
print(f"{'seed':<6}{'scale':<10}{'beta0 anchor':>14}{'guide mass':>12}"
      f"{'mf>thr pre':>12}{'mf>thr post':>13}", flush=True)
setup = setup_from_config(cfg, gt["beta0"])
stage1 = []
for seed in range(N):
    y, onsets = generate(cfg, seed)
    T = y.shape[-1]
    for sc in SCALES:
        mf = matched_filter_score(y, setup.kernel(), INJ, dt=dt, scale=sc)
        b0a, _ = mf_anchor(mf, INJ, dt, thresh=det["mf_thresh"],
                           refractory=det["refractory"])
        eg = WindowedCategoricalEventGuide(
            torch.as_tensor(mf, dtype=torch.float32), window=setup.window,
            thresh=setup.warm["thresh"], gain=setup.warm["gain"])
        mass = float(torch.sigmoid(eg.fitted_logits(y)).sum()) if hasattr(
            eg, "fitted_logits") else float("nan")
        npre = int((np.asarray(mf)[:INJ] > det["mf_thresh"]).sum())
        npost = int((np.asarray(mf)[INJ:] > det["mf_thresh"]).sum())
        stage1.append(dict(seed=seed, scale=sc, beta0=b0a, mass=mass,
                           n_pre=npre, n_post=npost, n_true=int(onsets.size)))
        print(f"{seed:<6}{sc:<10}{b0a:>14.4f}{mass:>12.1f}{npre:>12}{npost:>13}",
              flush=True)

by = {sc: [r for r in stage1 if r["scale"] == sc] for sc in SCALES}
db0 = [b["beta0"] - a["beta0"] for a, b in zip(by["trace"], by["baseline"], strict=True)]
dmass = [b["mass"] / a["mass"] for a, b in zip(by["trace"], by["baseline"], strict=True)
         if a["mass"]]
print(f"\n  beta0 anchor shift (baseline - trace): "
      f"median {np.median(db0):+.4f}, max |{max(abs(x) for x in db0):.4f}|")
if dmass:
    print(f"  guide initial mass ratio (baseline / trace): median {np.median(dmass):.3f}")
print("  a shift small against the prior scale "
      f"({f['beta0_prior_scale']}) cannot move the fit far.\n", flush=True)

# ------------------------------------------------------ stage 2: the fits themselves
print("STAGE 2 -- paired fits, same trace and seed under each scaling\n", flush=True)
print(f"{'seed':<6}{'scale':<10}{'beta_inj':>10}{'sd':>9}{'true':>9}{'MLE':>9}"
      f"{'err vs MLE':>12}{'min':>7}", flush=True)
rows = []
for seed in range(N):
    y, onsets = generate(cfg, seed)
    T = y.shape[-1]
    mle = count_mle(onsets, T)
    for sc in SCALES:
        t0 = time.time()
        mf = matched_filter_score(y, setup.kernel(), INJ, dt=dt, scale=sc)
        b0a, _ = mf_anchor(mf, INJ, dt, thresh=det["mf_thresh"],
                           refractory=det["refractory"])
        em = AdditiveEmission(sigma_y=setup.sigma_y)
        em.log_sigma_y.requires_grad_(False)
        kw = setup.beta_prior_kwargs()
        m = SingleTraceModel(
            PhasicEventProcess(setup.kernel(), beta0=b0a, beta_inj=0.0,
                               mu_a=setup.mu_a, sigma_a=setup.sigma_a, dt=dt,
                               history=None),
            fit_tonic_baseline(y, INJ, dt=dt), em, tau=setup.tau,
            beta0_prior_loc=b0a, beta_inj_prior_loc=0.0,
            event_guide=WindowedCategoricalEventGuide(
                torch.as_tensor(mf, dtype=torch.float32), window=setup.window,
                thresh=setup.warm["thresh"], gain=setup.warm["gain"]),
            couple_events=False, **kw)
        m.train(False)
        md, yd = prepare(m, y, resolve_device("cpu", batch=1))
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)
        svi = SVI(md.model, md.guide, ClippedAdam({"lr": f["lr"]}),
                  Trace_ELBO(max_plate_nesting=2))
        plateau = EstimandPlateau(tol=f["stop_tol"], window=f["stop_window"],
                                  min_steps=f["stop_min_steps"])
        stopped = None
        for step in range(1, f["max_steps"] + 1):
            svi.step(yd, INJ)
            if step % f["stop_probe"] == 0 and plateau.update(
                    step, float(md.beta_posterior()["beta_inj"][0])):
                stopped = step
                break
        bp = md.beta_posterior()
        r = dict(seed=seed, scale=sc, beta_inj=bp["beta_inj"][0],
                 sd=bp["beta_inj"][1], beta0=bp["beta0"][0], truth=gt["beta_inj"],
                 mle=mle, stopped=stopped, secs=time.time() - t0)
        rows.append(r)
        print(f"{seed:<6}{sc:<10}{r['beta_inj']:>10.4f}{r['sd']:>9.4f}"
              f"{gt['beta_inj']:>9.4f}{mle:>9.4f}{r['beta_inj'] - mle:>+12.4f}"
              f"{r['secs'] / 60:>7.1f}", flush=True)
    json.dump(rows, open(f"{OUT}/mf_scale.json", "w"), indent=1)

print("\nSUMMARY -- error against the count MLE, which is the achievable target")
for sc in SCALES:
    e = [r["beta_inj"] - r["mle"] for r in rows if r["scale"] == sc
         and math.isfinite(r["mle"])]
    s = [r["sd"] for r in rows if r["scale"] == sc]
    if e:
        print(f"  {sc:<10} mean {np.mean(e):+.4f}  sd {np.std(e, ddof=1):.4f}  "
              f"median reported width {np.median(s):.4f}  n={len(e)}")
paired = [b["beta_inj"] - a["beta_inj"]
          for a, b in zip([r for r in rows if r["scale"] == "trace"],
                          [r for r in rows if r["scale"] == "baseline"], strict=True)]
if paired:
    print(f"\n  paired difference (baseline - trace): mean {np.mean(paired):+.4f}, "
          f"max |{max(abs(x) for x in paired):.4f}|")
    print("  if this is small against the reported width, the stored results stand.")
print(f"\nwrote {OUT}/mf_scale.json")
