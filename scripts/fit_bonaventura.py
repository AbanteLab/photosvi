#!/usr/bin/env python3
"""Single-trace fits of the 22 real recordings, as a DIAGNOSTIC.

    python scripts/fit_bonaventura.py [start] [stop]

Not a result. What this establishes is whether the estimand is measurable on these
recordings at all -- how many baseline events each session has, whether the reported
width tracks the Poisson counting limit, and which sessions have no usable
denominator. The cohort analysis is a separate decision that this informs.

Two things are re-anchored per session, and both are forced by the data rather than
chosen. The traces are on an unknown normalised scale (the thesis describing them
says z-scored to baseline, but the baseline has mean 0.010 and sd 0.014, so it is
neither dF/F nor a z-score), and the shipped mark prior assumes dF/F: `mu_a = 1.7666`
means a mean amplitude of 5.85 against a kernel whose peak is 1.0, where the whole
signal here lives inside [-0.06, 0.42]. Left alone the model expects transients four
hundred times the noise and finds none.

  * `sigma_y` comes from the trace, which `photosvi.api` already does.
  * `mu_a` is set so the transient-to-noise ratio matches the value the priors were
    calibrated at. beta_inj is a log RATE ratio and is invariant to the scale of the
    marks, so this choice cannot move the estimand; it moves detection sensitivity,
    which is what it is for.

Run after the matched-filter fix of 2026-09-14. Under the old whole-trace
normalisation five of these sessions had an empty baseline and all five were opioid.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pyro
import scipy.io as sio
import torch
import yaml
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from scipy.signal import find_peaks

from photosvi.benchmark import matched_filter_score
from photosvi.experiments.sweep import mf_anchor, setup_from_config
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
    fit_tonic_baseline,
)

MAT = os.path.join(os.environ.get("PHOTOSVI_DATA", "data"), "bonaventura_lab/photometry/allAnimals3.mat")
LAB = os.path.join(os.environ.get("PHOTOSVI_DATA", "data"), "bonaventura_lab/photometry/FPdata_labels.csv")
OUT = os.environ.get("OUT", "runs_bonaventura")
FS, INJ = 25, 15000                      # both stated in the thesis describing the set
# The transient-to-noise ratio the mark prior is anchored at. MEASURED FROM EACH
# TRACE by default, not taken from the shipped value: `mu_a = 1.7666` means a mean
# amplitude of 5.85 against a kernel peaking at 1.0 and a shipped sigma_y of 0.1062,
# so a ratio of 55 -- while these recordings, scored with the experimenters' own rule
# (prominence 0.01 on dF/F), carry a median transient at 1.5 times the baseline
# standard deviation. Anchoring at 55 expects transients thirty-seven times more
# prominent than exist, finds almost none in the baseline, and returns rate ratios of
# several hundred from denominators of two or three events. Set TNR to override.
TNR = float(os.environ["TNR"]) if os.environ.get("TNR") else None
START = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STOP = int(sys.argv[2]) if len(sys.argv) > 2 else 22

cfg = yaml.safe_load(open("configs/benchmark_guppy.yaml"))
f, det = cfg["fit"], cfg["detection"]
dt = 1.0 / FS
torch.set_num_threads(int(os.environ.get("THREADS", 4)))
os.makedirs(OUT, exist_ok=True)

GROUPS = [("allSALs", "saline", 0), ("allMorphs", "morphine", 20),
          ("allFMorph40", "morphine", 40), ("allFents", "fentanyl", 0.01),
          ("allFent03", "fentanyl", 0.03)]
m = sio.loadmat(MAT, squeeze_me=True)
sessions = []
for key, drug, dose in GROUPS:
    v = np.atleast_2d(m[key].T).T
    for j in range(v.shape[1]):
        sessions.append((f"{key.replace('all','')}_{j}", drug, dose, v[:, j]))

for idx in range(START, min(STOP, len(sessions))):
    name, drug, dose, arr = sessions[idx]
    dest = f"{OUT}/{idx:02d}_{name}.json"
    if os.path.exists(dest):
        print(f"{idx:02d} {name}: done", flush=True)
        continue
    t0 = time.time()
    y = torch.as_tensor(arr, dtype=torch.float32).reshape(1, -1)
    T = y.shape[-1]
    sigma_y = float(y[0, :INJ].std())
    setup = setup_from_config(cfg, -2.85)
    # The prominence of this trace's own baseline transients, by the rule the
    # experimenters use, divided by its noise. Per trace because the baseline sd
    # varies fivefold across these sessions.
    _, _props = find_peaks(arr[:INJ], prominence=0.01)
    tnr = (TNR if TNR is not None
           else (float(np.median(_props["prominences"])) / sigma_y
                 if _props["prominences"].size else 1.5))
    # exp(mu_a) * kernel_peak / sigma_y == tnr, with the shipped kernel peaking at 1
    mu_a = math.log(tnr * sigma_y / float(setup.kernel().max()))
    mf = matched_filter_score(y, setup.kernel(), INJ, dt=dt)
    b0a, _ = mf_anchor(mf, INJ, dt, thresh=det["mf_thresh"], refractory=det["refractory"])
    em = AdditiveEmission(sigma_y=sigma_y)
    em.log_sigma_y.requires_grad_(False)
    kw = setup.beta_prior_kwargs()
    mod = SingleTraceModel(
        PhasicEventProcess(setup.kernel(), beta0=b0a, beta_inj=0.0, mu_a=mu_a,
                           sigma_a=setup.sigma_a, dt=dt, history=None),
        fit_tonic_baseline(y, INJ, dt=dt), em, tau=setup.tau,
        beta0_prior_loc=b0a, beta_inj_prior_loc=0.0,
        event_guide=WindowedCategoricalEventGuide(
            torch.as_tensor(mf, dtype=torch.float32), window=setup.window,
            thresh=setup.warm["thresh"], gain=setup.warm["gain"]),
        couple_events=False, **kw)
    mod.train(False)
    md, yd = prepare(mod, y, resolve_device("cpu", batch=1))
    pyro.clear_param_store(); pyro.set_rng_seed(idx)
    svi = SVI(md.model, md.guide, ClippedAdam({"lr": f["lr"]}), Trace_ELBO(max_plate_nesting=2))
    plateau = EstimandPlateau(tol=f["stop_tol"], window=f["stop_window"],
                              min_steps=f["stop_min_steps"])
    # Watches whether the model is explaining the recording, which the plateau rule
    # does not. These are the fits whose ratios were withdrawn: the estimand settled
    # and was written out while the residual sat at eleven times sigma_y.
    quality = FitQuality()
    stopped = None
    try:
        for step in range(1, f["max_steps"] + 1):
            svi.step(yd, INJ)
            if step % f["stop_probe"] == 0:
                quality.update(step, residual_ratio(md, yd, INJ, draws=2)[0])
                if plateau.update(step, float(md.beta_posterior()["beta_inj"][0])):
                    stopped = step; break
    except (ValueError, RuntimeError) as e:
        json.dump(dict(idx=idx, name=name, drug=drug, dose=dose,
                       diverged=type(e).__name__), open(dest, "w"))
        print(f"{idx:02d} {name}: DIVERGED {type(e).__name__}", flush=True)
        continue
    bp = md.beta_posterior()
    q = md.event_posterior(yd).detach().numpy().reshape(-1)
    npre = float(q[:INJ].sum()); npost = float(q[INJ:].sum())
    floor = math.sqrt(1 / npre + 1 / npost) if npre > 0 and npost > 0 else None
    r = dict(idx=idx, name=name, drug=drug, dose=dose, sigma_y=sigma_y, mu_a=mu_a,
             tnr=tnr,
             beta0_anchor=b0a, beta_inj=bp["beta_inj"][0], beta_inj_sd=bp["beta_inj"][1],
             rate_ratio=math.exp(bp["beta_inj"][0]), beta0=bp["beta0"][0],
             q_mass_pre=npre, q_mass_post=npost, poisson_floor=floor,
             stopped_at=stopped, fit_quality=quality.summary(),
             secs=time.time() - t0)
    json.dump(r, open(dest, "w"), indent=1)
    print(f"{idx:02d} {name:<12} {drug:<9} {dose:<5} | ratio {r['rate_ratio']:6.2f} "
          f"| b_inj {r['beta_inj']:+.3f} +- {r['beta_inj_sd']:.3f} "
          f"| floor {floor if floor is None else round(floor,3)} "
          f"| q pre {npre:6.1f} post {npost:7.1f} | {r['secs']/60:.0f} min", flush=True)
