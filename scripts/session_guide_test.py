"""Does the 2x2 block fix the per-session interval, at the production trace length?

The cohort study's per-session fits ran with `beta_guide="meanfield"` -- the default,
never overridden in cohort_twostage.py -- and their intervals are 3.16x too narrow
against a truth we know, with the error sitting exactly on the Poisson counting limit.
The block was measured against NUTS on T=8000 traces with the injection at halfway;
production sessions are T=60000 with it at 25%, and a median of 34 pre-injection
events against that study's denser regime. Whether the fix transfers is one run.
"""
import json
import math
import os
import sys
import time

import pyro
import torch
import yaml
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from photosvi.benchmark import matched_filter_score
from photosvi.experiments.cohort import simulate_cohort, spec_from_config
from photosvi.experiments.sweep import mf_anchor
from photosvi.models.points import (
    AdditiveEmission,
    PhasicEventProcess,
    SingleTraceModel,
    WindowedCategoricalEventGuide,
    build_kernel,
    fit_tonic_baseline,
)

GUIDE = sys.argv[1]; N = int(sys.argv[2]); DEV = sys.argv[3]
cfg = yaml.safe_load(open("configs/cohort_comparison.yaml"))
f, w = cfg["fit"], cfg["warm"]
spec = spec_from_config(cfg, 0.0)
torch.set_num_threads(2)
out = []
for seed in range(3):
    coh = simulate_cohort(spec, seed)
    for s in range(spec.n_sessions):
        if len(out) >= N: break
        y = coh["y"][s:s+1]; INJ = coh["inj_t"][s]; dt = spec.dt
        kern = build_kernel(dict(spec.kernel), spec.kernel_length, dt)
        mf = matched_filter_score(y, kern, INJ, dt=dt)
        b0a, _ = mf_anchor(mf, INJ, dt, thresh=cfg["detection"]["mf_thresh"],
                           refractory=cfg["detection"]["refractory"])
        em = AdditiveEmission(sigma_y=spec.sigma_y); em.log_sigma_y.requires_grad_(False)
        ton = fit_tonic_baseline(y, INJ, dt=dt)
        for p in ton.parameters(): p.requires_grad_(False)
        m = SingleTraceModel(
            PhasicEventProcess(kern, beta0=b0a, beta_inj=0.0, mu_a=spec.mu_a,
                               sigma_a=spec.sigma_a, dt=dt, history=None),
            ton, em, tau=f["tau"], beta0_prior_loc=b0a, beta_inj_prior_loc=0.0,
            beta_prior="normal", beta0_prior_scale=f.get("beta0_prior_scale", f.get("mu_b0_prior_scale", 2.0)),
            beta_inj_prior_scale=f.get("mu_drug_prior_scale", 1.0),
            event_guide=WindowedCategoricalEventGuide(
                torch.as_tensor(mf, dtype=torch.float32), window=f["window"],
                thresh=w["thresh"], gain=w["gain"]),
            couple_events=False, beta_guide=GUIDE)
        m.train(False); m = m.to(DEV); y = y.to(DEV)
        pyro.clear_param_store(); pyro.set_rng_seed(seed * 100 + s)
        svi = SVI(m.model, m.guide, ClippedAdam({"lr": f["lr"]}),
                  Trace_ELBO(max_plate_nesting=2))
        t0 = time.time()
        for _ in range(f["stop_min_steps"]): svi.step(y, INJ)
        bp = m.beta_posterior()
        true = coh["beta_inj_true"][s]; mle = coh["mle"][s]
        npre, npost = coh["n_pre"][s], coh["n_post"][s]
        poi = math.sqrt(1/max(npre,1) + 1/max(npost,1))
        out.append(dict(seed=seed, s=s, guide=GUIDE, loc=bp["beta_inj"][0],
                        sd=bp["beta_inj"][1], true=true, mle=mle, poisson=poi,
                        n_pre=npre, n_post=npost, secs=time.time()-t0))
        print(f"{GUIDE} s{seed}.{s:02d} | loc {bp['beta_inj'][0]:+.3f} sd {bp['beta_inj'][1]:.4f} "
              f"| true {true:+.3f} | poisson {poi:.4f} | {time.time()-t0:.0f}s", flush=True)
    if len(out) >= N: break
json.dump(out, open(f"{os.environ.get('PHOTOSVI_RUNS', 'runs')}/session_guide_{GUIDE}.json","w"))
err = [o["loc"]-o["true"] for o in out]
import statistics as st

print(f"\n{GUIDE}: n={len(out)} | err sd {st.pstdev(err):.4f} | reported sd {st.median(o['sd'] for o in out):.4f} "
      f"| ratio {st.pstdev(err)/st.median(o['sd'] for o in out):.2f} | poisson {st.median(o['poisson'] for o in out):.4f}")
