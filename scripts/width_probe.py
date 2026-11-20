"""Does the reported interval width converge by the time the fit stops?

The stopping rule watches the two opioid LEVELS -- chosen because the contrast alone
fires at the floor and all three levels never settle -- and nothing in it looks at
`q_global_tril`, which is what the interval is read from. If the width is still
climbing when the levels plateau, every width in the study was measured at an
arbitrary point on a trajectory, and the scatter we have been attributing to cohorts
is partly an artefact of where each fit happened to stop.
"""
import json
import os
import sys
import time

import pyro
import torch
import yaml
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from photosvi.experiments.cohort import (
    OPIOID1,
    OPIOID2,
    build_cohort_model,
    simulate_cohort,
    spec_from_config,
)
from photosvi.inference.points import EstimandPlateau, prepare, resolve_device

SEED = int(sys.argv[1]); GUIDE = sys.argv[2]; DEV = sys.argv[3]
cfg = yaml.safe_load(open("configs/cohort_comparison.yaml")); f = cfg["fit"]
spec = spec_from_config(cfg, 0.0); coh = simulate_cohort(spec, SEED)
torch.set_num_threads(2)
model, Y, _ = build_cohort_model(cfg, spec, coh, guide=GUIDE)
model, Y = prepare(model, Y, resolve_device(DEV, batch=spec.n_sessions))
pyro.clear_param_store(); pyro.set_rng_seed(SEED)
svi = SVI(model.model, model.guide, ClippedAdam({"lr": f["lr"]}),
          Trace_ELBO(max_plate_nesting=2))
plateau = EstimandPlateau(tol=f["stop_tol"], window=f["stop_window"],
                          min_steps=f["stop_min_steps"])
trace, stopped, t0 = [], None, time.time()
# Run to the ceiling regardless, recording where the rule WOULD have stopped.
for step in range(1, f["max_steps"] + 1):
    svi.step(Y)
    if step % f["stop_probe"] == 0:
        loc, sd = model.drug_contrast(OPIOID2, OPIOID1)
        lv = [model.drug_posterior()[d][0] for d in range(spec.n_drugs)]
        trace.append(dict(step=step, loc=loc, sd=sd,
                          sigma_animal=model.variance_components().get("sigma_animal")))
        if stopped is None and plateau.update(step, [lv[OPIOID1], lv[OPIOID2]]):
            stopped = step
        print(f"{GUIDE} s{SEED} step {step:>6} | sd {sd:.4f} | loc {loc:+.4f} | "
              f"sig_an {trace[-1]['sigma_animal']:.3f}"
              f"{'   <- rule would stop here' if stopped == step else ''}", flush=True)
json.dump(dict(seed=SEED, guide=GUIDE, stopped_at=stopped, trace=trace,
               secs=time.time() - t0),
          open(f"{os.environ.get('PHOTOSVI_RUNS', 'runs')}/width_probe_{GUIDE}_{SEED}.json", "w"))
