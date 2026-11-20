#!/usr/bin/env python3
"""Interval for the drug contrasts by leaving one animal out at a time.

    python scripts/cohort_jackknife.py <seed> [device]

Every other route to the cohort interval has tried to make the posterior right. This
one does not: it estimates the sampling distribution of the estimator directly, which
is the quantity we have been comparing against all along. Delete one animal, refit,
and the spread of what comes back is what the delete-one-cluster jackknife turns into
a variance:

    Var_jack(t) = (A - 1)/A * sum_a ( t_(a) - mean_a t_(a) )^2

with `t_(a)` the estimate from the cohort without animal `a`. The cluster is the
animal rather than the session, because sessions from one animal are not independent
evidence about a drug -- which is the same reason the hierarchy exists.

Three things it has that linear response does not. It corrects **every contrast at
once** rather than the one direction a tilt is aimed at, and the QQ grid says the
opioid-versus-vehicle contrasts are the badly calibrated ones (4.1x and 2.5x against
the estimand's 1.9x). It makes no assumption about the variational family being
right. And it is design-agnostic, which matters for the real cohort, where the design
is partially crossed and only five of thirteen animals have a vehicle session.

What it costs: one warm-started refit per animal. For three contrasts that is eight
refits here against linear response's twelve, and unlike those it needs no tilt to be
sized. What it risks: with eight animals the jackknife variance is itself estimated
from eight numbers, so it is noisy; and a warm-started refit may not travel far enough
from the full-cohort fit, which would understate the spread. Both are measurable --
the first from its own scatter, the second by comparing against cold refits on a few
cells -- and neither is assumed away here.
"""
from __future__ import annotations

import copy
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

from photosvi.experiments.cohort import (
    DRUG_NAMES,
    OPIOID1,
    OPIOID2,
    SALINE,
    build_cohort_model,
    simulate_cohort,
    spec_from_config,
)
from photosvi.inference.points import EstimandPlateau, prepare, resolve_device

SEED = int(sys.argv[1])
DEVICE = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("DEVICE", "auto")
GUIDE = os.environ.get("GUIDE", "fullrank")
DELTA = float(os.environ.get("DELTA", 0.0))
REFIT = int(os.environ.get("REFIT", 6000))
LR_SCALE = float(os.environ.get("REFIT_LR_SCALE", 0.25))
OUT = os.environ.get("OUT", "runs_cohort_jack")
CONTRASTS = ((OPIOID1, SALINE), (OPIOID2, SALINE), (OPIOID2, OPIOID1))

cfg = yaml.safe_load(open("configs/cohort_comparison.yaml"))
# Size overrides, for validating the machinery before spending a GPU-hour on it.
# A jackknife on a four-animal cohort is not a result; it is a check that the mask,
# the refits and the variance formula do what they say.
for key, env in (("n_animals", "N_ANIMALS"), ("T", "T"), ("inj_t", "INJ_T")):
    if os.environ.get(env):
        cfg["cohort"][key] = int(os.environ[env])
f = cfg["fit"]
spec = spec_from_config(cfg, DELTA)
coh = simulate_cohort(spec, SEED)
animal = list(coh["animal"])

torch.set_num_threads(2)
model, Y, _ = build_cohort_model(cfg, spec, coh, guide=GUIDE)
device = resolve_device(DEVICE, batch=spec.n_sessions)
model, Y = prepare(model, Y, device)
elbo = Trace_ELBO(max_plate_nesting=2)


def contrasts_now() -> dict[str, tuple[float, float]]:
    return {f"{DRUG_NAMES[a]}-{DRUG_NAMES[b]}": model.drug_contrast(a, b)
            for a, b in CONTRASTS}


# ---------------------------------------------------------------- the full fit
pyro.clear_param_store()
pyro.set_rng_seed(SEED)
svi = SVI(model.model, model.guide, ClippedAdam({"lr": f["lr"]}), elbo)
plateau = EstimandPlateau(tol=f["stop_tol"], window=f["stop_window"],
                          min_steps=f["stop_min_steps"])
t0, stopped = time.time(), None
for step in range(1, f["max_steps"] + 1):
    svi.step(Y)
    if step % f["stop_probe"] == 0:
        lv = [model.drug_posterior()[d][0] for d in range(spec.n_drugs)]
        if plateau.update(step, [lv[OPIOID1], lv[OPIOID2]]):
            stopped = step
            break
full = contrasts_now()
# get_state, not the items: the store returns CONSTRAINED values, and writing those
# back into a cleared store registers them as unconstrained -- a positive scale comes
# back negative on the first restore.
base = copy.deepcopy(pyro.get_param_store().get_state())
print(f"jack s{SEED:04d} | full fit stopped {stopped} in {(time.time() - t0) / 60:.0f} min"
      f" | {', '.join(f'{k} {v[0]:+.4f}' for k, v in full.items())}", flush=True)


# ------------------------------------------------------------ leave one animal out
def refit_without(a: int) -> dict[str, float]:
    """Warm-started refit with animal `a`'s sessions dropped from the likelihood.

    The sessions stay in the model -- removing them would change the design and the
    identifiable levels with it -- and are masked out of the observation instead, so
    the only thing that changes is which data the fit sees.
    """
    keep = [i for i, an in enumerate(animal) if an != a]
    pyro.get_param_store().set_state(copy.deepcopy(base))
    pyro.set_rng_seed(SEED + 7919 + a)
    svi_a = SVI(model.model, model.guide, ClippedAdam({"lr": f["lr"] * LR_SCALE}), elbo)
    mask = _session_mask(keep)
    for _ in range(REFIT):
        svi_a.step(Y, None, None, mask)
    return {k: v[0] for k, v in contrasts_now().items()}


def _session_mask(keep: list[int]) -> torch.Tensor:
    m = torch.zeros(spec.n_sessions, dtype=torch.bool, device=Y.device)
    m[keep] = True
    return m


n_animals = spec.n_animals
loo: list[dict[str, float]] = []
for a in range(n_animals):
    t1 = time.time()
    loo.append(refit_without(a))
    print(f"   without animal {a}: "
          + " | ".join(f"{k} {v:+.4f}" for k, v in loo[-1].items())
          + f"  ({(time.time() - t1) / 60:.0f} min)", flush=True)

jack: dict[str, dict[str, float]] = {}
for key in full:
    vals = [d[key] for d in loo]
    mean = sum(vals) / len(vals)
    var = (n_animals - 1) / n_animals * sum((v - mean) ** 2 for v in vals)
    jack[key] = dict(sd=math.sqrt(max(var, 0.0)), mean=mean,
                     reported_loc=full[key][0], reported_sd=full[key][1])
    print(f"   {key:>17}: jackknife sd {jack[key]['sd']:.4f} | "
          f"reported {full[key][1]:.4f} | ratio {jack[key]['sd'] / full[key][1]:.2f}",
          flush=True)

pyro.get_param_store().set_state(copy.deepcopy(base))
os.makedirs(OUT, exist_ok=True)
json.dump(dict(arm="cohort_jackknife", seed=SEED, delta=DELTA, guide=GUIDE,
               device=str(device), stopped_at=stopped, refit_steps=REFIT,
               n_animals=n_animals, design=model.design_summary(),
               contrasts={k: list(v) for k, v in full.items()},
               leave_one_out=loo, jackknife=jack,
               true_contrast=spec.true_contrast()),
          open(f"{OUT}/jack_{DELTA:.2f}_{SEED:04d}.json", "w"))
print(f"wrote {OUT}/jack_{DELTA:.2f}_{SEED:04d}.json", flush=True)
