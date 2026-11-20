#!/usr/bin/env python3
"""Linear-response correction of the cohort contrast's interval.

    python scripts/cohort_linear_response.py <seed> [device]

This is the experiment the two-mode framing rests on and the one that had not been
run. On a single trace, linear response recovers the exact posterior width from a
factorised fit (validated against NUTS in `linear_response.py`), but on a single trace
it is redundant: a 2x2 joint block over the coefficients gets there for two extra
parameters. The cohort is where the block is NOT enough -- widening it to all eight
globals moves coverage of the nominal 90% interval from 50% to 70%, and stops. So the
question here is whether the correction that closes the single-trace gap also closes
what the block leaves open on a cohort.

The estimand is the contrast mu_opioid2 - mu_opioid1, so the tilt is along that
direction in drug space: `{"mu_drug": [0, -t, +t]}`. Adding it to the log joint and
differentiating the fitted mean gives the contrast's variance directly, by the
cumulant-generating identity, without ever forming a covariance matrix over the
~16000 latents.

Both scales come out of the SAME fits -- the block's own `drug_contrast` scale and
the corrected one -- so coverage under each is measured on identical cohorts and no
comparison depends on matching a previous run.

Null cohorts only (delta = 0): coverage is a statement about the null, and a nominal
90% interval should contain zero on 90% of them.
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
GUIDE = os.environ.get("GUIDE", "fullrank")     # the shipped cohort default
DELTA = float(os.environ.get("DELTA", 0.0))
# Tilt size in units of the estimand's own posterior SD, not in absolute units. The
# equilibrium shift under a tilt t is t*Var, so a fixed absolute t means a different
# displacement on every cohort -- and on this estimand (SD ~ 0.05) it means a very
# large t. The first attempt used t = 16 and diverged: a tilt is a CONSTANT gradient
# of magnitude t on mu_drug, and Adam normalises gradient scale, so a persistent
# component that size walks the parameter at nearly the full learning rate for
# thousands of steps before the curvature (1/Var ~ 330) catches up. Sized this way,
# t*Var = frac*SD by construction and the tilt is comparable across cohorts.
TILT_SDS = [float(x) for x in os.environ.get("TILT_SDS", "0.5,1.0").split(",")]
REFITS = [int(x) for x in os.environ.get("REFITS", "6000,15000").split(",")]
# Ramp the tilt in over the first quarter of the refit rather than switching it on.
# Same fixed point, reached by tracking a moving optimum instead of being kicked.
RAMP = float(os.environ.get("RAMP", 0.25))
# The refit is a small displacement of a converged fit, not a fit from scratch.
REFIT_LR_SCALE = float(os.environ.get("REFIT_LR_SCALE", 0.25))
OUT = os.environ.get("OUT", "runs_cohort_lr")

cfg = yaml.safe_load(open("configs/cohort_comparison.yaml"))
f = cfg["fit"]
spec = spec_from_config(cfg, DELTA)
coh = simulate_cohort(spec, SEED)

torch.set_num_threads(2)
model, Y, _ = build_cohort_model(cfg, spec, coh, guide=GUIDE)
device = resolve_device(DEVICE, batch=spec.n_sessions)
model, Y = prepare(model, Y, device)

opt = ClippedAdam({"lr": f["lr"]})
elbo = Trace_ELBO(max_plate_nesting=2)


# ---------------------------------------------------------------- the base fit
# Same stopping rule as the study arm: the two opioid LEVELS, not the contrast and
# not saline. See the note in cohort_hierarchical.py for why both alternatives fail.
pyro.clear_param_store()
pyro.set_rng_seed(SEED)
svi = SVI(model.model, model.guide, opt, elbo)
plateau = EstimandPlateau(tol=f["stop_tol"], window=f["stop_window"],
                          min_steps=f["stop_min_steps"])
t0, stopped_at = time.time(), None
for step in range(1, f["max_steps"] + 1):
    svi.step(Y)
    if step % f["stop_probe"] == 0:
        lv = [model.drug_posterior()[d][0] for d in range(spec.n_drugs)]
        if plateau.update(step, [lv[OPIOID1], lv[OPIOID2]]):
            stopped_at = step
            break
base_secs = time.time() - t0
# get_state, not the items: the store returns CONSTRAINED values, and writing those
# back into a cleared store registers them as unconstrained -- a positive scale comes
# back negative on the first restore.
base = copy.deepcopy(pyro.get_param_store().get_state())

loc, scale_base = model.drug_contrast(OPIOID2, OPIOID1)
contrasts = {f"{DRUG_NAMES[a]}-{DRUG_NAMES[b]}": model.drug_contrast(a, b)
             for a, b in ((OPIOID1, SALINE), (OPIOID2, SALINE), (OPIOID2, OPIOID1))}
print(f"lr s{SEED:04d} | base stopped {stopped_at} in {base_secs / 60:.0f} min | "
      f"contrast {loc:+.4f} +- {scale_base:.4f}", flush=True)


def contrast_mean_after(t, steps):
    """Refit warm-started from the converged fit under a tilt, and read the mean.

    A FRESH optimiser each time: Pyro's ClippedAdam caches moment state per parameter,
    so reusing one would carry momentum from the +t refit into the -t refit, and the
    central difference would quietly absorb it as signal.
    """
    pyro.get_param_store().set_state(copy.deepcopy(base))
    pyro.set_rng_seed(SEED + 991)
    svi_t = SVI(model.model, model.guide,
                ClippedAdam({"lr": f["lr"] * REFIT_LR_SCALE}), elbo)
    ramp = max(1, int(RAMP * steps))
    coeff = [0.0] * spec.n_drugs
    for i in range(steps):
        scale = t * min(1.0, (i + 1) / ramp)
        coeff[OPIOID2], coeff[OPIOID1] = +scale, -scale
        svi_t.step(Y, None, {"mu_drug": list(coeff)})
    return model.drug_contrast(OPIOID2, OPIOID1)[0]


res = []
for steps in REFITS:
    for frac in TILT_SDS:
        t = frac / scale_base            # so that t * Var = frac * SD
        t1 = time.time()
        try:
            # Central difference: the antisymmetric part cancels the refit's own drift.
            plus, minus = contrast_mean_after(+t, steps), contrast_mean_after(-t, steps)
            var = (plus - minus) / (2 * t)
        except (ValueError, RuntimeError) as exc:
            # One diverged refit should cost its own cell, not the base fit that took
            # 20 minutes to reach and that every other row depends on.
            print(f"   frac={frac} refit={steps}: refit diverged ({type(exc).__name__})",
                  flush=True)
            plus = minus = var = float("nan")
        sd = math.sqrt(var) if var == var and var > 0 else float("nan")
        res.append(dict(tilt_sds=frac, tilt=t, refit_steps=steps, mean_plus=plus,
                        mean_minus=minus, var_contrast=var, sd_contrast=sd,
                        secs=time.time() - t1))
        ratio = f"{sd / scale_base:.2f}" if sd == sd else "--"
        print(f"   frac={frac:<4g} (t={t:.1f}) refit={steps:<6d} | LR sd {sd:.4f} | "
              f"block sd {scale_base:.4f} | ratio {ratio} | "
              f"{(time.time() - t1) / 60:.0f} min", flush=True)

pyro.get_param_store().set_state(copy.deepcopy(base))
os.makedirs(OUT, exist_ok=True)
out = dict(arm="cohort_lr", delta=DELTA, seed=SEED, guide=GUIDE, device=str(device),
           design=model.design_summary(), stopped_at=stopped_at, base_secs=base_secs,
           estimand_loc=loc, estimand_scale=scale_base,
           true_contrast=spec.true_contrast(),
           contrasts=contrasts, results=res,
           variance_components=model.variance_components())
json.dump(out, open(f"{OUT}/lr_{DELTA:.2f}_{SEED:04d}.json", "w"))
print(f"wrote {OUT}/lr_{DELTA:.2f}_{SEED:04d}.json", flush=True)
