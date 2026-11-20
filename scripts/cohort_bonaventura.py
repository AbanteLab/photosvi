#!/usr/bin/env python3
"""The hierarchical model on the real low-dose block.

    python scripts/cohort_bonaventura.py [device]

Fifteen recordings from eight animals under vehicle, morphine 20 mg/kg and fentanyl
0.01 mg/kg -- the part of the dataset that survives the per-session diagnostic and the
part whose design supports a contrast.

**Why this block and not the whole set.** The high-dose sessions (morphine 40,
fentanyl 0.03) come from five animals that never received vehicle, so nothing in that
block can be referenced to a control and dose is confounded with which cohort an
animal belongs to. All seven of them also failed the per-session diagnostic, placing
1.8 to 4.7 transients in a ten-minute baseline where the matched filter found 65 to
116 -- a collapsed baseline rather than a measurement. Fitting them jointly would
propagate that into the population estimate.

**Why all three conditions and not vehicle plus morphine.** The estimand this project
is built around is opioid against opioid, and morphine-versus-fentanyl is the best
paired contrast the dataset has: four animals received both. Vehicle-versus-morphine
is paired in three and vehicle-versus-fentanyl in two, so dropping fentanyl would
remove the strongest comparison to keep the weaker ones.

**One assumption that is not verifiable from the files.** Neither `allAnimals3.mat`
nor `FPdata_labels.csv` carries a session identifier: the link between a column of
`allMorphs` and a row of the label file is their shared position, and nothing in
either file confirms it. The column and row counts agree for all five conditions,
which is necessary and not sufficient. Since that mapping is exactly what fixes which
animal each recording belongs to, it determines the pairing the contrast rests on --
so it is on the list of things to confirm with the originating lab, and the per-animal
listing this script prints on startup is there to be checked against their records.

The design is partially crossed and the model reads it off the index vectors, exactly
as with the simulated cohorts: it detects animal, animal x drug and per-session
baseline levels, and with one recording per cell the interaction is not separable
from a session residual -- the same caveat the crossed simulations carry.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np
import pyro
import scipy.io as sio
import torch
import yaml
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from scipy.signal import find_peaks

from photosvi.experiments.cohort import build_cohort_model, spec_from_config
from photosvi.inference.points import (
    EstimandPlateau,
    FitQuality,
    block_spread,
    prepare,
    residual_ratio,
    resolve_device,
)

MAT = os.path.join(os.environ.get("PHOTOSVI_DATA", "data"), "bonaventura_lab/photometry/allAnimals3.mat")
LAB = os.path.join(os.environ.get("PHOTOSVI_DATA", "data"), "bonaventura_lab/photometry/FPdata_labels.csv")
OUT = os.environ.get("OUT", "runs_cohort_bonaventura")
DEVICE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DEVICE", "auto")
FS, INJ = 25, 15000
# Vehicle first so it is drug 0, matching the simulated cohorts' convention and so
# `drug_contrast(2, 1)` is fentanyl against morphine in both.
CONDITIONS = [("saline", 0), ("morphine", 20), ("fentanyl", 0.01)]
GROUPS = {("saline", 0): "allSALs", ("morphine", 20): "allMorphs",
          ("fentanyl", 0.01): "allFents"}

cfg = yaml.safe_load(open("configs/cohort_comparison.yaml"))
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------- assemble the cohort
lab = list(csv.DictReader(open(LAB)))
mat = sio.loadmat(MAT, squeeze_me=True)
# The label file lists sessions per condition in the same order as the columns of the
# matching array, which is the only link between the two files; there is no session id
# in either. Asserted rather than assumed.
sessions = []
for drug, dose in CONDITIONS:
    rows = [r for r in lab if r["DRUG"] == drug and float(r["DOSE (mg/kg)"]) == dose]
    arr = np.atleast_2d(mat[GROUPS[(drug, dose)]].T).T
    if arr.shape[1] != len(rows):
        raise SystemExit(
            f"{drug} {dose}: {arr.shape[1]} columns against {len(rows)} label rows")
    for j, r in enumerate(rows):
        sessions.append(dict(animal=r["ANIMAL ID"], drug=drug, dose=dose,
                             y=arr[:, j].astype(np.float32)))

animals = sorted({s["animal"] for s in sessions})
drugs = [f"{d} {q:g}" for d, q in CONDITIONS]
a_idx = [animals.index(s["animal"]) for s in sessions]
d_idx = [drugs.index(f"{s['drug']} {s['dose']:g}") for s in sessions]
Y = torch.as_tensor(np.stack([s["y"] for s in sessions]))
S, T = Y.shape

print(f"{len(animals)} animals, {S} sessions, {T} samples each", flush=True)
for a in animals:
    got = sorted(f"{s['drug']} {s['dose']:g}" for s in sessions if s["animal"] == a)
    print(f"  {a:<8} {got}", flush=True)

# Per-session mark scale, as the single-trace fits use: the shipped value assumes a
# transient tens of times more prominent than these recordings carry, and a cohort
# shares one mark prior, so it has to be anchored on the cohort's own transients.
proms = []
for s in sessions:
    _, p = find_peaks(s["y"][:INJ], prominence=0.01)
    sd = float(s["y"][:INJ].std())
    if p["prominences"].size:
        proms.append(float(np.median(p["prominences"])))
MU_A = math.log(float(np.median(proms)))
SIGMA_Y = float(np.median([s["y"][:INJ].std() for s in sessions]))
print(f"mark scale exp(mu_a) = {math.exp(MU_A):.4f}, sigma_y = {SIGMA_Y:.4f}", flush=True)

# The spec carries the fixed pieces the builder needs; the DATA comes from `coh`.
cfg["cohort"] |= dict(n_animals=len(animals), T=T, inj_t=INJ, inj_jitter=0,
                      sigma_y=SIGMA_Y, mu_a=MU_A)
spec = spec_from_config(cfg, 0.0)
coh = dict(y=Y, animal=a_idx, drug=d_idx, inj_t=[INJ] * S)

model, Yd, _ = build_cohort_model(cfg, spec, coh)
device = resolve_device(DEVICE, batch=S)
model, Yd = prepare(model, Yd, device)
print("design:", json.dumps(model.design_summary()), flush=True)

f = cfg["fit"]
pyro.clear_param_store()
pyro.set_rng_seed(0)
svi = SVI(model.model, model.guide, ClippedAdam({"lr": f["lr"]}),
          Trace_ELBO(max_plate_nesting=2))
plateau = EstimandPlateau(tol=f["stop_tol"], window=f["stop_window"],
                          min_steps=f["stop_min_steps"])
# See cohort_hierarchical.py: the plateau rule watches the estimand and this watches
# whether the model is explaining the recordings at all. On real data that is the
# difference that matters, and it is the one that was missing.
quality = FitQuality()
rho_per: list[float] = []
stopped, trace = None, []
for step in range(1, f["max_steps"] + 1):
    svi.step(Yd)
    if step % f["stop_probe"] == 0:
        lv = [model.drug_posterior()[d][0] for d in range(len(drugs))]
        c21 = model.drug_contrast(2, 1)
        rho, rho_per = residual_ratio(model, Yd)
        quality.update(step, rho)
        trace.append(dict(step=step, levels=lv, fent_minus_morph=c21,
                          residual_ratio=rho))
        print(f"  step {step:6d} | " + "  ".join(
            f"{drugs[d]} {lv[d]:+.3f}" for d in range(len(drugs)))
            + f" | fent-morph {c21[0]:+.3f} +- {c21[1]:.3f}", flush=True)
        if plateau.update(step, [lv[1], lv[2]]):
            stopped = step
            break

post = model.drug_posterior()
out = dict(arm="hierarchical-real", animals=animals, drugs=drugs,
           n_sessions=S, stopped_at=stopped, design=model.design_summary(),
           mu_a=MU_A, sigma_y=SIGMA_Y,
           mu_drug={drugs[d]: post[d] for d in range(len(drugs))},
           contrasts={f"{drugs[i]}-{drugs[j]}": model.drug_contrast(i, j)
                      for i, j in ((1, 0), (2, 0), (2, 1))},
           variance_components=model.variance_components(),
           tonic_scales=model.tonic_scales(),
           tonic_guide_widths=model.tonic_guide_widths(),
           tonic_posterior=model.tonic_posterior(),
           session_beta_inj=[float(x) for x in model.session_beta_inj().reshape(-1)],
           fit_quality=quality.summary(), block_spread=block_spread(),
           residual_ratio_per_session=rho_per,
           probe_trace=trace)
json.dump(out, open(f"{OUT}/cohort.json", "w"), indent=1, default=str)
print("\n" + quality.verdict(), flush=True)
print(f"wrote {OUT}/cohort.json", flush=True)
for k, v in out["contrasts"].items():
    print(f"  {k:<28} {v[0]:+.3f} +- {v[1]:.3f}   (rate ratio {math.exp(v[0]):.2f})")
