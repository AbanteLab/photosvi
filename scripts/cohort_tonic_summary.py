#!/usr/bin/env python3
"""The tonic-response benchmark's CPU-light arms on one cohort: FLMM and the lab's z-scored AUC.

    CONFIG=<cell config> OUT=<cell dir> python scripts/cohort_tonic_summary.py <dp> <seed>

Regenerates the cohort exactly as the PhotoSVI arms do (spec_from_config with delta = 0
on the cell's config, simulate_cohort with the seed), bins it per minute around each
injection, and writes tonic_<dp>_<seed>.json with both statistics. See
photosvi.experiments.tonic_benchmark.
"""
from __future__ import annotations

import json
import os
import sys

import yaml

from photosvi.experiments.cohort import apply_cohort_env, simulate_cohort, spec_from_config
from photosvi.experiments.tonic_benchmark import bfp_arms, binned_traces, flmm_arm

DP, SEED = float(sys.argv[1]), int(sys.argv[2])
OUT = os.environ.get("OUT", "runs_tonic")
cfg = apply_cohort_env(yaml.safe_load(open(os.environ["CONFIG"])))
spec = spec_from_config(cfg, 0.0)
coh = simulate_cohort(spec, SEED)
y = coh["y"].numpy()
binned, centres = binned_traces(y, coh["inj_t"], spec.fs)
animal = [int(a) for a in coh["animal"]]
drug = [int(d) for d in coh["drug"]]
out = dict(tonic_dp=DP, seed=SEED, tonic_peak_true=list(spec.tonic_peak),
           bins_min=centres.tolist(),
           **bfp_arms(y, coh["inj_t"], spec.fs, animal, drug))
os.makedirs(OUT, exist_ok=True)
out["flmm"] = flmm_arm(binned, centres, animal, drug,
                       os.path.join(OUT, "flmm_work", f"{DP:.2f}_{SEED:04d}"))
json.dump(out, open(f"{OUT}/tonic_{DP:.2f}_{SEED:04d}.json", "w"))
print(f"tonic dP={DP:.2f} s{SEED:04d} | z-AUC low-pass |t| {out['zlp']['stat']:.2f} "
      f"| z-AUC ALS |t| {out['zals']['stat']:.2f} "
      f"| flmm max|t| {out['flmm']['stat']:.2f} joint {out['flmm']['joint_excludes_zero']}")
