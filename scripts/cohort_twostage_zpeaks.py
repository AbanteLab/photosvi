#!/usr/bin/env python3
"""The two-stage arm on z-peaks counts: the Bonaventura lab's detector, counts tested.

    python scripts/cohort_twostage_zpeaks.py <delta> <seed>

Writes $OUT/zp_<delta>_<seed>_<session>.json, one per session, on the same cohorts the
other arms fit (regenerated from `(spec, seed)`), exactly as the GuPPy arm does
(scripts/cohort_twostage_guppy.py) but with `photosvi.benchmark.zpeaks_detect`: the
prominence rule of BonaventuraLab/fiber-photometry (GREEN_ISO_awake_inj.m) on the trace
z-scored against its own pre-injection segment, at that pipeline's defaults. Added
the rule behind the windowed summaries of the lab's earlier opioid work.

The same non-maximum suppression as every other detector, so a transient is counted
once; ZPEAKS_NMS=0 gives the raw counts. No fitting, seconds per cohort.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import yaml

from photosvi.benchmark import nms, zpeaks_detect
from photosvi.experiments.cohort import (
    apply_cohort_env,
    simulate_cohort,
    spec_from_config,
)

DELTA, SEED = float(sys.argv[1]), int(sys.argv[2])
OUT = os.environ.get("OUT", "runs_cohort_zpeaks")
cfg = apply_cohort_env(
    yaml.safe_load(open(os.environ.get("CONFIG", "configs/cohort_comparison.yaml"))))
spec = spec_from_config(cfg, DELTA)
os.makedirs(OUT, exist_ok=True)

todo = [s for s in range(spec.n_sessions)
        if not os.path.exists(f"{OUT}/zp_{DELTA:.2f}_{SEED:04d}_{s:02d}.json")]
if not todo:
    print(f"zp d={DELTA:.2f} s{SEED:04d} | all {spec.n_sessions} sessions already done")
    raise SystemExit(0)

coh = simulate_cohort(spec, SEED)
dt = spec.dt

for SESSION in todo:
    y = coh["y"][SESSION].numpy()
    INJ = coh["inj_t"][SESSION]
    T = y.size
    times, scores = zpeaks_detect(y, INJ, fs=1.0 / dt)
    if os.environ.get("ZPEAKS_NMS", "1") != "0":
        times, scores = nms(times, scores, refractory=int(cfg["detection"]["refractory"]))
    times = np.asarray(times)
    n_pre = int((times < INJ).sum())
    n_post = int((times >= INJ).sum())
    mle = (math.log((n_post / ((T - INJ) * dt)) / (n_pre / (INJ * dt)))
           if n_pre and n_post else None)
    out = dict(
        arm="two-stage-zpeaks", delta=DELTA, seed=SEED, session=SESSION,
        animal=coh["animal"][SESSION], drug=coh["drug"][SESSION], inj_t=INJ,
        n_pre=n_pre, n_post=n_post, beta_inj=mle,
        n_pre_true=coh["n_pre"][SESSION], n_post_true=coh["n_post"][SESSION],
        beta_inj_true=coh["beta_inj_true"][SESSION], mle_true=coh["mle"][SESSION])
    json.dump(out, open(f"{OUT}/zp_{DELTA:.2f}_{SEED:04d}_{SESSION:02d}.json", "w"))
    print(f"zp d={DELTA:.2f} s{SEED:04d} sess{SESSION:02d} "
          f"(a{out['animal']} d{out['drug']}) | z-peaks {n_pre:4d}/{n_post:5d} "
          f"| true {out['n_pre_true']:4d}/{out['n_post_true']:5d} "
          f"| b_inj {'  undefined' if mle is None else f'{mle:+.3f}'} "
          f"(true {out['beta_inj_true']:+.3f})", flush=True)
