#!/usr/bin/env python3
"""The two-stage arm as a lab would actually run it: GuPPy detects, counts are tested.

    python scripts/cohort_twostage_guppy.py <delta> <seed>

Writes runs_cohort_guppy/gp_<delta>_<seed>_<session>.json, one per session, on the
same cohorts the other arms fit -- regenerated from `(spec, seed)`, which is
deterministic, so all three arms see identical recordings without sharing a cache.

**Why a third arm.** The existing two-stage comparator fits each recording with this
project's own single-trace model and then tests the per-session estimates. That is a
*favourable* comparator: it shares the generative model with the hierarchy, so the
difference between them isolates the pooling and nothing else. It is not what anyone
does. What gets done is a threshold-based detector, a count of transients before and
after, and a t-test -- and measuring against that is the comparison a reader wants.

With all three the table separates two things it currently conflates:

  * GuPPy -> counts -> test          what is done today
  * our model per session -> test    what the generative model buys, without pooling
  * hierarchical                     what the pooling buys on top

No fitting happens here, so this is seconds per cohort rather than hours. The count
MLE of the log rate ratio is `log((n_post/T_post)/(n_pre/T_pre))`, undefined when
either count is zero -- which is not a rare edge case at the noise levels these
recordings sit at, and is recorded rather than dropped so the rate of it is visible.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import yaml

from photosvi.benchmark import guppy_detect, nms
from photosvi.experiments.cohort import (
    apply_cohort_env,
    simulate_cohort,
    spec_from_config,
)

DELTA, SEED = float(sys.argv[1]), int(sys.argv[2])
OUT = os.environ.get("OUT", "runs_cohort_guppy")
# The config is an environment variable so the three arms cannot be pointed at
# different studies: the grid sets it once and every arm of a cell reads the same
# file, which is the only reason their cohorts are identical.
cfg = apply_cohort_env(
    yaml.safe_load(open(os.environ.get("CONFIG", "configs/cohort_comparison.yaml"))))
spec = spec_from_config(cfg, DELTA)
os.makedirs(OUT, exist_ok=True)

todo = [s for s in range(spec.n_sessions)
        if not os.path.exists(f"{OUT}/gp_{DELTA:.2f}_{SEED:04d}_{s:02d}.json")]
if not todo:
    print(f"gp d={DELTA:.2f} s{SEED:04d} | all {spec.n_sessions} sessions already done")
    raise SystemExit(0)

coh = simulate_cohort(spec, SEED)
dt = spec.dt

for SESSION in todo:
    y = coh["y"][SESSION].numpy()
    INJ = coh["inj_t"][SESSION]
    T = y.size
    # GuPPy's own defaults, untouched. Tuning them on these traces would be scoring a
    # version of the tool nobody runs; the benchmark makes the same choice.
    times, scores = guppy_detect(y, fs=1.0 / dt)
    # The same non-maximum suppression the single-trace benchmark applies to every
    # detector (Table 1), with its refractory period: without it each transient's cluster
    # of local maxima is counted several times. GUPPY_NMS=0 restores the
    # raw counts.
    if os.environ.get("GUPPY_NMS", "1") != "0":
        times, scores = nms(times, scores, refractory=int(cfg["detection"]["refractory"]))
    times = np.asarray(times)
    n_pre = int((times < INJ).sum())
    n_post = int((times >= INJ).sum())
    mle = (math.log((n_post / ((T - INJ) * dt)) / (n_pre / (INJ * dt)))
           if n_pre and n_post else None)
    out = dict(
        arm="two-stage-guppy", delta=DELTA, seed=SEED, session=SESSION,
        animal=coh["animal"][SESSION], drug=coh["drug"][SESSION], inj_t=INJ,
        n_pre=n_pre, n_post=n_post, beta_inj=mle,
        # What a perfect detector would have counted on this realisation, so the gap
        # between the two columns is the detector's contribution and not the noise's.
        n_pre_true=coh["n_pre"][SESSION], n_post_true=coh["n_post"][SESSION],
        beta_inj_true=coh["beta_inj_true"][SESSION], mle_true=coh["mle"][SESSION])
    json.dump(out, open(f"{OUT}/gp_{DELTA:.2f}_{SEED:04d}_{SESSION:02d}.json", "w"))
    print(f"gp d={DELTA:.2f} s{SEED:04d} sess{SESSION:02d} "
          f"(a{out['animal']} d{out['drug']}) | guppy {n_pre:4d}/{n_post:5d} "
          f"| true {out['n_pre_true']:4d}/{out['n_post_true']:5d} "
          f"| b_inj {'  undefined' if mle is None else f'{mle:+.3f}'} "
          f"(true {out['beta_inj_true']:+.3f})", flush=True)
