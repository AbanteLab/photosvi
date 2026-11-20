#!/usr/bin/env python3
"""Power and estimation of the four arms of the tonic-response benchmark.

    python scripts/cohort_tonic_report.py --runs <root> --json <out.json>

<root> holds one directory per cell, dp<dP> (e.g. dp0.50), each with tonic_*.json (FLMM,
z-scored AUC), ts_0.00_<seed>_<session>.json (two-stage) and hier_0.00_<seed>.json
(hierarchical). Thresholds are calibrated on the dP = 0 cohorts as in cohort_report.py; the
two PhotoSVI estimates are of log(1 + dP).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

from photosvi.experiments.tonic_benchmark import ARMS, LABELS, hier_arm, twostage_arm

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--json", default=None)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--n-sessions", type=int, default=10)
ap.add_argument("--min-null", type=int, default=10,
                help="fewest null cohorts a threshold may rest on (1 for a smoke test)")
a = ap.parse_args()

rows = []
for cell in sorted(glob.glob(os.path.join(a.runs, "dp*"))):
    dp = float(os.path.basename(cell)[2:])
    by_seed: dict[int, dict] = defaultdict(lambda: {"dp": dp})
    for f in glob.glob(f"{cell}/tonic_*.json"):
        r = json.load(open(f))
        row = by_seed[r["seed"]]
        row["seed"] = r["seed"]
        row["flmm_t"] = r["flmm"]["stat"]
        row["flmm_joint"] = r["flmm"]["joint_excludes_zero"]
        for v in ("zlp", "zals"):
            row[f"{v}_t"] = r[v]["stat"]
    ts = defaultdict(list)
    for f in glob.glob(f"{cell}/ts_*.json"):
        r = json.load(open(f))
        if "tonic_peak" in r:
            ts[r["seed"]].append(r)
    for seed, rs in ts.items():
        if len(rs) == a.n_sessions:
            t = twostage_arm(rs)
            by_seed[seed].update(seed=seed, twostage_t=t["stat"], twostage_est=t["est"])
    for f in glob.glob(f"{cell}/hier_*.json"):
        r = json.load(open(f))
        if r.get("tonic_alpha_q"):
            h = hier_arm(r)
            by_seed[r["seed"]].update(seed=r["seed"], hier_t=h["stat"], hier_est=h["est"],
                                      hier_scale=h["scale"])
    rows += list(by_seed.values())


def calibrate(arm: str) -> float:
    null = sorted(r[f"{arm}_t"] for r in rows if r["dp"] == 0.0 and math.isfinite(r.get(f"{arm}_t", math.nan)))
    if len(null) < a.min_null:
        return float("inf")
    k = int(math.ceil((1 - a.alpha) * len(null))) - 1
    return null[min(max(k, 0), len(null) - 1)]


thresh = {arm: calibrate(arm) for arm in ARMS}
dps = sorted({r["dp"] for r in rows})
power, est = {}, {}
print(f"Power at a calibrated {a.alpha:.0%} false-positive rate (cohorts per cell in parentheses)")
for arm in ARMS:
    cells = []
    for dp in dps:
        v = [r[f"{arm}_t"] for r in rows if r["dp"] == dp and f"{arm}_t" in r]
        p = sum(x > thresh[arm] for x in v) / len(v) if v and math.isfinite(thresh[arm]) else math.nan
        power[(arm, dp)] = (p, len(v))
        cells.append(f"{p:5.0%} ({len(v)})" if v and math.isfinite(p) else f"  -- ({len(v)})")
    print(f"  {LABELS[arm]:<28}" + "  ".join(cells))
print("Bias / RMSE of the estimate of log(1 + dP)")
for arm in ("twostage", "hier"):
    cells = []
    for dp in dps:
        e = [r[f"{arm}_est"] for r in rows if r["dp"] == dp and f"{arm}_est" in r]
        truth = math.log1p(dp)
        if e:
            b = st.mean(e) - truth
            m = math.sqrt(st.mean([(x - truth) ** 2 for x in e]))
            est[(arm, dp)] = (b, m, len(e))
            cells.append(f"{b:+.3f}/{m:.3f}")
        else:
            cells.append("--")
    print(f"  {LABELS[arm]:<28}" + "  ".join(cells))
if a.json:
    json.dump({"rows": rows, "thresholds": thresh, "alpha": a.alpha, "dps": dps,
               "power": [{"arm": k[0], "dp": k[1], "power": v[0], "n": v[1]} for k, v in power.items()],
               "estimation": [{"arm": k[0], "dp": k[1], "bias": v[0], "rmse": v[1], "n": v[2]}
                              for k, v in est.items()]},
              open(a.json, "w"), indent=1, default=float)
    print(f"wrote {a.json}")
