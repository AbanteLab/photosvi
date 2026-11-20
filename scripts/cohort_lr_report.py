#!/usr/bin/env python3
"""Coverage of the cohort contrast, before and after the linear-response correction.

    python scripts/cohort_lr_report.py [--runs $DEV/runs_cohort_lr]

Coverage is the whole point: a nominal 90% interval on a null cohort should contain
zero on 90% of cohorts. The block reaches 70%. This reports what the correction does
to that number, per tilt size and refit budget, from fits that produced both scales,
so the comparison is within-cohort and not against a previous run.

The tilt sweep is the diagnostic, not a parameter to tune: linear response is only
meaningful where the response is linear, so agreement ACROSS tilt sizes is what says
the estimate is real. Disagreement means the tilt is either too small to clear the
optimiser's noise or large enough to leave the linear regime, and the number should
not be quoted either way.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.join(os.environ.get("PHOTOSVI_RUNS", "runs"), "runs_cohort_lr"))
ap.add_argument("--level", type=float, default=0.90)
a = ap.parse_args()

rows = [json.load(open(f)) for f in sorted(glob.glob(f"{a.runs}/lr_*.json"))]
rows = [r for r in rows if r["delta"] == 0.0]
if not rows:
    raise SystemExit(f"no cohorts under {a.runs}")

# Two-sided normal quantile for the nominal level, by bisection on the erf -- the
# alternative is a scipy import for one number.
z = 0.0
lo, hi = 0.0, 10.0
for _ in range(200):
    z = (lo + hi) / 2
    if math.erf(z / math.sqrt(2)) < a.level:
        lo = z
    else:
        hi = z


def coverage(scales):
    """Fraction of null cohorts whose interval contains zero, and the n behind it."""
    ok = [abs(loc) <= z * s for loc, s in scales if s == s and s > 0]
    return (sum(ok) / len(ok) if ok else float("nan")), len(ok)


locs = [r["estimand_loc"] for r in rows]
spread = st.pstdev(locs)
base = [(r["estimand_loc"], r["estimand_scale"]) for r in rows]
cb, nb = coverage(base)
med_base = st.median([s for _, s in base])

print(f"{len(rows)} null cohorts, guide={rows[0]['guide']}, nominal {a.level:.0%} (z={z:.3f})")
print(f"  estimator spread across cohorts   sd {spread:.4f}")
print(f"  {'block, uncorrected':<34} sd {med_base:.4f}  "
      f"({spread / med_base:.2f}x too narrow)  coverage {cb:.0%}  n={nb}")

# Grouped by the tilt in SD units, NOT by the absolute tilt: t = frac/SD differs on
# every cohort by construction, so grouping on it puts each cohort in its own bucket
# and every row reads n=1 -- which looks like a result and is an artefact of the key.
by = defaultdict(list)
for r in rows:
    for x in r["results"]:
        by[(x.get("tilt_sds", x["tilt"]), x["refit_steps"])].append(
            (r["estimand_loc"], x["sd_contrast"]))

print()
for key in sorted(by):
    frac, steps = key
    c, n = coverage(by[key])
    sds = [s for _, s in by[key] if s == s and s > 0]
    if not sds:
        print(f"  {'LR frac=' + str(frac) + ' refit=' + str(steps):<34} "
              "no finite variance")
        continue
    med = st.median(sds)
    lost = len(by[key]) - len(sds)
    note = f"  ({lost} non-positive)" if lost else ""
    print(f"  {'LR frac=' + str(frac) + ' refit=' + str(steps):<34} sd {med:.4f}  "
          f"({spread / med:.2f}x)  coverage {c:.0%}  n={n}{note}")

print("\nAgreement across tilt sizes at a fixed refit budget is the linearity check;")
print("agreement across refit budgets at a fixed tilt says the refits converged.")
