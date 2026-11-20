#!/usr/bin/env python3
"""Coverage of the cohort contrast's credible interval, mean-field against full-rank.

    python scripts/cohort_coverage.py --runs runs_cohort --fr runs_cohort_fr

At delta = 0 the true contrast is zero, so coverage of a nominal 1-alpha interval is
just how often the interval contains zero. Reported alongside the two quantities that
explain it: the estimator's spread across cohorts, and the width the posterior claims.

A calibrated interval has coverage at its nominal level AND a width matching the
spread. Both are printed because they can disagree: an interval can cover by being
uselessly wide, or miss by being centred wrong rather than too narrow.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st

Z = {0.90: 1.6449, 0.95: 1.9600}


def load(runs: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(f"{runs}/hier_*.json")):
        r = json.load(open(f))
        if r["delta"] == 0.0 and r.get("estimand_scale", 0) > 0:
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_cohort", help="mean-field cohorts")
    ap.add_argument("--fr", default="runs_cohort_fr", help="full-rank cohorts")
    a = ap.parse_args()

    arms = {"mean-field": load(a.runs), "full-rank": load(a.fr)}
    print(f"{'guide':>12} {'n':>4} | {'spread':>8} {'reported':>9} {'ratio':>6} | "
          + " ".join(f"{int(c*100)}% cov" for c in sorted(Z)))
    for name, rows in arms.items():
        if not rows:
            print(f"{name:>12} {0:>4} | {'--':>8} {'--':>9} {'--':>6} |")
            continue
        est = [r["estimand_loc"] for r in rows]
        sc = [r["estimand_scale"] for r in rows]
        spread, width = st.pstdev(est), st.mean(sc)
        cov = []
        for c in sorted(Z):
            hit = sum(1 for r in rows
                      if abs(r["estimand_loc"]) <= Z[c] * r["estimand_scale"])
            cov.append(f"{hit / len(rows):>6.0%}")
        print(f"{name:>12} {len(rows):>4} | {spread:>8.4f} {width:>9.4f} "
              f"{spread / width:>6.2f} | " + " ".join(cov))

    print("\nnominal coverage is the target; ratio 1.00 means the reported width")
    print("matches the estimator's actual spread across cohorts.")
    for name, rows in arms.items():
        if len(rows) > 4:
            se = st.pstdev([r["estimand_loc"] for r in rows]) / math.sqrt(len(rows))
            print(f"  {name}: mean estimate "
                  f"{st.mean([r['estimand_loc'] for r in rows]):+.4f} +- {se:.4f} (SE)")


if __name__ == "__main__":
    main()
