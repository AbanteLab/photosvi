#!/usr/bin/env python3
"""Compare the two arms' posterior overconfidence.

    python scripts/offsets_report.py [--runs runs_offsets]

The overconfidence factor is the empirical spread of the estimate across cohorts
divided by the width the posterior claims. At the null the estimand is zero, so the
empirical SD is a direct estimate of the estimator's sampling spread and the
comparison needs no ground-truth beyond that.

Both arms see the same simulated cohorts, seed for seed, so the paired difference is
the cleaner statistic: it removes cohort-to-cohort variation that both arms share.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st


def load(runs: str, arm: str) -> dict[int, dict]:
    out = {}
    for f in sorted(glob.glob(f"{runs}/{arm}_*.json")):
        r = json.load(open(f))
        out[r["seed"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_offsets")
    a = ap.parse_args()

    arms = {n: load(a.runs, n) for n in ("map", "bayes")}
    shared = sorted(set(arms["map"]) & set(arms["bayes"]))
    print(f"cells: map {len(arms['map'])}, bayes {len(arms['bayes'])}, "
          f"paired {len(shared)}\n")

    print(f"{'arm':>6} {'n':>3} | {'emp SD':>8} {'mean scale':>11} {'factor':>7} | "
          f"{'mean est':>9} {'stopped':>8}")
    factors = {}
    for name, rows in arms.items():
        if len(rows) < 3:
            continue
        est = [r["estimand_loc"] for r in rows.values()]
        sc = [r["estimand_scale"] for r in rows.values()]
        sd, ms = st.pstdev(est), st.mean(sc)
        factors[name] = sd / ms
        stopped = [r["stopped_at"] for r in rows.values() if r["stopped_at"]]
        print(f"{name:>6} {len(rows):>3} | {sd:>8.4f} {ms:>11.4f} {sd/ms:>7.2f}x | "
              f"{st.mean(est):>+9.4f} {int(st.median(stopped)) if stopped else 0:>8}")

    # The factor is emp_SD / mean_scale, and an SD from a handful of cohorts has a
    # relative error of about 1/sqrt(2(n-1)) -- 50% at n = 3. Printing a headline
    # "fraction of the excess closed" from that is worse than printing nothing,
    # because it is the line a reader will quote.
    MIN_N = 12
    thin = {n: len(r) for n, r in arms.items() if len(r) < MIN_N}
    if thin:
        print(f"\n[not enough cells for the factor comparison: {thin}, need "
              f"{MIN_N} per arm]")
        print("  An SD from this many cohorts carries a relative error of roughly "
              f"{1 / math.sqrt(2 * (min(thin.values()) - 1)):.0%}.")

    if len(factors) == 2 and shared:
        print(f"\nPaired over the {len(shared)} cohorts both arms completed:")
        d_sc = [arms["bayes"][s]["estimand_scale"] - arms["map"][s]["estimand_scale"]
                for s in shared]
        r_sc = [arms["bayes"][s]["estimand_scale"] / arms["map"][s]["estimand_scale"]
                for s in shared]
        se = st.stdev(d_sc) / math.sqrt(len(d_sc)) if len(d_sc) > 1 else float("nan")
        print(f"  reported scale, bayes - map: {st.mean(d_sc):+.4f} +- {se:.4f} (SE)")
        print(f"  reported scale, bayes / map: {st.mean(r_sc):.2f}x per cohort")
        # Paired point estimates: the only route left if the interval width does
        # not move, since the factor's denominator is then identical between arms.
        d_e = [abs(arms["bayes"][s]["estimand_loc"]) - abs(arms["map"][s]["estimand_loc"])
               for s in shared]
        se_e = st.stdev(d_e) / math.sqrt(len(d_e)) if len(d_e) > 1 else float("nan")
        print(f"  |estimate|, bayes - map:     {st.mean(d_e):+.4f} +- {se_e:.4f} (SE)")

        if thin:
            print("\n  Factor comparison withheld: see above.")
        else:
            print(f"\n  overconfidence factor  map {factors['map']:.2f}x  ->  "
                  f"bayes {factors['bayes']:.2f}x")
            closed = (factors["map"] - factors["bayes"]) / max(factors["map"] - 1.0, 1e-9)
            print(f"  fraction of the excess closed by making the offsets Bayesian: "
                  f"{closed:.0%}")
        print("\n  A factor near 1 means the interval is honest. What is left after "
              "this\n  change is the mean-field factorization's share, and needs a "
              "different remedy.")


if __name__ == "__main__":
    main()
