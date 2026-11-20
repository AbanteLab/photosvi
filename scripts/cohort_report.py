#!/usr/bin/env python3
"""Assemble the three arms of the cohort comparison and score them.

    python scripts/cohort_report.py --runs runs_cohort [--json out.json]

The three arms, all reading the same simulated cohorts:

  * **two-stage, session-level** -- Welch's two-sample t on the per-session
    `beta_inj` of the two opioids. This is pseudoreplication in the general case and,
    in this fully crossed design, it also throws away the pairing: it compares 8
    sessions against 8 as if they came from 16 animals. Included because it is what
    gets done.
  * **two-stage, animal-averaged** -- a paired t-test on the within-animal
    differences. This is the correct classical analysis and it is the comparator that
    matters; the hierarchy has to beat this one, not the arm above.
  * **hierarchical** -- the posterior of `mu_drug[op2] - mu_drug[op1]`.

**Power is compared at a matched false-positive rate, not at each arm's nominal
threshold.** An arm that declares differences more freely buys recall with precision,
and a table of raw detection rates cannot say which happened. So each arm's threshold
is calibrated on the `delta = 0` cohorts to a common alpha, and power is read at that
calibrated threshold. The uncalibrated rate is reported too, since the gap between
them is itself the result for the session-level arm.

Note what the two-stage arms cannot use: in the opioid-vs-opioid contrast the saline
sessions drop out entirely, while the hierarchical fit uses them to estimate every
variance component. That is a third of the recordings.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

# Set from --guppy. The GuPPy arm is produced by a different script at a different
# cost, so it usually lives in its own directory rather than beside the fits.
GUPPY_RUNS: str | None = None
# (total seconds, bin width) of a recording, from --config; needed for GuPPy rates.
DURATION = None

SALINE, OPIOID1, OPIOID2 = 0, 1, 2
# The two conditions whose difference is the estimand. In the three-drug study they are
# the two opioids and the vehicle drops out; in the two-condition study the recordings
# actually have, they are the vehicle and the treated arm. Taken from the LAST TWO drug
# indices present in the cohort rather than fixed at (1, 2), which is what the file did
# until 2026-10-05: on a two-condition cell `by_drug[2]` is empty and the report died
# with "mean requires at least one data point" -- on every cell of the new grid.
#
# Last two, not first two, so the three-drug case keeps comparing the opioids to each
# other and not either of them to the vehicle. It matches CohortSpec.true_contrast,
# which indexes the effect vector from the end for the same reason.


def _contrast_pair(drugs) -> tuple[int, int]:
    d = sorted(set(drugs))
    if len(d) < 2:
        raise ValueError(f"a contrast needs two conditions, found {d}")
    return d[-2], d[-1]


def _welch(a: list[float], b: list[float]) -> float:
    """|t| for a two-sample Welch test. Zero if either group is too small."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    return abs(st.mean(b) - st.mean(a)) / se if se > 0 else 0.0


def _paired(diffs: list[float]) -> float:
    """|t| for a paired test on within-animal differences."""
    if len(diffs) < 2:
        return 0.0
    se = math.sqrt(st.variance(diffs) / len(diffs))
    return abs(st.mean(diffs)) / se if se > 0 else 0.0


def _rm_anova_interaction(rates: dict) -> float:
    """|t| for the condition x period interaction of a two-way repeated-measures ANOVA.

    `rates[animal][drug] = (pre, post)` in events per second. With two conditions and two
    periods, both within animal, the interaction F(1, n-1) is the square of a paired t on
    the per-animal difference of differences, (post - pre)_treated - (post - pre)_vehicle,
    so that is what is computed. This is the analysis a lab runs on event rates in their
    natural, linear units, which is what distinguishes it from the paired test on the log
    rate ratio.
    """
    drugs = sorted({d for v in rates.values() for d in v})
    if len(drugs) != 2:
        return 0.0
    lo, hi = drugs
    did = [(v[hi][1] - v[hi][0]) - (v[lo][1] - v[lo][0])
           for v in rates.values() if lo in v and hi in v]
    return _paired(did)


def load(runs: str) -> dict:
    """Group every finished result by (delta, seed). Incomplete cohorts are dropped."""
    hier = {}
    for f in sorted(glob.glob(f"{runs}/hier_*.json")):
        r = json.load(open(f))
        hier[(r["delta"], r["seed"])] = r

    sess = defaultdict(list)
    for f in sorted(glob.glob(f"{runs}/ts_*.json")):
        r = json.load(open(f))
        sess[(r["delta"], r["seed"])].append(r)

    # The GuPPy arm sometimes lives in its own directory, because it is produced by a
    # different script and costs seconds rather than hours; `--guppy` points at it. The
    # grid writes it into the cell beside the other arms, so the two globs can resolve
    # to the SAME files -- and then every session was counted twice, no cohort matched
    # `len(gs) == n_sessions`, and the arm reported "complete 0" with 240 results on
    # disk. Deduplicated by real path rather than by trusting the caller.
    gup = defaultdict(list)
    seen: set[str] = set()
    for f in sorted(glob.glob(f"{runs}/gp_*.json")) + sorted(
            glob.glob(f"{GUPPY_RUNS}/gp_*.json") if GUPPY_RUNS else []):
        real = os.path.realpath(f)
        if real in seen:
            continue
        seen.add(real)
        r = json.load(open(f))
        gup[(r["delta"], r["seed"])].append(r)
    # The z-peaks counts arm (scripts/cohort_twostage_zpeaks.py), beside GuPPy's.
    zp = defaultdict(list)
    for f in sorted(glob.glob(f"{runs}/zp_*.json")):
        r = json.load(open(f))
        zp[(r["delta"], r["seed"])].append(r)
    return {"hier": hier, "sess": dict(sess), "guppy": dict(gup), "zpeaks": dict(zp)}


def _counts_arm(row: dict, gs: list[dict], prefix: str) -> None:
    """The unpaired, paired and RM-ANOVA statistics on a threshold detector's counts."""
    by_drug = defaultdict(list)
    by_animal = defaultdict(dict)
    for r in gs:
        by_drug[r["drug"]].append(r["beta_inj"])
        by_animal[r["animal"]][r["drug"]] = r["beta_inj"]
    lo, hi = _contrast_pair(r["drug"] for r in gs)
    g1, g2 = by_drug[lo], by_drug[hi]
    row[f"{prefix}_session_t"] = _welch(g1, g2)
    row[f"{prefix}_session_est"] = (st.mean(g2) - st.mean(g1)
                                    if g1 and g2 else float("nan"))
    diffs = [v[hi] - v[lo] for v in by_animal.values() if lo in v and hi in v]
    row[f"{prefix}_animal_t"] = _paired(diffs)
    row[f"{prefix}_animal_est"] = st.mean(diffs) if diffs else float("nan")
    if DURATION is not None:
        T_s, dt_ = DURATION
        rates = defaultdict(dict)
        for r in gs:
            t_pre, t_post = r["inj_t"] * dt_, T_s - r["inj_t"] * dt_
            rates[r["animal"]][r["drug"]] = (r["n_pre"] / t_pre, r["n_post"] / t_post)
        row[f"{prefix}_anova_t"] = _rm_anova_interaction(rates)


def statistics_per_cohort(data: dict, n_sessions: int) -> list[dict]:
    """One row per cohort: each arm's test statistic and point estimate."""
    rows = []
    # Every arm contributes cohorts, the GuPPy one included. It was left out of this
    # union until 2026-10-05, so a cell holding only GuPPy results reported "cohorts with
    # any result: 0" with 240 of them on disk -- and the GuPPy arm is the one that
    # finishes first, so it was the arm most likely to be alone in a cell.
    keys = sorted(set(data["hier"]) | set(data["sess"]) | set(data["guppy"])
                  | set(data.get("zpeaks", {})))
    for delta, seed in keys:
        row = {"delta": delta, "seed": seed}

        rs = data["sess"].get((delta, seed), [])
        if len(rs) == n_sessions:          # only complete cohorts
            by_drug = defaultdict(list)
            by_animal = defaultdict(dict)
            for r in rs:
                by_drug[r["drug"]].append(r["beta_inj"])
                by_animal[r["animal"]][r["drug"]] = r["beta_inj"]
            lo, hi = _contrast_pair(r["drug"] for r in rs)
            g1, g2 = by_drug[lo], by_drug[hi]
            row["ts_session_t"] = _welch(g1, g2)
            row["ts_session_est"] = (st.mean(g2) - st.mean(g1)
                                     if g1 and g2 else float("nan"))
            # the correct classical analysis: pair within animal
            diffs = [v[hi] - v[lo] for v in by_animal.values() if lo in v and hi in v]
            row["ts_animal_t"] = _paired(diffs)
            row["ts_animal_est"] = st.mean(diffs) if diffs else float("nan")
            # posterior-mean rates of the single-trace model, before and after
            if all("beta0" in r for r in rs):   # older records carry beta_inj only
                ts_rates = defaultdict(dict)
                for r in rs:
                    ts_rates[r["animal"]][r["drug"]] = (math.exp(r["beta0"]),
                                                        math.exp(r["beta0"] + r["beta_inj"]))
                row["ts_anova_t"] = _rm_anova_interaction(ts_rates)
            row["n_pairs"] = len(diffs)

        # The same two tests on counts a threshold detector produced. This is the
        # arm that measures what is actually done: GuPPy calls transients, the
        # before-and-after counts become a rate ratio, and those are tested. The
        # arm above uses this project's own per-session model, which shares the
        # generative model with the hierarchy and is therefore a comparator that
        # flatters us; the difference between the two is what the generative model
        # buys before any pooling.
        gs = [r for r in data["guppy"].get((delta, seed), [])
              if r["beta_inj"] is not None]
        if len(gs) == n_sessions:
            _counts_arm(row, gs, "gp")
        zs = [r for r in data.get("zpeaks", {}).get((delta, seed), [])
              if r["beta_inj"] is not None]
        if len(zs) == n_sessions:
            _counts_arm(row, zs, "zp")

        h = data["hier"].get((delta, seed))
        if h is not None:
            # Averaged over the last probes (estimand_*_avg), not read at the stopping
            # step: the width moves by about a fifth of itself between probes (2026-10-13).
            loc_ = h.get("estimand_loc_avg", h["estimand_loc"])
            sc_ = h.get("estimand_scale_avg", h["estimand_scale"])
            row["hier_t"] = abs(loc_ / sc_) if sc_ > 0 else 0.0
            row["hier_est"] = loc_
            row["hier_scale"] = sc_
            row["hier_stopped"] = h.get("stopped_at")
        rows.append(row)
    return rows


# WHY THE NULL CELL ALONE MISLEADS, and it does so in this arm's favour.
#
# At delta = 0 the right answer is zero, so an estimator that compresses toward zero
# scores well on both bias and RMSE. GuPPy compresses hard: its per-session estimate
# has a median of -0.273 against a truth of +0.912 and the wrong SIGN on 406 of 480
# sessions, because its threshold is a per-chunk MAD that the drug's own response
# inflates. Most of that cancels from a contrast between two drugs, and what is left
# is a short ruler -- which on the null cell reads as the most accurate arm in the
# table, bias -0.005 and RMSE 0.052 against the hierarchy's +0.001 and 0.106.
#
# A reader shown only delta = 0 would conclude that the best way to estimate an
# opioid contrast is a threshold detector that gets the direction of the effect wrong
# five times in six. The non-null cells are not a refinement of this table; without
# them it argues the opposite of what is true.
ARMS = {"gp_session": "GuPPy counts, unpaired (sessions)",
        "gp_animal": "GuPPy counts, paired (within animal)",
        "gp_anova": "GuPPy counts, RM ANOVA on rates",
        "zp_session": "z-peaks counts, unpaired (sessions)",
        "zp_animal": "z-peaks counts, paired (within animal)",
        "zp_anova": "z-peaks counts, RM ANOVA on rates",
        "ts_session": "two-stage, unpaired (sessions)",
        "ts_animal": "two-stage, paired (within animal)",
        "ts_anova": "two-stage, RM ANOVA on rates",
        "hier": "hierarchical"}


def calibrate(rows: list[dict], arm: str, alpha: float) -> float:
    """The threshold on |t| that makes this arm err on `alpha` of the null cohorts.

    Taken as the (1 - alpha) empirical quantile of the null statistics. Returns inf
    when there are too few nulls to place it, so nothing is reported rather than a
    threshold resting on a handful of cohorts.
    """
    null = sorted(r[f"{arm}_t"] for r in rows
                  if r["delta"] == 0.0 and f"{arm}_t" in r)
    if len(null) < 10:
        return float("inf")
    k = int(math.ceil((1 - alpha) * len(null))) - 1
    return null[min(max(k, 0), len(null) - 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_cohort")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-sessions", type=int, default=24)
    ap.add_argument("--json", default=None)
    ap.add_argument("--config", default=None,
                    help="cohort config, for the recording duration the GuPPy rates need")
    ap.add_argument("--guppy", default=None,
                    help="directory of the GuPPy-count arm, if it is not in --runs")
    a = ap.parse_args()
    global GUPPY_RUNS, DURATION
    GUPPY_RUNS = a.guppy
    if a.config:
        import yaml
        from photosvi.experiments.cohort import spec_from_config
        spec = spec_from_config(yaml.safe_load(open(a.config)), 0.0)
        DURATION = (spec.T * spec.dt, spec.dt)

    rows = statistics_per_cohort(load(a.runs), a.n_sessions)
    deltas = sorted({r["delta"] for r in rows})
    thresh = {arm: calibrate(rows, arm, a.alpha) for arm in ARMS}

    print(f"cohorts with any result: {len(rows)}")
    for arm, label in ARMS.items():
        n = sum(1 for r in rows if f"{arm}_t" in r)
        t = thresh[arm]
        print(f"  {label:<28} complete {n:>4}   calibrated |t| "
              f"{'--' if math.isinf(t) else f'{t:.3f}'}")

    print(f"\nPower at a calibrated {a.alpha:.0%} false-positive rate")
    print(f"{'delta':>7} {'n':>5} | " + " | ".join(f"{lb:>26}" for lb in ARMS.values()))
    for d in deltas:
        sub = [r for r in rows if r["delta"] == d]
        cells = []
        for arm in ARMS:
            have = [r for r in sub if f"{arm}_t" in r]
            if not have or math.isinf(thresh[arm]):
                cells.append(f"{'--':>26}")
                continue
            cal = sum(1 for r in have if r[f"{arm}_t"] > thresh[arm]) / len(have)
            raw = sum(1 for r in have if r[f"{arm}_t"] > 1.96) / len(have)
            cells.append(f"{cal:>10.0%} (raw {raw:>4.0%}) n={len(have):<3}")
        print(f"{d:>7.2f} {len(sub):>5} | " + " | ".join(cells))

    print("\nEstimation of the contrast (bias / RMSE against the true delta)")
    print(f"{'delta':>7} | " + " | ".join(f"{lb:>26}" for lb in ARMS.values()))
    for d in deltas:
        sub = [r for r in rows if r["delta"] == d]
        cells = []
        for arm in ARMS:
            e = [r[f"{arm}_est"] for r in sub if f"{arm}_est" in r]
            if not e:
                cells.append(f"{'--':>26}")
                continue
            bias = st.mean(e) - d
            rmse = math.sqrt(st.mean([(x - d) ** 2 for x in e]))
            cells.append(f"{bias:>+11.3f} / {rmse:<12.3f}")
        print(f"{d:>7.2f} | " + " | ".join(cells))

    # Coverage: only the hierarchical arm reports an interval for the contrast.
    print("\nCoverage of the hierarchical 90% interval")
    for d in deltas:
        sub = [r for r in rows if r["delta"] == d and "hier_scale" in r]
        if not sub:
            continue
        hit = sum(1 for r in sub
                  if abs(r["hier_est"] - d) <= 1.645 * r["hier_scale"])
        print(f"  delta {d:.2f}: {hit}/{len(sub)} = {hit/len(sub):.0%}  "
              f"(mean reported scale {st.mean([r['hier_scale'] for r in sub]):.3f}, "
              f"empirical SD {st.pstdev([r['hier_est'] for r in sub]):.3f})")

    if a.json:
        json.dump({"rows": rows, "thresholds": thresh, "alpha": a.alpha},
                  open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
