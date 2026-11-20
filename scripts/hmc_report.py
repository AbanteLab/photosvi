#!/usr/bin/env python3
"""Pool the HMC chains per dataset and compare the two posteriors honestly.

    python scripts/hmc_report.py --runs runs_hmc_chains [--also runs_hmc_deep]

Independent chains on the SAME data are what make the reference usable: they give a
between-chain R-hat rather than the split-R-hat of a single chain, and they pool into
an effective sample size large enough for the posterior SD to mean something. More
*datasets* do not help, because a badly mixed chain's SD is biased rather than merely
noisy, and averaging biased estimates leaves the bias.

Nothing is reported for a dataset whose chains disagree: a reference that has not
converged is not evidence about the variational posterior, and quoting it would be
worse than quoting nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from collections import defaultdict

R_HAT_MAX = 1.05
N_EFF_MIN = 100


def rhat(chains: list[list[float]]) -> float:
    """Between/within R-hat over chains of equal length (Gelman-Rubin)."""
    m = len(chains)
    n = min(len(c) for c in chains)
    if m < 2 or n < 4:
        return float("nan")
    ch = [c[:n] for c in chains]
    means = [st.mean(c) for c in ch]
    variances = [st.variance(c) for c in ch]
    W = st.mean(variances)
    B = n * st.variance(means) if m > 1 else 0.0
    if W <= 0:
        return float("nan")
    var_hat = ((n - 1) / n) * W + B / n
    return math.sqrt(var_hat / W)


def ess(chains: list[list[float]]) -> float:
    """Effective sample size from the pooled autocorrelation, Geyer-truncated."""
    n = min(len(c) for c in chains)
    m = len(chains)
    ch = [c[:n] for c in chains]
    W = st.mean([st.variance(c) for c in ch])
    if W <= 0:
        return float("nan")
    means = [st.mean(c) for c in ch]
    B = n * st.variance(means) if m > 1 else 0.0
    var_hat = ((n - 1) / n) * W + B / n

    def acov(c, lag):
        mu = st.mean(c)
        return sum((c[i] - mu) * (c[i + lag] - mu) for i in range(n - lag)) / n

    total, t = 0.0, 1
    while t < n - 2:
        rho_t = 1 - (W - st.mean([acov(c, t) for c in ch])) / var_hat
        rho_t1 = 1 - (W - st.mean([acov(c, t + 1) for c in ch])) / var_hat
        if rho_t + rho_t1 < 0:
            break
        total += rho_t + rho_t1
        t += 2
    return m * n / (1 + 2 * total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_hmc_chains")
    ap.add_argument("--also", nargs="*", default=[])
    a = ap.parse_args()

    by_seed = defaultdict(list)
    for d in [a.runs, *a.also]:
        for f in sorted(glob.glob(f"{d}/*.json")):
            r = json.load(open(f))
            if "draws" in r:
                by_seed[r["seed"]].append(r)

    print(f"{'seed':>4} {'chains':>6} {'MLE':>7} | {'VI loc':>8} {'VI sd':>7} | "
          f"{'HMC mean':>9} {'HMC sd':>7} | {'HMC/VI':>7} {'n_eff':>7} {'R-hat':>6}  ok")
    usable = []
    for seed in sorted(by_seed):
        rs = by_seed[seed]
        chains = [r["draws"] for r in rs]
        if len(chains) < 2:
            continue
        pooled = [x for c in chains for x in c]
        rh, ne = rhat(chains), ess(chains)
        vi_sd = st.mean([r["vi_scale"] for r in rs])
        vi_loc = st.mean([r["vi_loc"] for r in rs])
        sd = st.pstdev(pooled)
        ok = rh < R_HAT_MAX and ne > N_EFF_MIN
        usable.append((seed, sd / vi_sd, st.mean(pooled), vi_loc, rs[0]["mle"], ok))
        print(f"{seed:>4} {len(chains):>6} {rs[0]['mle']:>+7.3f} | {vi_loc:>+8.3f} "
              f"{vi_sd:>7.4f} | {st.mean(pooled):>+9.3f} {sd:>7.4f} | {sd/vi_sd:>7.2f} "
              f"{ne:>7.0f} {rh:>6.3f}  {'yes' if ok else 'NO'}")

    good = [u for u in usable if u[5]]
    print(f"\ndatasets meeting R-hat < {R_HAT_MAX} and n_eff > {N_EFF_MIN}: "
          f"{len(good)}/{len(usable)}")
    if not good:
        print("\nNothing is quotable. A reference that has not converged is not")
        print("evidence about the variational posterior -- the ratio below would be")
        print("a statement about the sampler, not about the approximation.")
        return
    r = [u[1] for u in good]
    print(f"\nHMC/VI posterior width: {st.mean(r):.2f}x "
          f"(range {min(r):.2f}-{max(r):.2f}, n = {len(r)})")
    print("location, mean error against the count MLE:")
    print(f"  VI  {st.mean([u[3] - u[4] for u in good]):+.3f}")
    print(f"  HMC {st.mean([u[2] - u[4] for u in good]):+.3f}")


if __name__ == "__main__":
    main()
