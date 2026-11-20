#!/usr/bin/env python3
"""Figure 5: what each readout needs, and which mode supplies it.

    python scripts/paper_fig_modes.py --out $DEV/diagnostics/fig_modes.pdf

The method ships in two modes and the split is easy to state wrongly. It is not
"detection versus intervals": detection reads the per-bin event probabilities and never
touches the coefficients' posterior width, and a single-trace interval is already exact
once the coefficients share a joint variational block. What still needs work is the
*cohort* interval.

  A  the posterior on one recording against Hamiltonian Monte Carlo -- the reference
     that makes the rest of the figure a measurement rather than a claim
  B  reported width relative to exact, across configurations, for a single trace
  C  what linear response costs to get there: the correction tracks the refit budget
  D  cohort coverage, which the same joint block does not close
  E  which readout needs which mode
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

INK, MUTED, GRID = "#2b2b2b", "#8a8a8a", "#d9d9d9"
MF, BLOCK, LR = "#c2622d", "#3b6ea5", "#2f8f6f"

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default=os.environ.get("PHOTOSVI_RUNS", "runs"))
ap.add_argument("--out", default=None)
ap.add_argument("--dpi", type=int, default=200)
a = ap.parse_args()
D = a.dev
out = Path(a.out or f"{D}/diagnostics/fig_modes.pdf")

# ------------------------------------------------------------------ the reference
hmc = defaultdict(list)
for f in glob.glob(f"{D}/runs_hmc_chains/*.json") + glob.glob(f"{D}/runs_hmc_deep/*.json"):
    r = json.load(open(f))
    if "draws" in r:
        hmc[r["seed"]].extend(r["draws"])
H = {s: st.pstdev(v) for s, v in hmc.items()}

# Variants are keyed by filename stem minus the seed: the earliest cells predate the
# `guide` and `objective` fields, so reading them off the record drops those cells.
prod = defaultdict(dict)
for f in sorted(glob.glob(f"{D}/runs_hmc_prod/*.json")):
    r = json.load(open(f))
    prod[Path(f).stem.rsplit("_", 1)[0]][r["seed"]] = r

lr = defaultdict(lambda: defaultdict(list))
for d in ("runs_lr", "runs_lr_long"):
    for f in glob.glob(f"{D}/{d}/*.json"):
        r = json.load(open(f))
        for x in r["results"]:
            if x["sd_inj"] == x["sd_inj"]:
                lr[x["refit_steps"]][r["seed"]].append(x["sd_inj"])


def ratio(key):
    """Median over seeds of reported/exact width. 1.0 is honest, below 1 too narrow."""
    ss = [s for s in prod.get(key, {}) if s in H]
    return st.median([prod[key][s]["prod_scale"] / H[s] for s in ss]) if ss else np.nan


fig = plt.figure(figsize=(11.0, 7.0))
gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.92],
                       hspace=0.52, wspace=0.34,
                       left=0.065, right=0.978, top=0.915, bottom=0.075)
axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
axD = fig.add_subplot(gs[1, 0])
axE = fig.add_subplot(gs[1, 1:])
for ax, letter in zip((axA, axB, axC, axD, axE), "ABCDE", strict=True):
    ax.text(-0.16 if ax is not axE else -0.075, 1.14, letter, transform=ax.transAxes,
            fontsize=13, weight="bold", color=INK, va="top")
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.tick_params(labelsize=8.2, colors="#555555")

# ------------------------------------------------- A: the posterior vs the reference
axA.grid(alpha=0.25, lw=0.7, color=GRID)
axA.set_axisbelow(True)
seed = sorted(set(H) & set(prod["prod_fullrank"]) & set(prod["prod_meanfield"]))[0]
draws = np.array(hmc[seed])
axA.hist(draws, bins=30, density=True, color="#dedede", edgecolor="#b0b0b0", lw=0.6,
         label="exact (NUTS)")
grid = np.linspace(draws.min() - 0.45, draws.max() + 0.45, 500)


def gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


for key, col, lab, ls in (("prod_meanfield", MF, "factorised", (0, (5, 2))),
                          ("prod_fullrank", BLOCK, "joint block", "-")):
    r = prod[key][seed]
    axA.plot(grid, gauss(grid, r["prod_loc"], r["prod_scale"]), lw=2.0, color=col,
             ls=ls, label=lab)
axA.set_xlabel(r"$\beta_{\rm inj}$  (one recording)", fontsize=9)
axA.set_ylabel("posterior density", fontsize=9)
axA.set_title("Factorising is what costs the width", fontsize=9.3, color=INK, pad=8)
axA.legend(fontsize=7.6, frameon=False, loc="upper left", handletextpad=0.6,
           borderaxespad=0.2)

# ------------------------------------------------------- B: the single-trace ladder
axB.grid(alpha=0.25, lw=0.7, color=GRID, axis="x")
axB.set_axisbelow(True)
lr_deep = 10000
lr_r = st.median([st.mean(v) / H[s] for s, v in lr[lr_deep].items() if s in H])
bars = [("relaxed bound", ratio("prod_meanfield_elbo-plain"), MF),
        (r"+ R\'enyi ($\alpha$=0, K=4)", ratio("prod_meanfield_renyi0p4-plain"), MF),
        ("analytic KL (shipped)", ratio("prod_meanfield"), MF),
        ("+ joint block", ratio("prod_fullrank"), BLOCK),
        ("+ linear response", lr_r, LR)]
bars = [(n, v, c) for n, v, c in bars if v == v]
pos = np.arange(len(bars))
axB.barh(pos, [v for _, v, _ in bars], color=[c for *_, c in bars], height=0.62,
         alpha=0.88)
axB.axvline(1.0, lw=1.4, color=INK)
axB.text(1.03, 0.15, "exact", fontsize=7.8, color=INK, rotation=90,
         va="bottom")
for i, (_, v, _) in enumerate(bars):
    # A label just outside a bar near 1.0 lands on the "exact" rule; put those inside.
    inside = 0.85 < v < 1.15
    axB.text(v - 0.035 if inside else v + 0.035, i, f"{v:.2f}", va="center",
             ha="right" if inside else "left", fontsize=8.2,
             color="white" if inside else INK)
axB.set_yticks(pos)
axB.set_yticklabels([n.replace(r"\'e", "é") for n, _, _ in bars], fontsize=8.2)
axB.set_xlim(0, 1.32)
axB.set_xlabel("reported width / exact width", fontsize=9)
axB.set_title("Single trace: the block closes it", fontsize=9.3, color=INK, pad=8)

# --------------------------------------------- C: what linear response costs
axC.grid(alpha=0.25, lw=0.7, color=GRID)
axC.set_axisbelow(True)
steps = sorted(lr)
for s in sorted(H):
    ys = [st.mean(lr[k][s]) / H[s] for k in steps if s in lr[k]]
    xs = [k for k in steps if s in lr[k]]
    axC.plot(xs, ys, lw=0.9, color=LR, alpha=0.35, marker="o", ms=2.6)
mean = [st.median([st.mean(v) / H[s] for s, v in lr[k].items() if s in H])
        for k in steps]
axC.plot(steps, mean, lw=2.1, color=LR, marker="o", ms=4.6)
axC.axhline(1.0, lw=1.4, color=INK)
axC.axhline(ratio("prod_fullrank"), lw=1.6, color=BLOCK, ls=(0, (5, 2)))
axC.text(steps[0] * 1.05, ratio("prod_fullrank") + 0.035, "joint block, no refits",
         fontsize=7.5, color=BLOCK)
axC.axhline(ratio("prod_meanfield"), lw=1.2, color=MF, ls=(0, (2, 2)))
axC.text(steps[0] * 1.05, ratio("prod_meanfield") + 0.035, "uncorrected", fontsize=7.5,
         color=MF)
axC.set_xscale("log")
axC.set_xticks(steps)
axC.set_xticklabels([f"{k // 1000}k" for k in steps])
axC.minorticks_off()
axC.set_ylim(0.3, 1.32)
axC.set_xlabel("refit steps per tilt  (× 4 tilts)", fontsize=9)
axC.set_ylabel("reported width / exact width", fontsize=9)
axC.set_title("...and what the correction costs", fontsize=9.3, color=INK, pad=8)

# ------------------------------------------------------------ D: cohort coverage
axD.grid(alpha=0.25, lw=0.7, color=GRID, axis="y")
axD.set_axisbelow(True)
cov = []
for name, sub, col in (("factorised", "runs_cohort", MF),
                       ("block 4×4", "runs_cohort_fr", BLOCK),
                       ("block 8×8", "runs_cohort_fr8", BLOCK)):
    rows = [json.load(open(f)) for f in glob.glob(f"{D}/{sub}/hier_*.json")]
    rows = [r for r in rows if r["delta"] == 0.0 and r.get("estimand_scale", 0) > 0]
    if len(rows) < 5:
        continue
    hit = sum(1 for r in rows if abs(r["estimand_loc"]) <= 1.645 * r["estimand_scale"])
    cov.append((name, hit / len(rows), len(rows), col))
pos = np.arange(len(cov))
axD.bar(pos, [c for _, c, _, _ in cov], color=[c for *_, c in cov], width=0.62,
        alpha=0.88)
axD.axhline(0.90, lw=1.5, color=INK, ls=(0, (4, 3)))
axD.text(len(cov) - 0.42, 0.925, "nominal 90%", fontsize=7.8, color=INK, ha="right")
for i, (_, c, n, _) in enumerate(cov):
    axD.text(i, c + 0.025, f"{c:.0%}", ha="center", fontsize=8.4, color=INK)
    axD.text(i, 0.04, f"n={n}", ha="center", fontsize=7.2, color="white")
axD.set_xticks(pos)
axD.set_xticklabels([n for n, *_ in cov], fontsize=8.4)
axD.set_ylim(0, 1.1)
axD.set_ylabel("coverage of the 90% interval", fontsize=9)
axD.set_title("Cohort: the block does not", fontsize=9.3, color=INK, pad=8)

# ----------------------------------------------------------------- E: the two modes
axE.set_xlim(0, 1)
axE.set_ylim(0, 1)
axE.set_xticks([])
axE.set_yticks([])
for sp in axE.spines.values():
    sp.set_visible(False)
rows = [("detect transients", "any", r"$q(s_t)$", "default", BLOCK),
        ("rate ratio, point estimate", "any", r"$\mathbb{E}[\beta_{\rm inj}]$",
         "default", BLOCK),
        ("rate ratio, interval", "one recording", r"$\mathrm{SD}[\beta_{\rm inj}]$",
         "default", BLOCK),
        ("drug contrast, interval", "a cohort",
         r"$\mathrm{SD}[\mu_{d_1}\!-\!\mu_{d_2}]$", "calibrated", LR)]
for x, lab in ((0.015, "question"), (0.335, "data"), (0.52, "reads"), (0.80, "mode")):
    axE.text(x, 0.955, lab, fontsize=8.2, color=MUTED, style="italic")
axE.plot([0.01, 0.99], [0.905, 0.905], lw=0.8, color=GRID)
for i, (what, data, reads, mode, col) in enumerate(rows):
    yy = 0.79 - i * 0.185
    axE.text(0.015, yy, what, fontsize=8.8, color=INK, va="center")
    axE.text(0.335, yy, data, fontsize=8.4, color=MUTED, va="center")
    axE.text(0.52, yy, reads, fontsize=9.0, color=INK, va="center")
    axE.add_patch(FancyBboxPatch((0.78, yy - 0.055), 0.20, 0.11,
                                 boxstyle="round,pad=0.006,rounding_size=0.025",
                                 fc="white", ec=col, lw=1.3))
    axE.text(0.88, yy, mode, fontsize=8.4, color=col, ha="center", va="center")
axE.text(0.5, 0.015,
         "The line falls between the two models, not between detection and inference: "
         "three of the four\nreadouts are served by one fit, and only the cohort "
         "interval needs the post-hoc correction.",
         ha="center", va="bottom", fontsize=8.2, color=MUTED, linespacing=1.7)
axE.set_title("Which mode answers which question", fontsize=9.3, color=INK, pad=8)

fig.savefig(out, facecolor="white", bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=a.dpi, facecolor="white", bbox_inches="tight")
print(f"wrote {out}")
for n, v, _ in bars:
    print(f"  {n:>30}: {v:.2f}")
for k in steps:
    m = st.median([st.mean(v) / H[s] for s, v in lr[k].items() if s in H])
    print(f"  {'LR refit ' + str(k):>30}: {m:.2f}")
for n, c, nn, _ in cov:
    print(f"  {'cohort ' + n:>30}: {c:.0%} (n={nn})")
