#!/usr/bin/env python3
"""QQ plots of the standardised residual, per contrast and per method.

    python scripts/paper_fig_qq.py --out $DEV/diagnostics/fig_qq.pdf

What a credible interval promises is that `z = (estimate - truth) / width` behaves
like a standard normal. Plotting the sorted z against normal quantiles turns that
promise into a picture: a slope above one is an interval too narrow by that factor,
an intercept away from zero is bias, and a bent line is neither.

It is worth doing per contrast rather than only for the estimand. The three contrasts
are estimated from the same fit but ask different questions -- two of them compare an
opioid against vehicle, which is a between-condition comparison anchored on five
animals, and the third compares the opioids to each other, which is within-animal for
most of the cohort. There is no reason for their calibration to be the same, and the
grid is what shows whether it is.

Linear response occupies only one panel of its row, and that is the point rather than
an omission: the tilt runs along one direction in drug space, so it corrects the
contrast it was aimed at. The other two would each need their own tilt and their own
refits.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#2b2b2b", "#8a8a8a", "#d9d9d9"
BLOCK, LR, BAD = "#3b6ea5", "#2f8f6f", "#b3453a"

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default=os.environ.get("PHOTOSVI_RUNS", "runs"))
ap.add_argument("--out", default=None)
ap.add_argument("--config", default="configs/cohort_comparison.yaml")
ap.add_argument("--dpi", type=int, default=200)
a = ap.parse_args()
out = Path(a.out or f"{a.dev}/diagnostics/fig_qq.pdf")

import yaml  # noqa: E402

from photosvi.experiments.cohort import DRUG_NAMES, spec_from_config  # noqa: E402

spec = spec_from_config(yaml.safe_load(open(a.config)), 0.0)
MU = dict(zip(DRUG_NAMES, spec.mu_drug, strict=True))
CONTRASTS = [
    ("opioid1-saline", "opioid 1 $-$ vehicle", MU["opioid1"] - MU["saline"]),
    ("opioid2-saline", "opioid 2 $-$ vehicle", MU["opioid2"] - MU["saline"]),
    ("opioid2-opioid1", "opioid 2 $-$ opioid 1", MU["opioid2"] - MU["opioid1"]),
]


def block_rows(sub: str = "runs_cohort_fr8"):
    """(loc, scale) per contrast from one hierarchical arm."""
    out_ = {k: [] for k, _, _ in CONTRASTS}
    for f in glob.glob(f"{a.dev}/{sub}/hier_*.json"):
        d = json.load(open(f))
        if d["delta"] != 0.0:
            continue
        for k, _, _ in CONTRASTS:
            loc, sc = d["contrasts"][k]
            if sc > 0:
                out_[k].append((loc, sc))
    return out_


def lr_rows():
    """The estimand only: the tilt was aimed at opioid2 - opioid1.

    Also returns the block's own widths on the SAME cohorts. The block row is drawn
    from 23 cohorts and this one from however many have finished, so comparing their
    slopes directly would be comparing different cohorts and calling the difference a
    correction.
    """
    out_ = {k: [] for k, _, _ in CONTRASTS}
    paired = []
    for f in glob.glob(f"{a.dev}/runs_cohort_lr/lr_*.json"):
        d = json.load(open(f))
        best = [x["sd_contrast"] for x in d["results"]
                if x["tilt_sds"] == 0.5 and x["refit_steps"] == 15000]
        sd = next((v for v in best if v == v), None)
        if sd is None:                       # deep refit diverged; fall back
            sd = next((x["sd_contrast"] for x in d["results"]
                       if x["tilt_sds"] == 0.5 and x["sd_contrast"] == x["sd_contrast"]),
                      None)
        if sd:
            out_["opioid2-opioid1"].append((d["estimand_loc"], sd))
            paired.append((d["estimand_loc"], d["estimand_scale"]))
    return out_, paired


def qq(ax, pairs, truth, colour, label):
    """Sorted z against normal quantiles, with the through-origin slope."""
    if not pairs:
        return None
    z = np.sort([(loc - truth) / sc for loc, sc in pairs])
    n = z.size
    # Blom plotting positions: the standard choice, and unbiased enough at these n.
    q = np.array([_probit((i + 1 - 0.375) / (n + 0.25)) for i in range(n)])
    slope = float((q @ z) / (q @ q))         # least squares through the origin
    ax.plot([-3, 3], [-3, 3], lw=1.2, color=INK, ls=(0, (4, 3)), zorder=1)
    ax.plot([-3, 3], [-3 * slope, 3 * slope], lw=1.6, color=colour, alpha=.8, zorder=2)
    ax.scatter(q, z, s=26, color=colour, edgecolor="white", lw=.7, zorder=3)
    ax.text(.04, .96, f"{label}\nslope {slope:.2f}   n={n}", transform=ax.transAxes,
            fontsize=8.2, va="top", color=INK, linespacing=1.5)
    return slope


def _probit(p: float) -> float:
    """Inverse standard normal CDF by bisection -- one number, no scipy import."""
    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.6), sharex=True, sharey=True)
fig.subplots_adjust(left=.075, right=.975, top=.90, bottom=.095, hspace=.24, wspace=.12)
B = block_rows()
L, PAIRED = lr_rows()

for col, (key, title, truth) in enumerate(CONTRASTS):
    for row, (data, colour, name) in enumerate(
            ((B, BLOCK, "8$\\times$8 block"), (L, LR, "block + linear response"))):
        ax = axes[row, col]
        ax.grid(alpha=.25, lw=.7, color=GRID)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
        ax.tick_params(labelsize=8.5, colors="#555555")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-6, 6)
        if data[key]:
            qq(ax, data[key], truth, colour, name)
            if row == 1 and PAIRED:
                # The same cohorts without the correction, so the gain is within-cohort.
                zb = np.sort([(lo - truth) / sc for lo, sc in PAIRED])
                n = zb.size
                qb = np.array([_probit((i + 1 - 0.375) / (n + 0.25)) for i in range(n)])
                sb = float((qb @ zb) / (qb @ qb))
                ax.plot([-3, 3], [-3 * sb, 3 * sb], lw=1.4, color=BLOCK, alpha=.55,
                        ls=(0, (5, 2)), zorder=2)
                ax.text(.04, .80, f"same cohorts, block alone: {sb:.2f}",
                        transform=ax.transAxes, fontsize=7.8, color=BLOCK, va="top")
        else:
            ax.text(.5, .5, "not corrected\n\nthe tilt runs along one\ndirection in drug"
                            " space,\nso it corrects the contrast\nit was aimed at",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8.2, color=MUTED, linespacing=1.6)
        if row == 0:
            ax.set_title(f"{title}\ntrue value {truth:+.2f}", fontsize=9.4,
                         color=INK, pad=8)
        if col == 0:
            ax.set_ylabel("observed  $(\\hat\\theta-\\theta)/\\mathrm{sd}$", fontsize=9)
        if row == 1:
            ax.set_xlabel("normal quantile", fontsize=9)

fig.suptitle("Is the reported interval the width it claims to be?",
             fontsize=11.5, color=INK, x=.075, ha="left", y=1.035)
fig.text(.075, .975,
         "Under a calibrated interval the standardised residual is standard normal: "
         "points on the dashed line. A slope above one is an interval too narrow by "
         "that factor; an offset is bias.",
         fontsize=8.6, color=MUTED, ha="left", linespacing=1.6)

fig.savefig(out, facecolor="white", bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=a.dpi, facecolor="white", bbox_inches="tight")
print(f"wrote {out}")
for key, title, truth in CONTRASTS:
    for data, name in ((B, "block"), (L, "block+LR")):
        if data[key]:
            z = [(loc - truth) / sc for loc, sc in data[key]]
            print(f"  {title:<24} {name:<9} n={len(z):>2}  mean z {np.mean(z):+.2f}  "
                  f"sd z {np.std(z):.2f}")
