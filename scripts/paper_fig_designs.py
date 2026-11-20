#!/usr/bin/env python3
"""The uncrossed cohort against the crossed one: what the design costs.

    python scripts/paper_fig_designs.py --dev $PHOTOSVI_RUNS \
        --config configs/cohort_comparison.yaml --out diagnostics/fig_uncrossed.pdf

The two cohorts are fitted by the same model with the same settings and the same
seeds, and differ only in which animal received which drug: crossed, every animal
receives every drug; uncrossed, every animal receives vehicle and one opioid. Both
are simulated at the same number of RECORDINGS, which is what an experiment pays for.

What changes is the estimand. Crossed, the animal effect enters both terms of the
opioid contrast and cancels, so the between-animal spread contributes nothing to its
width. Uncrossed, nothing cancels and that spread enters directly, putting a floor of
sigma_animal * sqrt(2/n_per_group) under the contrast's standard error that no
inference can go below. Panel C marks it, because a widening that matches the floor
is a cost of the experiment and a widening beyond it would be something else.

Panel D is the panel that licenses reading panel C at all: a design comparison on
interval widths is meaningless unless the interval is calibrated under each design,
so both are shown against their realised spread before the widths are compared.

  A  the design, as the model reads it off the index vectors
  B  variance components under the uncrossed design against their simulated values
  C  the two null distributions on one axis, with the arithmetic floor marked
  D  reported width against realised spread, both designs
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from photosvi.experiments.cohort import spec_from_config  # noqa: E402

INK, MUTED, GRID = "#2b2b2b", "#8a8a8a", "#d9d9d9"
ACCENT = "#b03030"
CROSSED_C, UNCROSSED_C = "#3b6ea5", "#c2622d"
DRUG_COLOR = ("#9a9a9a", "#3b6ea5", "#c2622d")

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default=os.environ.get("PHOTOSVI_RUNS", "runs"))
ap.add_argument("--config", default="configs/cohort_comparison.yaml")
ap.add_argument("--crossed", default="runs_cohort_frz")
ap.add_argument("--uncrossed", default="runs_cohort_uncrossed")
ap.add_argument("--uncrossed-animals", type=int, default=12)
ap.add_argument("--seeds", type=int, default=20,
                help="replicate window: seeds 0..N-1, matching the collectors. A\n"
                     "fixed window means a divergence inside it is reported by the\n"
                     "n that appears, not selected away by taking whatever ran.")
ap.add_argument("--out", default="diagnostics/fig_uncrossed.pdf")
ap.add_argument("--dpi", type=int, default=200)
ap.add_argument("--macros", default=None, help="write the caption's numbers here")
a = ap.parse_args()


def rows(sub: str) -> list[dict]:
    r = [json.load(open(f)) for f in sorted(glob.glob(f"{a.dev}/{sub}/hier_*.json"))]
    r = [x for x in r if x["delta"] == 0.0 and x.get("estimand_scale", 0) > 0]
    return [x for x in r if x["seed"] < a.seeds]


cr, un = rows(a.crossed), rows(a.uncrossed)
if len(cr) < 5:
    raise SystemExit(f"need the crossed cohorts in {a.dev}/{a.crossed}")
if len(un) < 5:
    raise SystemExit(
        f"only {len(un)} uncrossed cohorts in {a.dev}/{a.uncrossed}; "
        "run scripts/run_cohort_uncrossed.sh first")

DRUGS = tuple(cr[0]["tonic_posterior"].keys())   # condition names in index order
base = yaml.safe_load(open(a.config))
c = base["cohort"]
TRUE_SIGMA = {"sigma_animal": c["sigma_animal"], "sigma_cell": c["sigma_cell"],
              "sigma_b0": c["sigma_b0"], "sigma_b0_session": c["sigma_b0_session"]}
per_group = a.uncrossed_animals // 2
FLOOR = c["sigma_animal"] * math.sqrt(2 / per_group)


def indices(design: str, animals: int):
    cfg = yaml.safe_load(open(a.config))
    cfg["cohort"] |= dict(design=design, n_animals=animals)
    return spec_from_config(cfg, 0.0).indices()


fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0))
fig.subplots_adjust(left=0.085, right=0.975, top=0.93, bottom=0.09,
                    hspace=0.38, wspace=0.28)
for ax, letter in zip(axes.ravel(), "abcd", strict=True):
    ax.grid(alpha=0.25, lw=0.7, color=GRID)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.tick_params(labelsize=8.5, colors="#555555")
    ax.text(-0.14, 1.10, letter, transform=ax.transAxes, fontsize=14,
            weight="bold", color=INK, va="top")

# --------------------------------------------------------------- A: the designs
ax = axes[0, 0]
ax.grid(False)
for k, (label, design, animals, y0) in enumerate((
        ("crossed", "crossed", cr[0]["design"]["n_animals"], 1.0),
        ("uncrossed", "uncrossed", a.uncrossed_animals, 0.0))):
    an, dr = indices(design, animals)
    for s, (ai, di) in enumerate(zip(an, dr, strict=True)):
        ax.add_patch(plt.Rectangle((ai + 0.06, y0 + 0.06 + 0.28 * di), 0.88, 0.24,
                                   facecolor=DRUG_COLOR[di], lw=0))
    ax.text(-0.4, y0 + 0.5, label, fontsize=9, color=INK, ha="right", va="center",
            rotation=90)
    ax.text(animals / 2, y0 + 0.92, f"{animals} animals, {len(an)} recordings",
            fontsize=8, color=MUTED, ha="center")
ax.set_xlim(-0.9, max(a.uncrossed_animals, cr[0]["design"]["n_animals"]) + 0.2)
ax.set_ylim(-0.15, 2.05)
ax.set_yticks([])
ax.set_xlabel("animal", fontsize=9)
for j, name in enumerate(DRUGS):
    ax.add_patch(plt.Rectangle((0.2 + j * 3.0, -0.13), 0.6, 0.09,
                               facecolor=DRUG_COLOR[j], lw=0, clip_on=False))
    ax.text(0.95 + j * 3.0, -0.085, name, fontsize=7.5, color=MUTED, va="center")

# -------------------------------------------- B: components under the uncrossed
ax = axes[0, 1]
names = [n for n in ("sigma_animal", "sigma_cell", "sigma_b0", "sigma_b0_session")
         if n in un[0]["variance_components"]]
pos = np.arange(len(names))
data = [[r["variance_components"][n] for r in un] for n in names]
bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                medianprops=dict(color=INK, lw=1.3),
                boxprops=dict(facecolor="#f6e9e1", edgecolor=UNCROSSED_C, lw=0.9),
                whiskerprops=dict(color=UNCROSSED_C, lw=0.9),
                capprops=dict(color=UNCROSSED_C, lw=0.9),
                flierprops=dict(marker="o", ms=2.5, mfc=MUTED, mec="none"))
for i, n in enumerate(names):
    ax.plot([i - 0.34, i + 0.34], [TRUE_SIGMA[n]] * 2, lw=2.0, color=ACCENT, zorder=5,
            solid_capstyle="butt")
ax.plot([], [], lw=2.0, color=ACCENT, label="simulated value")
ax.set_xticks(pos)
ax.set_xticklabels([n.replace("sigma_", r"$\sigma_{").replace("_", r"\,") + "}$"
                    for n in names], fontsize=8)
ax.set_ylabel("posterior median", fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")

# ------------------------------------ C: the two null distributions, and the floor
ax = axes[1, 0]
g = None
for r, col, label in ((cr, CROSSED_C, "crossed"), (un, UNCROSSED_C, "uncrossed")):
    est = np.array([x["estimand_loc"] for x in r])
    s = float(st.stdev(est))
    if g is None:
        w = max(4 * s, 4 * float(st.stdev([x["estimand_loc"] for x in un])))
        g = np.linspace(-w, w, 400)
    ax.plot(g, np.exp(-0.5 * (g / s) ** 2) / (s * np.sqrt(2 * np.pi)), lw=2.0,
            color=col, label=f"{label}   spread {s:.3f}")
    ax.fill_between(g, 0, np.exp(-0.5 * (g / s) ** 2) / (s * np.sqrt(2 * np.pi)),
                    color=col, alpha=0.12)
for sign in (-1, 1):
    ax.axvline(sign * FLOOR, lw=1.2, color=ACCENT, ls=(0, (4, 3)))
# An UNBIASED estimator's floor. Partial pooling shrinks the contrast toward zero,
# trading bias for variance, and these are null cohorts -- so the shrinkage costs no
# bias and the hierarchical estimate can legitimately sit below this line. Saying
# "floor" without the qualifier turns a correct estimate into an apparent violation.
ax.text(FLOOR, ax.get_ylim()[1] * 0.96,
        f"  unbiased-estimator floor\n  $\\pm${FLOOR:.3f}", fontsize=7.5, color=ACCENT,
        ha="left", va="top")
ax.axvline(0.0, lw=1.0, color=MUTED)
ax.set_xlabel("estimated contrast (true value 0)", fontsize=9)
ax.set_ylabel("density", fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc="upper left", handletextpad=0.5)

# ------------------------------------- D: is the interval calibrated under each?
ax = axes[1, 1]
tallest = 0.0
for i, (r, col, label) in enumerate(((cr, CROSSED_C, "crossed"),
                                     (un, UNCROSSED_C, "uncrossed"))):
    sd = st.median(x["estimand_scale"] for x in r)
    spread = st.stdev([x["estimand_loc"] for x in r])
    cover = sum(1 for x in r
                if abs(x["estimand_loc"]) <= 1.645 * x["estimand_scale"]) / len(r)
    tallest = max(tallest, sd, spread)
    ax.bar([i - 0.18], [sd], width=0.34, color=col, alpha=0.45, lw=0)
    ax.bar([i + 0.18], [spread], width=0.34, color=col, lw=0)
    ax.text(i, max(sd, spread) * 1.04,
            f"{spread / sd:.2f}$\\times$\n{100 * cover:.0f}% cover",
            fontsize=8, color=INK, ha="center", va="bottom")
    ax.text(i, -0.004, f"{label}\nn={len(r)}", fontsize=8, color=MUTED,
            ha="center", va="top")
# Explicit handles: an empty `bar` call does not carry its colour into the legend,
# so both swatches came out the same and the light/solid distinction was lost.
ax.legend(handles=[Patch(facecolor=MUTED, alpha=0.45, label="reported width"),
                   Patch(facecolor=MUTED, label="realised spread")],
          fontsize=7.5, frameon=False, loc="upper right", handletextpad=0.5)
ax.set_xlim(-0.6, 1.9)
# Headroom for the per-bar annotation, which otherwise runs into the title.
ax.set_ylim(0, tallest * 1.38)
ax.set_xticks([])
ax.set_ylabel("standard deviation of the contrast", fontsize=9)

if a.macros:
    def _stats(r):
        sd = st.median(x["estimand_scale"] for x in r)
        sp = st.stdev([x["estimand_loc"] for x in r])
        cv = sum(1 for x in r if abs(x["estimand_loc"]) <= 1.645 * x["estimand_scale"]) / len(r)
        return sd, sp, cv
    (csd, csp, ccv), (usd, usp, ucv) = _stats(cr), _stats(un)
    Path(a.macros).write_text(
        "% Generated by scripts/paper_fig_designs.py -- do not edit.\n"
        f"\\newcommand{{\\ppFigUnN}}{{{len(un)}}}\n"
        f"\\newcommand{{\\ppFigUnAnimals}}{{{a.uncrossed_animals}}}\n"
        f"\\newcommand{{\\ppFigUnRatio}}{{{usp / usd:.2f}}}\n"
        f"\\newcommand{{\\ppFigUnCover}}{{{100 * ucv:.0f}\\%}}\n"
        f"\\newcommand{{\\ppFigUnSpread}}{{{usp:.3f}}}\n"
        f"\\newcommand{{\\ppFigCrSpread}}{{{csp:.3f}}}\n"
        f"\\newcommand{{\\ppFigUnWidthRatio}}{{{usd / csd:.1f}}}\n"
        f"\\newcommand{{\\ppFigFloor}}{{{FLOOR:.3f}}}\n")
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor="white", bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=a.dpi, facecolor="white", bbox_inches="tight")
print(f"wrote {out}  (crossed n={len(cr)}, uncrossed n={len(un)})")
for r, label in ((cr, "crossed"), (un, "uncrossed")):
    sd = st.median(x["estimand_scale"] for x in r)
    spread = st.stdev([x["estimand_loc"] for x in r])
    cover = sum(1 for x in r
                if abs(x["estimand_loc"]) <= 1.645 * x["estimand_scale"]) / len(r)
    print(f"  {label:>10}: reported {sd:.4f} | spread {spread:.4f} "
          f"| ratio {spread / sd:.2f} | cover {100 * cover:.0f}%")
print(f"  between-animal floor on the uncrossed contrast: {FLOOR:.4f}")
