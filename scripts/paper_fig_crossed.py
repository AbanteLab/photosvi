#!/usr/bin/env python3
"""What the cohort model recovers on a fully crossed design.

    python scripts/paper_fig_crossed.py --runs runs_cohort_frz \
        --config configs/cohort_comparison.yaml --out diagnostics/fig_crossed.pdf

Replaces `paper_fig_cohort.py`, which this supersedes on two counts. That figure was
drawn from the mean-field arm, whose interval was three times narrower than the
estimator's spread, and its panel C was a statement about that failure; the shipped
family does not have it, so panel C now shows an agreement and has to say which
family is responsible. And it assigned a session's drug by `i % n_drugs`, which is
true only of the crossed ordering -- the companion figure for the uncrossed design
would have silently mislabelled every point. Indices are now regenerated from the
spec, which is deterministic in (config, seed) and therefore cannot drift from what
was fitted.

Four panels:

  A  partial pooling -- per-session estimates against their simulated truth, with
     the shrinkage toward each drug's mean visible as a slope below one
  B  per-drug pharmacokinetics -- recovered Bateman peak times against the truth,
     the quantity no per-session fit and no two-stage analysis can produce
  C  the null distribution of the contrast against the width the posterior claims
  D  variance components against their simulated values, with the one that governs
     THIS design's contrast marked -- in a crossed design the animal effect cancels
     out of a drug contrast and the animal x drug interaction is what remains
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
# TrueType, not Type 3, so the PDF stays editable in a vector editor.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)

from photosvi.experiments.cohort import spec_from_config  # noqa: E402

INK, MUTED, GRID = "#2b2b2b", "#8a8a8a", "#d9d9d9"
GOVERNS = "#b03030"
PALETTE = ("#9a9a9a", "#3b6ea5", "#c2622d")

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs_cohort_frz")
ap.add_argument("--config", default="configs/cohort_comparison.yaml")
ap.add_argument("--design", default="crossed", choices=("crossed", "uncrossed"))
ap.add_argument("--animals", type=int, default=None,
                help="override n_animals; default reads it from the first run record")
ap.add_argument("--out", default="diagnostics/fig_crossed.pdf")
ap.add_argument("--dpi", type=int, default=200)
ap.add_argument("--macros", default=None, help="write the caption's numbers here")
ap.add_argument("--panels", default="abcd",
                help="which of the four panels to draw, relettered in order; the paper "
                     "uses 'ac' (shrinkage and calibration) since 2026-10-08")
a = ap.parse_args()

rows = [json.load(open(f)) for f in sorted(glob.glob(f"{a.runs}/hier_*.json"))]
rows = [r for r in rows if r["delta"] == 0.0 and r.get("estimand_scale", 0) > 0]
if not rows:
    raise SystemExit(f"no usable hierarchical results in {a.runs}")

cfg = yaml.safe_load(open(a.config))
cfg["cohort"] |= dict(design=a.design,
                      n_animals=a.animals or rows[0]["design"]["n_animals"])
spec = spec_from_config(cfg, 0.0)
_, drug_of_session = spec.indices()
if len(drug_of_session) != len(rows[0]["session_beta_inj"]):
    raise SystemExit(
        f"design mismatch: spec gives {len(drug_of_session)} sessions, the runs have "
        f"{len(rows[0]['session_beta_inj'])}. Check --design and --animals.")

# Condition names in drug-index order, from the run itself: the three-drug study calls
# them saline/opioid1/opioid2, the two-condition one vehicle/treated.
DRUGS = tuple(rows[0]["tonic_posterior"].keys())
DRUG_COLOR = dict(zip(DRUGS, PALETTE))
c = cfg["cohort"]
TRUE_SIGMA = {"sigma_animal": c["sigma_animal"], "sigma_cell": c["sigma_cell"],
              "sigma_b0": c["sigma_b0"], "sigma_b0_session": c["sigma_b0_session"]}
# Which component sets the width of the opioid-vs-opioid contrast. Crossed: the
# animal effect enters both terms and cancels, leaving the interaction. Uncrossed:
# nothing cancels and the between-animal spread enters directly.
GOVERNING = "sigma_cell" if a.design == "crossed" else "sigma_animal"

PANELS = a.panels
if len(PANELS) == 4:
    fig, _ax = plt.subplots(2, 2, figsize=(9.6, 8.0))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.935, bottom=0.09,
                        hspace=0.36, wspace=0.28)
else:
    fig, _ax = plt.subplots(1, len(PANELS), figsize=(4.8 * len(PANELS), 4.0))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.90, bottom=0.14, wspace=0.28)
AX = dict(zip(PANELS, np.atleast_1d(_ax).ravel(), strict=True))
for (key, ax), letter in zip(AX.items(), "abcd"):
    ax.grid(alpha=0.25, lw=0.7, color=GRID)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.tick_params(labelsize=8.5, colors="#555555")
    ax.text(-0.14, 1.10, letter, transform=ax.transAxes, fontsize=14,
            weight="bold", color=INK, va="top")

# ---------------------------------------------------------------- A: shrinkage
ax = AX.get("a") or plt.figure().add_subplot()
for d, name in enumerate(DRUGS):
    tx = [b for r in rows for i, b in enumerate(r["session_beta_inj_true"])
          if drug_of_session[i] == d]
    fy = [b for r in rows for i, b in enumerate(r["session_beta_inj"])
          if drug_of_session[i] == d]
    if tx:
        ax.scatter(tx, fy, s=7, alpha=0.45, lw=0, color=DRUG_COLOR[name], label=name)
allx = [b for r in rows for b in r["session_beta_inj_true"]]
ally = [b for r in rows for b in r["session_beta_inj"]]
lim = (min(allx + ally) - 0.1, max(allx + ally) + 0.1)
ax.plot(lim, lim, lw=1.0, color=INK, ls=(0, (4, 3)), zorder=1)
sl, ic = np.polyfit(allx, ally, 1)
xs = np.array(lim)
ax.plot(xs, sl * xs + ic, lw=1.4, color=GOVERNS, zorder=2)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel(r"true per-session $\beta_{\rm inj}$", fontsize=9)
ax.set_ylabel(r"estimated $\beta_{\rm inj}$", fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc="upper left", handletextpad=0.3)

# ------------------------------------------------------- B: pharmacokinetics
ax = AX.get("b") or plt.figure().add_subplot()
# Histograms, not a scatter against the truth. The true peak takes one of three
# values, so a scatter puts 24 points on each of three vertical lines and the shape
# of the error -- which is the whole content of this panel -- is hidden by overplot.
# Each drug gets its own distribution and its own rule at the simulated value, so
# bias reads as an offset between a histogram and its rule, and precision as width.
peaks = {name: [r["tonic_posterior"][name]["peak_s"] for r in rows]
         for name in DRUGS}
true_peak = {name: rows[0]["tonic_peak_true"][d] for d, name in enumerate(DRUGS)}
# Range from the estimates and the truths, not from zero: when every estimate sits
# within a few percent of the truth, a zero-based axis squeezes the whole panel into
# one bar.
vals = [x for v in peaks.values() for x in v] + list(true_peak.values())
lo_, hi = min(vals), max(vals)
pad = 0.15 * (hi - lo_) + 0.02 * hi
bins = np.linspace(lo_ - pad, hi + pad, 26)
for name in DRUGS:
    fit = peaks[name]
    hits = sum(1 for f in fit if abs(f - true_peak[name]) / true_peak[name] < 0.25)
    ax.hist(fit, bins=bins, color=DRUG_COLOR[name], alpha=0.42, lw=0)
    ax.hist(fit, bins=bins, histtype="step", color=DRUG_COLOR[name], lw=1.3,
            label=f"{name}   true {true_peak[name]:.0f}s   ({hits}/{len(fit)} within 25%)")
    # The rule carries the drug's colour, so the value belongs in the legend rather
    # than as a rotated label competing with it for the top of the axes.
    ax.axvline(true_peak[name], lw=1.5, color=DRUG_COLOR[name], ls=(0, (4, 2.5)))
# Headroom so the legend clears the tallest bar.
ax.set_ylim(0, max(np.histogram(v, bins=bins)[0].max() for v in peaks.values()) * 1.42)
ax.set_xlabel("estimated Bateman peak (s after injection)", fontsize=9)
ax.set_ylabel(f"cohorts (of {len(rows)})", fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc="upper right", handletextpad=0.6)

# -------------------------------- C: the reported width against the actual spread
ax = AX.get("c") or plt.figure().add_subplot()


def _averaged_loc(r: dict, tail: int = 6) -> float:
    """The estimand over the last probes, matching tools/collect_variants.py.

    The stopping rule watches the drug levels, not the contrast, and the contrast
    moves by a good fraction of its own size between consecutive probes -- so the
    value at the step the rule happened to fire samples that oscillation once.
    Reading it raw here while the paper's table reads it averaged gave two numbers
    for one quantity, 1.03x in this panel against 0.99x in S2 Text.
    """
    # the last contrast is the one reported: op2 - op1 with three drugs, the only one with two
    tr = [t["contrasts"][-1] for t in r.get("probe_trace", []) if "contrasts" in t]
    return st.mean(tr[-tail:]) if tr else r["estimand_loc"]


est = np.array([_averaged_loc(r) for r in rows])
sc = float(st.median([r["estimand_scale"] for r in rows]))
# Sample standard deviation: these cohorts are a sample from the sampling
# distribution being estimated, and dividing by n understates it.
emp = float(st.stdev(est))
ax.hist(est, bins=12, color="#c9d6e4", edgecolor="#5b7fa6", lw=0.8, density=True)
g = np.linspace(est.min() - 0.12, est.max() + 0.12, 400)


def norm(x, s):
    return np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))


ax.plot(g, norm(g, sc), lw=1.8, color=GOVERNS, label=f"posterior claims  SD {sc:.3f}")
ax.plot(g, norm(g, emp), lw=1.8, color=INK, ls=(0, (5, 2)),
        label=f"actual spread  SD {emp:.3f}")
ax.axvline(0.0, lw=1.0, color=MUTED)
ax.set_xlabel("estimated contrast (true value 0)", fontsize=9)
ax.set_ylabel("density", fontsize=9)
cover = sum(1 for r in rows
            if abs(_averaged_loc(r)) <= 1.645 * r["estimand_scale"]) / len(rows)
ax.legend(fontsize=7.5, frameon=False, loc="upper right", handletextpad=0.5)

# ------------------------------------------------------- D: variance components
ax = AX.get("d") or plt.figure().add_subplot()
names = [n for n in ("sigma_animal", "sigma_cell", "sigma_b0", "sigma_b0_session")
         if n in rows[0]["variance_components"]]
pos = np.arange(len(names))
data = [[r["variance_components"][n] for r in rows] for n in names]
bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                medianprops=dict(color=INK, lw=1.3),
                boxprops=dict(facecolor="#e8eef4", edgecolor="#5b7fa6", lw=0.9),
                whiskerprops=dict(color="#5b7fa6", lw=0.9),
                capprops=dict(color="#5b7fa6", lw=0.9),
                flierprops=dict(marker="o", ms=2.5, mfc=MUTED, mec="none"))
# Mark the box whose component sets the contrast's width, because which one that is
# changes with the design and reading the wrong row is how this project spent a week.
if GOVERNING in names:
    i = names.index(GOVERNING)
    bp["boxes"][i].set_facecolor("#f6e4e4")
    bp["boxes"][i].set_edgecolor(GOVERNS)
    # Anchored in axes fraction, not data: the free space is reliably above the
    # boxes, while a data-coordinate offset collided with the axis labels below.
    ax.annotate("sets the contrast's width\nunder this design",
                xy=(i, max(data[i])), xycoords="data",
                xytext=(0.30, 0.96), textcoords="axes fraction",
                fontsize=7.5, color=GOVERNS, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=GOVERNS, lw=0.8,
                                shrinkA=2, shrinkB=4))
for i, n in enumerate(names):
    ax.plot([i - 0.34, i + 0.34], [TRUE_SIGMA[n]] * 2, lw=2.0, color=GOVERNS,
            zorder=5, solid_capstyle="butt")
ax.plot([], [], lw=2.0, color=GOVERNS, label="simulated value")
ax.set_xticks(pos)
ax.set_xticklabels([n.replace("sigma_", r"$\sigma_{").replace("_", r"\,") + "}$"
                    for n in names], fontsize=8)
ax.set_ylabel("posterior median", fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")

if a.macros:
    Path(a.macros).write_text(
        "% Generated by scripts/paper_fig_crossed.py -- do not edit.\n"
        f"\\newcommand{{\\ppFigCrN}}{{{len(rows)}}}\n"
        f"\\newcommand{{\\ppFigCrSlope}}{{{sl:.2f}}}\n"
        f"\\newcommand{{\\ppFigCrRatio}}{{{emp / sc:.2f}}}\n"
        f"\\newcommand{{\\ppFigCrCover}}{{{100 * cover:.0f}\\%}}\n"
        "\\newcommand{\\ppFigTail}{6}\n")
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor="white", bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=a.dpi, facecolor="white", bbox_inches="tight")
print(f"wrote {out} and {out.with_suffix('.png')}  ({len(rows)} cohorts, {a.design})")
print(f"  shrinkage slope {sl:.3f} | spread/width {emp / sc:.2f}x "
      f"(spread {emp:.4f} vs reported {sc:.4f}) | coverage {100 * cover:.0f}%")
for n in names:
    print(f"  {n:>18}: median {st.median(x for x in data[names.index(n)]):.3f} "
          f"| true {TRUE_SIGMA[n]:.3f}"
          + ("   <- governs this design's contrast" if n == GOVERNING else ""))
