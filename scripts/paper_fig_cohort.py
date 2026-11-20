#!/usr/bin/env python3
"""Figure 4: what the cohort model recovers, and where its interval is not honest.

    python scripts/paper_fig_cohort.py --runs runs_cohort --out diagnostics/fig_cohort.pdf

**Preliminary.** Every cohort available when this was written is a null cohort
(delta = 0), so the figure shows what the model estimates and how well, and says
nothing about power -- the panel that compares arms needs the alternative cells and
is deliberately absent rather than faked from the nulls.

Four panels:

  A  partial pooling -- per-session estimates against their simulated truth, with
     the shrinkage toward each drug's mean visible as a slope below one
  B  per-drug pharmacokinetics -- recovered Bateman peak times against the truth,
     the quantity no per-session fit and no two-stage analysis can produce
  C  the null distribution of the contrast against the width the posterior claims.
     The two disagree by a factor of about three; this is a calibration statement
     about the reported interval, not an attribution of the cause
  D  variance components against their simulated values
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
# TrueType, not Type 3, so the PDF stays editable in a vector editor.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)

INK, MUTED, GRID = "#2b2b2b", "#8a8a8a", "#d9d9d9"
DRUG_COLOR = {"saline": "#9a9a9a", "opioid1": "#3b6ea5", "opioid2": "#c2622d"}
DRUGS = ("saline", "opioid1", "opioid2")

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs_cohort")
ap.add_argument("--config", default="configs/cohort_comparison.yaml")
ap.add_argument("--out", default="diagnostics/fig_cohort.pdf")
ap.add_argument("--dpi", type=int, default=200)
a = ap.parse_args()

rows = [json.load(open(f)) for f in sorted(glob.glob(f"{a.runs}/hier_*.json"))]
if not rows:
    raise SystemExit(f"no hierarchical results in {a.runs}")
cfg = yaml.safe_load(open(a.config))["cohort"]
TRUE_SIGMA = {"sigma_animal": cfg["sigma_animal"], "sigma_cell": cfg["sigma_cell"],
              "sigma_b0": cfg["sigma_b0"], "sigma_b0_session": cfg["sigma_b0_session"]}

fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0))
fig.subplots_adjust(left=0.085, right=0.975, top=0.935, bottom=0.105,
                    hspace=0.36, wspace=0.28)
for ax, letter in zip(axes.ravel(), "ABCD", strict=True):
    ax.grid(alpha=0.25, lw=0.7, color=GRID)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.tick_params(labelsize=8.5, colors="#555555")
    ax.text(-0.14, 1.10, letter, transform=ax.transAxes, fontsize=14,
            weight="bold", color=INK, va="top")

# ---------------------------------------------------------------- A: shrinkage
ax = axes[0, 0]
n_drugs = len(DRUGS)
for d, name in enumerate(DRUGS):
    tx = [b for r in rows for i, b in enumerate(r["session_beta_inj_true"])
          if i % n_drugs == d]
    fy = [b for r in rows for i, b in enumerate(r["session_beta_inj"])
          if i % n_drugs == d]
    ax.scatter(tx, fy, s=7, alpha=0.45, lw=0, color=DRUG_COLOR[name], label=name)
allx = [b for r in rows for b in r["session_beta_inj_true"]]
ally = [b for r in rows for b in r["session_beta_inj"]]
lim = (min(allx + ally) - 0.1, max(allx + ally) + 0.1)
ax.plot(lim, lim, lw=1.0, color=INK, ls=(0, (4, 3)), zorder=1)
sl, ic = np.polyfit(allx, ally, 1)
xs = np.array(lim)
ax.plot(xs, sl * xs + ic, lw=1.4, color="#b03030", zorder=2)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r"true per-session $\beta_{\rm inj}$", fontsize=9)
ax.set_ylabel(r"estimated $\beta_{\rm inj}$", fontsize=9)
ax.set_title(f"Partial pooling shrinks toward the drug mean\n"
             f"slope {sl:.2f} (dashed: identity)", fontsize=9.5, color=INK, pad=8)
ax.legend(fontsize=7.5, frameon=False, loc="upper left", handletextpad=0.3)

# ------------------------------------------------------- B: pharmacokinetics
ax = axes[0, 1]
# Report the hit rate per drug rather than asserting "recovered": the three differ
# a lot, and the spread across cohorts is the point -- a median close to the truth
# would hide that vehicle lands anywhere from 10 s to 490 s.
hits = {}
for d, name in enumerate(DRUGS):
    true = [r["tonic_peak_true"][d] for r in rows]
    fit = [r["tonic_posterior"][name]["peak_s"] for r in rows]
    hits[name] = sum(1 for t, f in zip(true, fit, strict=True)
                     if abs(f - t) / t < 0.25)
    ax.scatter(true, fit, s=22, alpha=0.6, lw=0, color=DRUG_COLOR[name],
               label=f"{name}  ({hits[name]}/{len(fit)})")
ax.plot([0, 700], [0, 700], lw=1.0, color=INK, ls=(0, (4, 3)), zorder=1)
ax.set_xlim(0, 620); ax.set_ylim(0, 700)
ax.set_xlabel("true Bateman peak (s after injection)", fontsize=9)
ax.set_ylabel("estimated peak (s)", fontsize=9)
ax.set_title("Per-drug pharmacokinetics, and how often\n"
             "each lands within 25% of the truth", fontsize=9.5, color=INK, pad=8)
ax.legend(fontsize=7.5, frameon=False, loc="upper left", handletextpad=0.3,
          title="within 25%", title_fontsize=7.5)

# ------------------------- C: the interval does not match the estimator's spread
ax = axes[1, 0]
est = np.array([r["estimand_loc"] for r in rows])
sc = float(np.mean([r["estimand_scale"] for r in rows]))
emp = float(np.std(est))
ax.hist(est, bins=12, color="#c9d6e4", edgecolor="#5b7fa6", lw=0.8, density=True)
g = np.linspace(est.min() - 0.12, est.max() + 0.12, 400)
norm = lambda x, s: np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))  # noqa: E731
ax.plot(g, norm(g, sc), lw=1.8, color="#b03030",
        label=f"posterior claims  SD {sc:.3f}")
ax.plot(g, norm(g, emp), lw=1.8, color=INK, ls=(0, (5, 2)),
        label=f"actual spread  SD {emp:.3f}")
ax.axvline(0.0, lw=1.0, color=MUTED)
ax.set_xlabel(r"$\hat\mu_{\rm op2} - \hat\mu_{\rm op1}$   (true value 0)", fontsize=9)
ax.set_ylabel("density", fontsize=9)
# "narrower than the estimator's spread" is what was measured; "too narrow" would
# assert a cause -- an exact posterior need not match a sampling spread at a fixed
# truth, and the spread also carries the optimiser's own variability.
ax.set_title(f"The estimator's spread is {emp/sc:.1f}$\\times$ the reported width\n"
             f"{len(est)} null cohorts", fontsize=9.5, color=INK, pad=8)
ax.legend(fontsize=7.5, frameon=False, loc="upper right", handletextpad=0.5)

# ------------------------------------------------------- D: variance components
ax = axes[1, 1]
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
for i, n in enumerate(names):
    ax.plot([i - 0.34, i + 0.34], [TRUE_SIGMA[n]] * 2, lw=2.0, color="#b03030",
            zorder=5, solid_capstyle="butt")
ax.plot([], [], lw=2.0, color="#b03030", label="simulated value")
ax.set_xticks(pos)
ax.set_xticklabels([n.replace("sigma_", r"$\sigma_{").replace("_", r"\,") + "}$"
                    for n in names], fontsize=8)
ax.set_ylabel("posterior median", fontsize=9)
ax.set_title("Variance components, which no per-session\nanalysis produces",
             fontsize=9.5, color=INK, pad=8)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")

fig.text(0.5, 0.022,
         f"Preliminary: all {len(rows)} cohorts shown are null cohorts "
         r"($\Delta = 0$). Power against effect size needs the alternative cells.",
         ha="center", fontsize=8, color=MUTED, style="italic")

out = Path(a.out)
fig.savefig(out, facecolor="white", bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=a.dpi, facecolor="white", bbox_inches="tight")
print(f"wrote {out} and {out.with_suffix('.png')}  ({len(rows)} cohorts)")
print(f"  shrinkage slope {sl:.3f} | overconfidence {emp/sc:.2f}x "
      f"(emp {emp:.4f} vs claimed {sc:.4f})")
