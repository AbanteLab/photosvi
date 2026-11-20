#!/usr/bin/env python3
"""S4 Fig: does the relaxation temperature change the answer?

    python scripts/paper_fig_tau.py --runs runs_tau --out diagnostics/fig_tau.pdf

Four panels, all against tau, with the two ways of scoring the event factor drawn
against each other:

  A  the estimated log rate ratio against the count MLE -- the quantity everything
     else in the paper rests on
  B  the guide's total mass against the true number of onsets: whether the events
     are being counted or smeared
  C  detection, which has no reason to depend on tau and is the control
  D  where the fit stopped, since a temperature that makes the geometry hard shows
     up as a fit that never settles

The value used throughout the paper is marked. The point of the figure is to show
that it was chosen on evidence rather than by default -- and, if the analytic scoring
is flatter in tau than the relaxed one, to show why that configuration was preferred.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#2b2b2b", "#8a8a8a", "#d9d9d9"
STYLE = {"analytic": ("#3b6ea5", "o", "analytic Bernoulli divergence"),
         "relaxed": ("#c2622d", "s", "relaxed densities")}
SHIPPED = 0.5

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs_tau")
ap.add_argument("--out", default="diagnostics/fig_tau.pdf")
ap.add_argument("--dpi", type=int, default=200)
a = ap.parse_args()

rows = [json.load(open(f)) for f in sorted(glob.glob(f"{a.runs}/*.json"))]
if not rows:
    raise SystemExit(f"no results in {a.runs}")
taus = sorted({r["tau"] for r in rows})


def agg(kl, fn):
    """mean and SD over seeds at each tau, skipping diverged cells."""
    m, s, n = [], [], []
    for t in taus:
        v = [fn(r) for r in rows
             if r["kl"] == kl and r["tau"] == t and not r["diverged"]
             and fn(r) is not None]
        v = [x for x in v if x == x]                      # drop NaN from broken cells
        m.append(st.mean(v) if v else np.nan)
        s.append(st.pstdev(v) if len(v) > 1 else 0.0)
        n.append(len(v))
    return np.array(m), np.array(s), n


fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2))
fig.subplots_adjust(left=0.085, right=0.975, top=0.925, bottom=0.105,
                    hspace=0.40, wspace=0.27)
for ax, letter in zip(axes.ravel(), "ABCD", strict=True):
    ax.grid(alpha=0.25, lw=0.7, color=GRID)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.tick_params(labelsize=8.5, colors="#555555")
    ax.set_xscale("log")
    ax.set_xlabel(r"relaxation temperature  $\tau$", fontsize=9)
    ax.axvline(SHIPPED, lw=1.0, color=INK, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.145, 1.11, letter, transform=ax.transAxes, fontsize=14,
            weight="bold", color=INK, va="top")

panels = [
    (axes[0, 0], "error in the log rate ratio",
     r"$\hat\beta_{\rm inj}$ $-$ count MLE",
     lambda r: r["beta_inj"] - r["mle"], 0.0),
    (axes[0, 1], "is the guide counting events or smearing them?",
     r"guide mass $\sum_t q_t$ / true onsets",
     lambda r: r["q_mass"] / r["n_true"], 1.0),
    (axes[1, 0], "detection: the control, with no reason to depend on $\\tau$",
     r"best $F_1$", lambda r: r["best_f1"], None),
    (axes[1, 1], "did the fit settle?", "steps to the stopping criterion",
     lambda r: r["stopped_at"] if r["stopped_at"] else np.nan, None),
]
for ax, title, ylab, fn, ref in panels:
    for kl, (col, mk, lab) in STYLE.items():
        m, s, n = agg(kl, fn)
        ax.errorbar(taus, m, yerr=s, color=col, marker=mk, ms=5, lw=1.5, capsize=3,
                    elinewidth=1.0, label=lab, zorder=3)
    if ref is not None:
        ax.axhline(ref, lw=1.0, color=MUTED, ls=(0, (2, 2)), zorder=1)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title, fontsize=9, color=INK, pad=7)

axes[0, 0].legend(fontsize=7.8, frameon=False, loc="best", handletextpad=0.5)
axes[0, 0].annotate("used throughout", xy=(SHIPPED, axes[0, 0].get_ylim()[1]),
                    xytext=(SHIPPED * 1.15, axes[0, 0].get_ylim()[1]),
                    fontsize=7.4, color=INK, va="top")

n_div = sum(1 for r in rows if r["diverged"])
fig.text(0.5, 0.022,
         f"{len(rows)} fits, {len(taus)} temperatures x 2 scorings x "
         f"{len({r['seed'] for r in rows})} seeds"
         + (f"; {n_div} diverged and are excluded" if n_div else
            "; none diverged")
         + ". Bars are standard deviations across seeds.",
         ha="center", fontsize=7.8, color=MUTED, style="italic")

out = Path(a.out)
fig.savefig(out, facecolor="white", bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=a.dpi, facecolor="white", bbox_inches="tight")
print(f"wrote {out} ({len(rows)} cells, {n_div} diverged)")
for kl in STYLE:
    m, s, n = agg(kl, lambda r: r["beta_inj"] - r["mle"])
    rng = np.nanmax(m) - np.nanmin(m)
    print(f"  {kl:>8}: error in beta_inj spans {rng:.3f} across tau "
          f"(n per tau: {n})")
