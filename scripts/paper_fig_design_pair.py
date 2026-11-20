#!/usr/bin/env python3
"""Figs 3 and 4 of the paper: the crossed and the uncrossed (incomplete) design side by side.

    python scripts/paper_fig_design_pair.py --which shrinkage --out fig_designs_shrinkage.pdf
    python scripts/paper_fig_design_pair.py --which posteriors --out fig_designs_posteriors.pdf

Layout of 2026-10-08, replacing paper_fig_crossed.py and paper_fig_designs.py
in the paper (both kept for their other uses):

  shrinkage   a  crossed:   per-session beta_inj estimates against their simulated values
              b  uncrossed: the same
  posteriors  a  crossed: the null distribution of the contrast against the width the
                 posterior reports
              b  both designs: the two null distributions on one axis, with the spread of
                 an unbiased estimator under the uncrossed design marked

Both designs come from the same config at the null (delta = 0) and have the same number of
recordings; the uncrossed one has twice the animals, one condition each. The variance
components are nuisance parameters and are not drawn.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from photosvi.experiments.cohort import spec_from_config  # noqa: E402

INK, MUTED, GRID, ACCENT = "#2b2b2b", "#8a8a8a", "#d9d9d9", "#b03030"
PALETTE = ("#9a9a9a", "#3b6ea5", "#c2622d")
CROSSED_C, UNCROSSED_C = "#3b6ea5", "#c2622d"

ap = argparse.ArgumentParser()
ap.add_argument("--which", required=True, choices=("shrinkage", "posteriors", "both", "main"))
ap.add_argument("--crossed", required=True, help="directory of crossed hier_*.json")
ap.add_argument("--uncrossed", required=True, help="directory of uncrossed hier_*.json")
ap.add_argument("--config", default="configs/cohort_power_48.yaml")
ap.add_argument("--tail", type=int, default=6,
                help="convergence probes the contrast is averaged over")
ap.add_argument("--out", required=True)
ap.add_argument("--report", default=None,
                help="cohort_report.py --json output, for the power panel of --which main")
ap.add_argument("--macros", default=None, help="write the caption's numbers here")
ap.add_argument("--dpi", type=int, default=200)
ap.add_argument("--min-cohorts", type=int, default=3,
                help="refuse to draw with fewer null cohorts per design (1 for a smoke test)")
a = ap.parse_args()


def load(d: str) -> list[dict]:
    r = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/hier_*.json"))]
    return [x for x in r if x["delta"] == 0.0 and x.get("estimand_scale", 0) > 0]


cr, un = load(a.crossed), load(a.uncrossed)
if len(cr) < a.min_cohorts or len(un) < a.min_cohorts:
    raise SystemExit(f"need both designs: {len(cr)} crossed, {len(un)} uncrossed cohorts")
DRUGS = tuple(cr[0]["tonic_posterior"].keys())
COLOR = dict(zip(DRUGS, PALETTE))


def drugs_of(rows: list[dict], design: str) -> list[int]:
    cfg = yaml.safe_load(open(a.config))
    cfg["cohort"] |= dict(design=design, n_animals=rows[0]["design"]["n_animals"])
    _, d = spec_from_config(cfg, 0.0).indices()
    if len(d) != len(rows[0]["session_beta_inj"]):
        raise SystemExit(f"{design}: spec gives {len(d)} sessions, runs have "
                         f"{len(rows[0]['session_beta_inj'])}")
    return list(d)


def contrast(r: dict) -> float:
    """The contrast averaged over the last probes, as the tables read it."""
    tr = [t["contrasts"][-1] for t in r.get("probe_trace", []) if "contrasts" in t]
    return st.mean(tr[-a.tail:]) if tr else r["estimand_loc"]


def style(ax, letter):
    ax.grid(alpha=0.25, lw=0.7, color=GRID)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.tick_params(labelsize=8.5, colors="#555555")
    ax.text(-0.14, 1.08, letter, transform=ax.transAxes, fontsize=14, weight="bold",
            color=INK, va="top")


def norm(x, s):
    return np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))


macros = {}


def draw_shrinkage(axes):
    global macros
    allv = [b for rows in (cr, un) for r in rows
            for b in r["session_beta_inj_true"] + r["session_beta_inj"]]
    lim = (min(allv) - 0.1, max(allv) + 0.1)
    for ax, rows, design, tag in ((axes[0], cr, "crossed", "Cr"),
                                  (axes[1], un, "uncrossed", "Un")):
        dos = drugs_of(rows, design)
        for d, name in enumerate(DRUGS):
            tx = [b for r in rows for i, b in enumerate(r["session_beta_inj_true"]) if dos[i] == d]
            fy = [b for r in rows for i, b in enumerate(r["session_beta_inj"]) if dos[i] == d]
            ax.scatter(tx, fy, s=7, alpha=0.5, lw=0, color=COLOR[name], label=name)
        x = [b for r in rows for b in r["session_beta_inj_true"]]
        y = [b for r in rows for b in r["session_beta_inj"]]
        sl, ic = np.polyfit(x, y, 1)
        ax.plot(lim, lim, lw=1.0, color=INK, ls=(0, (4, 3)), zorder=1)
        ax.plot(np.array(lim), sl * np.array(lim) + ic, lw=1.4, color=ACCENT, zorder=2)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(r"simulated per-session $\beta_{\rm inj}$", fontsize=9)
        ax.set_ylabel(r"estimated $\beta_{\rm inj}$", fontsize=9)
        macros[f"ppFig{tag}Slope"] = f"{sl:.2f}"
        macros[f"ppFig{tag}N"] = str(len(rows))
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left", handletextpad=0.3)


def draw_posteriors(axes):
    global macros
    # a: crossed, the realised spread against the width the posterior reports
    ax = axes[0]
    est = np.array([contrast(r) for r in cr])
    sc = float(st.median(r.get("estimand_scale_avg", r["estimand_scale"]) for r in cr))
    emp = float(st.stdev(est))
    cover = sum(1 for r in cr if abs(contrast(r)) <= 1.645 * r.get("estimand_scale_avg", r["estimand_scale"])) / len(cr)
    ax.hist(est, bins=10, color="#c9d6e4", edgecolor="#5b7fa6", lw=0.8, density=True)
    w = 4 * max(sc, emp)
    g = np.linspace(-w, w, 400)
    ax.plot(g, norm(g, sc), lw=1.8, color=ACCENT, label=f"reported  s.d. {sc:.3f}")
    ax.plot(g, norm(g, emp), lw=1.8, color=INK, ls=(0, (5, 2)),
            label=f"across cohorts  s.d. {emp:.3f}")
    ax.axvline(0.0, lw=1.0, color=MUTED)
    ax.set_xlabel("estimated contrast (true value 0)", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", handletextpad=0.5)
    macros |= {"ppFigCrN": str(len(cr)), "ppFigCrRatio": f"{emp / sc:.2f}",
               "ppFigCrCover": f"{100 * cover:.0f}\\%", "ppFigTail": str(a.tail)}

    # b: both designs' null distributions, and the unbiased-estimator spread
    ax = axes[1]
    c = yaml.safe_load(open(a.config))["cohort"]
    per_group = un[0]["design"]["n_animals"] // 2
    floor = c["sigma_animal"] * math.sqrt(2 / per_group)
    s_un = float(st.stdev([contrast(r) for r in un]))
    w = 4 * max(emp, s_un, floor)
    g = np.linspace(-w, w, 400)
    for s, col, label in ((emp, CROSSED_C, "crossed"), (s_un, UNCROSSED_C, "incomplete")):
        ax.plot(g, norm(g, s), lw=2.0, color=col, label=f"{label}  s.d. {s:.3f}")
        ax.fill_between(g, 0, norm(g, s), color=col, alpha=0.12)
    for sign in (-1, 1):
        ax.axvline(sign * floor, lw=1.2, color=ACCENT, ls=(0, (4, 3)))
    ax.axvline(0.0, lw=1.0, color=MUTED)
    ax.set_xlabel("estimated contrast (true value 0)", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", handletextpad=0.5)
    sd_un = float(st.median(r.get("estimand_scale_avg", r["estimand_scale"]) for r in un))
    cov_un = sum(1 for r in un if abs(contrast(r)) <= 1.645 * r.get("estimand_scale_avg", r["estimand_scale"])) / len(un)
    macros |= {"ppFigUnN": str(len(un)), "ppFigUnAnimals": str(un[0]["design"]["n_animals"]),
               "ppFigCrSpread": f"{emp:.3f}", "ppFigUnSpread": f"{s_un:.3f}",
               "ppFigUnRatio": f"{s_un / sd_un:.2f}", "ppFigUnCover": f"{100 * cov_un:.0f}\\%",
               "ppFigFloor": f"{floor:.3f}", "ppFigUnWidthRatio": f"{sd_un / sc:.1f}"}


# 'both': one four-panel figure, shrinkage on top (a, b) and the null
# distributions below (c, d), so the design comparison is one display item.
if a.which == "main":
    # Fig. 3: power curves at medium SNR (a; they replace Table 2) and the
    # null distributions of the contrast in the two designs (b). The shrinkage scatters go
    # to the SI (--which shrinkage) and the crossed-design histogram is dropped.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paper_power_curves import draw_power
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.90, bottom=0.14, wspace=0.28)
    for ax, letter in zip(axes, "ab"):
        style(ax, letter)
    draw_power(axes[0], json.load(open(a.report)))
    _tmp = plt.figure()
    draw_posteriors([_tmp.add_subplot(111), axes[1]])
    plt.close(_tmp)
elif a.which == "both":
    fig, grid = plt.subplots(2, 2, figsize=(9.6, 8.0))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.95, bottom=0.07, wspace=0.28, hspace=0.32)
    for ax, letter in zip(grid.flat, "abcd"):
        style(ax, letter)
    draw_shrinkage(grid[0])
    draw_posteriors(grid[1])
else:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.90, bottom=0.14, wspace=0.28)
    for ax, letter in zip(axes, "ab"):
        style(ax, letter)
    (draw_shrinkage if a.which == "shrinkage" else draw_posteriors)(axes)

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, transparent=True, bbox_inches="tight")  # PDF: no white patch over LaTeX labels
fig.savefig(out.with_suffix(".png"), dpi=a.dpi, facecolor="white", bbox_inches="tight")
print(f"wrote {out} ({len(cr)} crossed, {len(un)} uncrossed): "
      + ", ".join(f"{k}={v}" for k, v in macros.items()))
if a.macros:
    Path(a.macros).write_text(
        f"% Generated by scripts/paper_fig_design_pair.py --which {a.which} -- do not edit.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
