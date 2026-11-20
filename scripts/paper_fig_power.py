#!/usr/bin/env python3
"""Paper figure: what each analysis can conclude, against the number of animals.

    python scripts/paper_fig_power.py --summary <grid>/summary.json \
        --out .../diagnostics/fig_power.pdf

Three panels on one axis, the number of animals, which is the axis an experimenter plans
on. (A) power at a matched false-positive rate, one line per analysis. (B) the width of the
interval each analysis reports. (C) coverage of the hierarchical interval against its
nominal level, which is the quantity the paper is honest about rather than the one it would
prefer.

The real cohort's own size is marked, so a reader can see where this study's recordings sit
on a curve rather than having to hold two sections in mind at once.
"""
from __future__ import annotations

import argparse, json, pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

ARMS = [("gp_session", "GuPPy counts, per session", "#eb6834", ":"),
        ("gp_animal", "GuPPy counts, per animal", "#eb6834", "-"),
        ("ts_session", "two-stage, per session", "#8a8f98", ":"),
        ("ts_animal", "two-stage, per animal", "#8a8f98", "-"),
        ("hier", "hierarchical", "#2a78d6", "-")]

ap = argparse.ArgumentParser()
ap.add_argument("--summary", required=True)
ap.add_argument("--out", default="diagnostics/fig_power.pdf")
ap.add_argument("--delta", type=float, default=0.4, help="the effect size the power panel shows")
ap.add_argument("--real-n", type=int, default=5, help="the recordings' own cohort size")
a = ap.parse_args()

S = [s for s in json.load(open(a.summary)) if s["design"] == "crossed"]
if not S:
    raise SystemExit("no crossed cells in the summary")
S.sort(key=lambda s: s["n_animals"])
ns = [s["n_animals"] for s in S]

fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
fig.patch.set_facecolor("white")

def power(cell, arm):
    """Fraction of that cell's delta cohorts the arm calls, at its calibrated threshold."""
    t = cell["thresholds"].get(arm)
    rows = [r for r in cell["rows"] if abs(r["delta"] - a.delta) < 1e-9 and f"{arm}_t" in r]
    if t is None or not rows:
        return None
    return sum(1 for r in rows if abs(r[f"{arm}_t"]) >= t) / len(rows)

ax = axes[0]
for arm, lab, col, ls in ARMS:
    ys = [power(s, arm) for s in S]
    xs = [n for n, y in zip(ns, ys) if y is not None]
    ys = [y for y in ys if y is not None]
    if xs:
        ax.plot(xs, ys, ls, color=col, marker="o", ms=4, lw=1.8, label=lab)
ax.set_ylim(-0.03, 1.03)
ax.set_ylabel(f"power at a matched {S[0]['alpha']:.0%} false-positive rate".replace("%", "%"))
ax.set_title(f"A   Detecting an effect of {a.delta:g}", loc="left", fontweight="bold",
             fontsize=11)
ax.legend(fontsize=7.5, loc="lower right", framealpha=.95)

ax = axes[1]
for arm, lab, col, ls in ARMS:
    ys = []
    for s in S:
        w = [r[f"{arm}_scale"] for r in s["rows"]
             if f"{arm}_scale" in r and abs(r["delta"] - a.delta) < 1e-9]
        ys.append(float(np.mean(w)) if w else None)
    xs = [n for n, y in zip(ns, ys) if y is not None]
    yy = [y for y in ys if y is not None]
    if xs:
        ax.plot(xs, yy, ls, color=col, marker="o", ms=4, lw=1.8)
ax.set_ylabel("reported interval half-width")
ax.set_title("B   What each analysis claims to know", loc="left", fontweight="bold",
             fontsize=11)

ax = axes[2]
cov = []
for s in S:
    rows = [r for r in s["rows"] if "hier_cover" in r]
    cov.append(float(np.mean([r["hier_cover"] for r in rows])) if rows else None)
xs = [n for n, c in zip(ns, cov) if c is not None]
yy = [c for c in cov if c is not None]
if xs:
    ax.plot(xs, yy, "-", color="#2a78d6", marker="o", ms=5, lw=2)
ax.axhline(0.90, color="#b3453a", ls="--", lw=1.2)
ax.text(ns[-1], 0.905, "nominal 90%", ha="right", va="bottom", fontsize=8.5, color="#b3453a")
ax.set_ylim(0, 1.03)
ax.set_ylabel("coverage of the hierarchical interval")
ax.set_title("C   And whether it is entitled to", loc="left", fontweight="bold", fontsize=11)

for ax in axes:
    ax.set_xlabel("animals in the cohort")
    ax.set_xticks(ns)
    ax.axvline(a.real_n, color="#555", lw=1.0, ls="-", alpha=.35, zorder=0)
    ax.grid(alpha=.15, lw=.7)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
axes[0].text(a.real_n, 1.0, " this study", fontsize=8, color="#555", va="top")

out = pathlib.Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out, facecolor="white", bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=150, facecolor="white", bbox_inches="tight")
print(f"  wrote {out} over n = {ns}")
