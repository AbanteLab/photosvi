#!/usr/bin/env python3
"""Paper figure: what each analysis reports, cohort by cohort, against the truth.

    python scripts/paper_fig_benchmark.py --report <cell>/report.json \
        --out .../diagnostics/fig_benchmark.pdf

One cell of the cohort grid, read from the JSON `scripts/cohort_report.py` writes, so
this figure and the tables built from the same file cannot disagree.

(A) every cohort's point estimate, one dot per cohort, grouped by analysis and by true
effect size, with the truth as a rule. This is the panel that shows attenuation for what
it is: a cloud that sits below its rule and does not widen to reach it.

(B) power at each arm's own calibrated threshold, which is the same quantity the table
reports -- drawn because a reader comparing five analyses across effect sizes reads bars
faster than a grid of percentages, not because it says anything new.

A figure of one cell says nothing about how any of this scales with the number of
animals; `paper_fig_power.py` is the one that sweeps that axis.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

# Same keys, labels and colours as paper_fig_power.py, so the two figures read as one
# family: the counting arms warm, the per-session model grey, the hierarchy blue.
ARMS = [("gp_session", "GuPPy\nper session", "#eb6834"),
        ("gp_animal", "GuPPy\nper animal", "#eb6834"),
        ("ts_session", "two-stage\nper session", "#8a8f98"),
        ("ts_animal", "two-stage\nper animal", "#8a8f98"),
        ("hier", "hierarchical", "#2a78d6")]

ap = argparse.ArgumentParser()
ap.add_argument("--report", required=True)
ap.add_argument("--out", default="diagnostics/fig_benchmark.pdf")
ap.add_argument("--macros", default=None)
a = ap.parse_args()

rep = json.loads(pathlib.Path(a.report).read_text())
rows, thresh, alpha = rep["rows"], rep["thresholds"], rep["alpha"]
deltas = sorted({r["delta"] for r in rows})
if not deltas:
    raise SystemExit("the report holds no cohorts")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0),
                         gridspec_kw={"width_ratios": [1.55, 1.0]})
fig.patch.set_facecolor("white")

# ---------------------------------------------------------------- (A) the estimates
ax = axes[0]
width = 0.8 / max(len(deltas), 1)
rng = np.random.default_rng(0)        # jitter only, fixed so the figure is reproducible
for i, (key, lab, col) in enumerate(ARMS):
    for j, d in enumerate(deltas):
        v = [r[f"{key}_est"] for r in rows
             if r["delta"] == d and f"{key}_est" in r
             and math.isfinite(r[f"{key}_est"])]
        if not v:
            continue
        x = i + (j - (len(deltas) - 1) / 2) * width
        ax.scatter(x + rng.uniform(-width / 5, width / 5, len(v)), v,
                   s=11, color=col, alpha=0.55, linewidths=0, zorder=3)
        ax.plot([x - width / 2.4, x + width / 2.4], [np.mean(v)] * 2,
                color=col, lw=1.8, zorder=4)
        # The truth for this group, drawn as a short rule at the group's own position so
        # the gap to the mean is read vertically and not across the panel.
        ax.plot([x - width / 2.4, x + width / 2.4], [d] * 2,
                color="#222222", lw=1.0, ls="--", zorder=5)

ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels([lab for _, lab, _ in ARMS], fontsize=8)
ax.set_ylabel(r"reported contrast $\hat\Delta$")
ax.axhline(0.0, color="#cccccc", lw=0.8, zorder=1)
ax.set_title("A   every cohort's estimate, with its true value dashed",
             fontsize=9, loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---------------------------------------------------------------- (B) calibrated power
ax = axes[1]
nd = len(deltas)
bw = 0.8 / max(nd, 1)
for j, d in enumerate(deltas):
    ys, xs = [], []
    for i, (key, _, _) in enumerate(ARMS):
        t = thresh.get(key)
        v = [r[f"{key}_t"] for r in rows
             if r["delta"] == d and f"{key}_t" in r and math.isfinite(r[f"{key}_t"])]
        xs.append(i + (j - (nd - 1) / 2) * bw)
        ys.append(sum(x > t for x in v) / len(v)
                  if v and isinstance(t, (int, float)) and math.isfinite(t) else np.nan)
    ax.bar(xs, ys, width=bw * 0.9, label=rf"$\Delta={d:.2f}$",
           color=plt.cm.viridis(0.15 + 0.7 * j / max(nd - 1, 1)), zorder=3)

ax.axhline(alpha, color="#222222", lw=1.0, ls="--", zorder=4)
ax.annotate(f"calibrated to {alpha:.0%}", (len(ARMS) - 0.5, alpha), fontsize=7,
            ha="right", va="bottom", color="#222222")
ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels([lab for _, lab, _ in ARMS], fontsize=8)
ax.set_ylabel("power at the calibrated threshold")
ax.set_ylim(0, 1)
ax.legend(fontsize=7, frameon=False)
ax.set_title("B   power, every arm at its own calibrated threshold",
             fontsize=9, loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
out = pathlib.Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")

if a.macros:
    n = {d: sum(1 for r in rows if r["delta"] == d and "hier_t" in r) for d in deltas}
    pathlib.Path(a.macros).write_text(
        "% Generated by scripts/paper_fig_benchmark.py -- do not edit.\n"
        f"\\newcommand{{\\ppBenchFigCohorts}}{{{max(n.values()) if n else 0}}}\n"
        f"\\newcommand{{\\ppBenchFigDeltas}}{{{len(deltas)}}}\n")
    print(f"wrote {a.macros}")
