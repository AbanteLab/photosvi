#!/usr/bin/env python3
"""Power curves of the cohort benchmark (they replace the power tables).

    python scripts/paper_power_curves.py --report mid.json --out fig.pdf
    python scripts/paper_power_curves.py --report low.json high.json --titles "low SNR" "high SNR" --out sfig.pdf

Each report is what scripts/cohort_report.py --json writes; power at each delta is the
fraction of cohorts whose statistic exceeds the arm's threshold calibrated on the null
cohorts, exactly as in the tables. Bars: Wilson 95% intervals over cohorts. Colour is the
estimate (hierarchical, PhotoSVI two-stage, z-peaks counts, GuPPy counts), line style the
test (unpaired, paired, RM ANOVA). `draw_power(ax, report)` is reused by the design figure.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

ARMS = [("hier", "hierarchical", "#b03030", "-"),
        ("ts_session", "PhotoSVI two-stage, unpaired", "#3b6ea5", "-"),
        ("ts_animal", "PhotoSVI two-stage, paired", "#3b6ea5", "--"),
        ("ts_anova", "PhotoSVI two-stage, RM ANOVA", "#3b6ea5", ":"),
        ("zp_session", "z-peaks counts, unpaired", "#6a8d3a", "-"),
        ("zp_animal", "z-peaks counts, paired", "#6a8d3a", "--"),
        ("zp_anova", "z-peaks counts, RM ANOVA", "#6a8d3a", ":"),
        ("gp_session", "GuPPy counts, unpaired", "#c2622d", "-"),
        ("gp_animal", "GuPPy counts, paired", "#c2622d", "--"),
        ("gp_anova", "GuPPy counts, RM ANOVA", "#c2622d", ":")]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def power(report: dict, key: str):
    thr = report["thresholds"].get(key)
    if thr is None or not math.isfinite(thr):
        return None
    deltas = sorted({r["delta"] for r in report["rows"]})
    out = []
    for d in deltas:
        v = [r[f"{key}_t"] for r in report["rows"]
             if r["delta"] == d and f"{key}_t" in r and math.isfinite(r[f"{key}_t"])]
        if not v:
            continue
        k = sum(x > thr for x in v)
        out.append((d, k / len(v), *wilson(k, len(v))))
    return np.array(out) if out else None


def style(ax):
    ax.grid(alpha=0.25, lw=0.7, color="#d9d9d9")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.tick_params(labelsize=8.5, colors="#555555")


def draw_power(ax, report: dict, legend: bool = True):
    style(ax)
    for key, label, col, ls in ARMS:
        p = power(report, key)
        if p is None:
            continue
        lw = 2.0 if key == "hier" else 1.3
        ax.errorbar(p[:, 0], 100 * p[:, 1], yerr=[100 * (p[:, 1] - p[:, 2]), 100 * (p[:, 3] - p[:, 1])],
                    color=col, ls=ls, lw=lw, marker="o", ms=3 if key != "hier" else 4,
                    capsize=0, elinewidth=0.6, label=label)
    ax.axhline(100 * report.get("alpha", 0.05), color="#8a8a8a", lw=0.8, ls=(0, (2, 2)))
    ax.set_ylim(-2, 102)
    ax.set_xlabel(r"true $\Delta$ (log rate ratio)", fontsize=9)
    ax.set_ylabel("power (%)", fontsize=9)
    if legend:
        ax.legend(fontsize=7, frameon=False, loc="upper left", handlelength=2.6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", nargs="+", required=True)
    ap.add_argument("--titles", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = len(a.report)
    fig, axes = plt.subplots(1, n, figsize=(4.9 * n, 4.0), squeeze=False)
    for i, (ax, path) in enumerate(zip(axes[0], a.report)):
        draw_power(ax, json.load(open(path)), legend=(i == 0))
        if n > 1:
            ax.text(-0.14, 1.06, "ab"[i], transform=ax.transAxes, fontsize=14, weight="bold",
                    color="#2b2b2b", va="top")
            if a.titles:
                ax.text(0.98, 0.04, a.titles[i], transform=ax.transAxes, ha="right",
                        fontsize=9, color="#555555")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, transparent=True, bbox_inches="tight")  # PDF: no white patch over LaTeX labels
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")
