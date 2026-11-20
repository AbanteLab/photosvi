#!/usr/bin/env python3
"""Supplementary figure: the cohort model as a probabilistic graphical model.

    python scripts/paper_fig_pgm.py --out diagnostics/fig_pgm.pdf

Moved out of Figure 1 and into the supplement: the main text needs the workflow, and
a reader who wants the exact conditional structure wants it complete rather than
compressed into half a figure. Complete now includes the condition-specific tonic
response, which Figure 1 never showed.

The single-trace model is drawn as what it is -- the same graph with one animal, one
session and no population level -- rather than as a separate diagram.

Plate membership is the load-bearing part: a node inside a plate is a claim that it
is replicated over it. So the animal effects sit above the session plate, and the
kernel and mark hyperparameters sit outside every plate, being shared by the cohort.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

INK, MUTED = "#2b2b2b", "#8a8a8a"
LATENT, OBSERVED, DETERM = "#ffffff", "#c5c5c5", "#ededed"
ACCENT, TONIC = "#3b6ea5", "#c2622d"
W = 1.90

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="diagnostics/fig_pgm.pdf")
ap.add_argument("--dpi", type=int, default=200)
a = ap.parse_args()

fig = plt.figure(figsize=(10.4, 7.2))
ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
ax.set_xlim(0, W)
ax.set_ylim(-0.26, 1.02)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)


def arrow(p, q, *, color=INK, lw=1.0, rad=0.0, shrink=20.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=shrink, shrinkB=shrink, zorder=1))


def node(x, y, label, *, r=0.046, fc=LATENT, fs=9.0, ec=INK):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=1.1, zorder=2))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=INK, zorder=3)


def dnode(x, y, label, *, w=0.115, h=0.085, fs=9.0, ec=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.022",
                                fc=DETERM, ec=ec, lw=1.0, zorder=2))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=INK, zorder=3)


def plate(x0, y0, x1, y1, label, *, fc="none", alpha=1.0, ec=INK, dx=0.022):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=1.0,
                           alpha=alpha, zorder=0))
    if label:
        ax.text(x0 + dx, y0 + 0.020, label, ha="left", va="bottom", fontsize=8.4,
                color=MUTED, style="italic", zorder=1)


# ---------------------------------------------------------------- population
ax.text(0.170, 1.005, "population", fontsize=8.8, color=MUTED, style="italic",
        ha="center", va="top")
node(0.082, 0.905, r"$\mu_{\beta_0}$", fs=8.4)
node(0.230, 0.905, r"$\sigma_{\beta_0}$", fs=8.4)
node(0.082, 0.790, r"$\sigma_{\rm an}$", fs=8.0)
node(0.230, 0.790, r"$\sigma_{{\rm an}\times d}$", fs=6.4)
node(0.156, 0.575, r"$\sigma_{\psi}$", fs=8.4, ec=TONIC)

# per-condition: the rate mean AND the tonic response
plate(0.014, 0.360, 0.306, 0.530, r"condition  $d = 1 \ldots D$")
node(0.096, 0.462, r"$\mu_d$", fs=9.0)
node(0.232, 0.462, r"$\bar\psi_d$", fs=9.0, ec=TONIC)

# ------------------------------------------------------------- nested plates
plate(0.375, 0.150, 1.875, 0.960, r"animal  $a = 1 \ldots N$")
plate(0.430, 0.200, 1.825, 0.815, "", fc=ACCENT, alpha=0.05, ec="none")
plate(0.430, 0.200, 1.825, 0.815, r"session  $j = 1 \ldots S$")
plate(0.905, 0.290, 1.795, 0.760, r"time  $t = 1 \ldots T$")

node(0.500, 0.885, r"$z^{\rm an}_a$", fs=8.2)
node(0.680, 0.885, r"$z^{\rm inj}_a$", fs=8.2)

node(0.520, 0.590, r"$z^{\rm cell}_{a(j)d(j)}$", r=0.054, fs=5.6)
node(0.520, 0.330, r"$z^{\psi}_j$", fs=7.8, ec=TONIC)

dnode(0.730, 0.690, r"$\beta_0^{(j)}$", w=0.130, h=0.092, fs=9.0)
dnode(0.730, 0.500, r"$\beta_{\rm inj}^{(j)}$", w=0.145, h=0.092, fs=8.6)
dnode(0.730, 0.360, r"$\psi^{(j)}$", w=0.118, h=0.085, fs=9.0, ec=TONIC)
node(0.730, 0.248, r"$b_j, \delta_j$", fc=DETERM, r=0.054, fs=6.6)
ax.text(0.800, 0.248, "per-session MAP:\nlevel and bleaching", ha="left", va="center",
        fontsize=6.6, color=MUTED, style="italic", linespacing=1.4)

# ------------------------------------------------------------------ time level
dnode(1.010, 0.645, r"$\lambda_t$", w=0.105, h=0.085, fs=9.2)
node(1.170, 0.645, r"$s_t$", fs=9.5)
node(1.170, 0.475, r"$a_t$", fs=9.5)
dnode(1.355, 0.560, r"$c_t$", w=0.105, h=0.085, fs=9.2)
dnode(1.355, 0.370, r"$x_t$", w=0.105, h=0.085, fs=9.2, ec=TONIC)
node(1.640, 0.465, r"$y_t$", fc=OBSERVED, r=0.050, fs=9.5)
# The emission is autoregressive, so the time plate is NOT conditionally independent:
# y_t carries the residual of the bins before it. Drawn as a recurrence on the observed
# node rather than by unrolling the plate, which is the usual compromise when a plate
# holds a time series. Without it the graph asserts an independence the model does not
# have, which is exactly the kind of thing a plate diagram exists to settle.
ax.add_patch(FancyArrowPatch((1.640, 0.540), (1.712, 0.465),
                             connectionstyle="arc3,rad=1.7", arrowstyle="-|>",
                             mutation_scale=9, lw=1.0, color=INK,
                             shrinkA=2, shrinkB=2, zorder=1))
ax.text(1.757, 0.545, r"AR($p$)", fontsize=6.6, color=MUTED, ha="left", va="center")

# shared across the cohort: outside every plate
node(1.180, 0.060, r"$\mu_a,\sigma_a$", fc=DETERM, r=0.056, fs=6.8)
node(1.430, 0.060, r"$h$", fc=DETERM, r=0.042, fs=9.2)
ax.text(1.305, -0.012, "shared across the cohort", ha="center", va="center",
        fontsize=7.0, color=MUTED, style="italic")

for p, q, rad, col in (
        ((0.082, 0.905), (0.730, 0.690), -0.34, INK),      # mu_b0     -> beta0
        ((0.230, 0.905), (0.500, 0.885), -0.10, INK),      # sigma_b0  -> z_an
        ((0.082, 0.790), (0.680, 0.885), -0.36, INK),      # sigma_an  -> z_inj
        ((0.230, 0.790), (0.520, 0.590), -0.10, INK),      # sigma_anxd-> z_cell
        ((0.156, 0.575), (0.520, 0.330), -0.12, TONIC),    # sigma_psi -> z_psi
        ((0.096, 0.462), (0.730, 0.500), 0.12, INK),       # mu_d      -> beta_inj
        ((0.232, 0.462), (0.730, 0.360), 0.10, TONIC),     # psibar_d  -> psi
        ((0.500, 0.885), (0.730, 0.690), 0.0, INK),
        ((0.680, 0.885), (0.730, 0.500), 0.0, INK),
        ((0.520, 0.590), (0.730, 0.500), 0.0, INK),
        ((0.520, 0.330), (0.730, 0.360), 0.0, TONIC),
        ((0.730, 0.248), (0.730, 0.360), 0.0, MUTED),
        ((0.730, 0.690), (1.010, 0.645), 0.0, INK),
        ((0.730, 0.500), (1.010, 0.645), 0.0, INK),
        ((0.730, 0.360), (1.355, 0.370), -0.05, TONIC),
        ((1.010, 0.645), (1.170, 0.645), 0.0, INK),
        ((1.170, 0.645), (1.355, 0.560), 0.0, INK),
        ((1.170, 0.475), (1.355, 0.560), 0.0, INK),
        ((1.355, 0.560), (1.640, 0.465), 0.0, INK),
        ((1.355, 0.370), (1.640, 0.465), 0.0, TONIC),
        ((1.180, 0.060), (1.170, 0.475), 0.0, MUTED),
        ((1.430, 0.060), (1.355, 0.560), -0.08, MUTED),
):
    arrow(p, q, rad=rad, color=col)

ax.text(1.805, 0.895,
        "the single-trace model is this graph with $N = 1$, one session,\n"
        r"and $(\beta_0,\ \beta_{\rm inj})$ given priors directly instead of a "
        "population",
        ha="right", va="center", fontsize=8.0, color=ACCENT, linespacing=1.6,
        style="italic")
ax.text(1.805, 0.800, "orange: the condition-specific tonic response",
        ha="right", va="center", fontsize=7.8, color=TONIC, style="italic")

# --------------------------------------------------------------------- legend
lx, ly = 0.040, 0.265
ax.text(lx, ly + 0.020, "legend", fontsize=8.4, color=MUTED, style="italic")
node(lx + 0.032, ly - 0.055, "", r=0.030)
ax.text(lx + 0.085, ly - 0.055, "latent", fontsize=8.0, color=INK, va="center")
node(lx + 0.032, ly - 0.145, "", r=0.030, fc=OBSERVED)
ax.text(lx + 0.085, ly - 0.145, "observed", fontsize=8.0, color=INK, va="center")
dnode(lx + 0.032, ly - 0.235, "", w=0.064, h=0.052)
ax.text(lx + 0.085, ly - 0.235, "deterministic /\npoint estimate", fontsize=8.0,
        color=INK, va="center", linespacing=1.4)

ax.text(0.95, -0.115,
        r"$\beta_0^{(j)} = \mu_{\beta_0} + \sigma_{\beta_0} z^{\rm an}_{a(j)}$"
        "        "
        r"$\beta_{\rm inj}^{(j)} = \mu_{d(j)} + \sigma_{\rm an} z^{\rm inj}_{a(j)} "
        r"+ \sigma_{{\rm an}\times d} z^{\rm cell}_{a(j)d(j)}$",
        ha="center", va="center", fontsize=8.8, color=INK)
ax.text(0.95, -0.185,
        r"$\psi^{(j)} = \bar\psi_{d(j)} + \sigma_{\psi} \odot z^{\psi}_j$ "
        r"with $\psi = (\alpha, \rho, \gamma)$;  $b$ and $\delta$ per session, unpooled."
        "   A variance component is included only when the design identifies it.",
        ha="center", va="center", fontsize=7.8, color=MUTED)

out = Path(a.out)
fig.savefig(out, transparent=True, bbox_inches="tight")  # PDF: no white patch over LaTeX labels
fig.savefig(out.with_suffix(".png"), dpi=a.dpi, facecolor="white", bbox_inches="tight")
print(f"wrote {out} and {out.with_suffix('.png')}")
