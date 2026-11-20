#!/usr/bin/env python3
"""The README's two model graphics: what each model is handed, and what it returns.

    python scripts/readme_fig_modes.py --out docs/assets

Writes `single-trace.png` and `cohort.png`, one per README section. Neither carries a
title or a caption: the section heading above it already names the model and the prose
below already states its caveat, and a figure that repeats both is two more places for
the text to drift out of step with itself.

The layout is horizontal -- traces on the left, outputs on the right -- because a
README is read on a wide screen and stacking them wasted the width that the cohort
traces needed to be legible at all.
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib
import numpy as np
import yaml

from photosvi.experiments.cohort import simulate_cohort, spec_from_config

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, RULE = "#2b2b2b", "#7b7b7b", "#dcdcdc"
ONE, MANY = "#3b6ea5", "#2f8f6f"
TRACE, INJ = "#4a4a4a", "#c2622d"
# One colour per condition, none of them the injection's orange. Order matches
# cohort.DRUG_NAMES, so the legend cannot drift from what was simulated.
DRUG_COLOURS = ("#9aa5b1", "#2f6f8f", "#6a4c93")
DRUG_LABELS = ("vehicle", "opioid 1", "opioid 2")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="diagnostics/fig1_trace.npz")
ap.add_argument("--config", default="configs/cohort_comparison.yaml")
ap.add_argument("--cohort-seed", type=int, default=3)
ap.add_argument("--out", default="docs/assets")
ap.add_argument("--dpi", type=int, default=200)
a = ap.parse_args()
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)

# Left half holds the data, right half the outputs; the rule between them carries the
# model's colour, which is the only thing left doing the job the header band did.
LEFT, RIGHT, DIVIDER = 0.035, 0.545, 0.495
ROWS_Y = (0.85, 0.62, 0.39, 0.16)


def panel(name: str, accent: str, rows, draw_left, height: float, caption: str):
    fig = plt.figure(figsize=(12.4, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.plot([DIVIDER, DIVIDER], [0.06, 0.94], lw=2.0, color=accent, alpha=0.5,
            solid_capstyle="round")
    draw_left(fig, ax)
    ax.text(LEFT, 0.045, caption, fontsize=9, color=MUTED)
    ax.set_clip_on(False)

    for (label, detail), y in zip(rows, ROWS_Y, strict=True):
        ax.text(RIGHT, y, label, fontsize=11, color=INK, va="center")
        ax.text(RIGHT + 0.165, y, detail, fontsize=10.4, color=MUTED, va="center")

    fig.savefig(out / f"{name}.png", dpi=a.dpi, facecolor="white",
                bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.png + .pdf")


def trace_axes(fig, rect, series, inj_x, colour=TRACE, lw=0.5, ylim=None):
    b = fig.add_axes(rect)
    b.plot(np.arange(len(series)), series, lw=lw, color=colour)
    b.axvline(inj_x, lw=1.1, color=INJ, alpha=0.9)
    if ylim:
        b.set_ylim(*ylim)
    b.margins(x=0.01)
    b.axis("off")
    return b


# ------------------------------------------------------------------ one recording
d = np.load(a.data)
y, fs, inj = d["y"], int(d["fs"]), int(d["inj"])
step = max(1, len(y) // 2000)
Y, INJ_X = y[::step], inj / step

TRACE_RECT = (LEFT, 0.30, DIVIDER - LEFT - 0.04, 0.52)


def _single(fig, ax):
    trace_axes(fig, TRACE_RECT, Y, INJ_X)
    x = TRACE_RECT[0] + TRACE_RECT[2] * INJ_X / len(Y)
    ax.text(x, TRACE_RECT[1] + TRACE_RECT[3] + 0.03, "injection", fontsize=9,
            color=INJ, ha="center")


panel("single-trace", ONE,
      [("transient times", "every event, with its amplitude"),
       ("rate ratio", "how much the drug changed the rate"),
       ("a credible interval", "for that recording's rate ratio"),
       ("the carrier", "absorption and elimination, fitted")],
      _single, height=3.4, caption="one trace, one injection")

# ---------------------------------------------------------------- a whole cohort
# Really simulated, not the single trace repainted: the three conditions have
# genuinely different carriers, and drawing them any other way would invent the
# structure the hierarchy exists to model.
cfg = yaml.safe_load(open(a.config))
cfg["cohort"] |= dict(n_animals=2)
spec = spec_from_config(cfg, 0.0)
coh = simulate_cohort(spec, a.cohort_seed)
YC, drug, animal, injc = coh["y"], coh["drug"], coh["animal"], coh["inj_t"]
sub = max(1, YC.shape[1] // 1600)
# One y-scale for all six, from a robust range: autoscaling each row independently
# would flatten vehicle and the opioids onto the same apparent amplitude, hiding the
# per-drug carrier that is the reason for colouring them at all.
LO, HI = (float(np.percentile(YC.numpy(), q)) for q in (0.3, 99.7))


def _cohort(fig, ax):
    top, row_h, gap, animal_gap = 0.88, 0.088, 0.016, 0.045
    w = DIVIDER - LEFT - 0.075
    for i in range(YC.shape[0]):
        yy = top - row_h - i * (row_h + gap) - animal[i] * animal_gap
        trace_axes(fig, (LEFT, yy, w, row_h), YC[i, ::sub].numpy(),
                   injc[i] / sub, colour=DRUG_COLOURS[drug[i]], lw=0.6,
                   ylim=(LO, HI))
        if drug[i] == 1:                       # middle row of the animal's block
            ax.text(LEFT + w + 0.012, yy + row_h / 2, f"animal {animal[i] + 1}",
                    fontsize=8.6, color=MUTED, va="center")
    for j, (col, lab) in enumerate(zip(DRUG_COLOURS, DRUG_LABELS, strict=True)):
        x = LEFT + 0.175 + j * 0.105
        ax.plot([x, x + 0.022], [0.045] * 2, lw=2.4, color=col, solid_capstyle="butt")
        ax.text(x + 0.029, 0.045, lab, fontsize=9, color=MUTED, va="center")


panel("cohort", MANY,
      [("everything above", "for every session in the cohort"),
       ("a drug contrast", "with between-animal variance in it"),
       ("variance components", "what a power calculation needs"),
       ("per-drug carriers", "pooled across animals, not refitted")],
      _cohort, height=4.2, caption="many traces, one model")
