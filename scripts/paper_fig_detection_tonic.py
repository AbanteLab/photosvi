"""Paper figure: the tonic side of the detection benchmark, under the Fig. 2 trace.

(b) One recording's tonic at each noise level: the simulated truth, PhotoSVI's fitted
    x_hat(t), and the Bonaventura lab's 0.1 Hz causal low-pass of the same trace, over the
    1 s means of the data.
(c) Error of three summaries scripts/detection_tonic.py takes from each curve -- the peak
    height above the pre-injection level and the time of that peak after the injection --
    against the SNR, PhotoSVI against the low-pass and the lab's ALS
    baseline (lambda 1e6 * fs), mean +- sd over seeds. ALS is drawn in (c) only, to keep (b)
    legible.
"""
import argparse
import glob
import json
import math
import pathlib
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_tonic import area_rel_err  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--config", default=str(
    pathlib.Path(__file__).resolve().parent.parent / "configs/final_detection_48.yaml"))
ap.add_argument("--seed", type=int, default=0, help="the representative recording of (b)")
ap.add_argument("--out", default="diagnostics/fig_detection_tonic.pdf")
a = ap.parse_args()

R = [json.load(open(f)) for f in sorted(glob.glob(f"{a.runs}/bm_*.json"))]
R = [r for r in R if "tonic" in r]
sig = sorted({r["sigma_y"] for r in R})
_gt = yaml.safe_load(open(a.config))["gt"]
# SNR as the paper defines it (Methods): the median mark over sigma_y, e^mu_a / sigma_y.
EA = math.exp(float(_gt["mu_a"]))
# Fig. 2's colours: PhotoSVI blue, the lab's method the olive of its z-peaks row; the truth
# black here, since its green would sit next to that olive.
C_PP, C_LP, C_TRUE, C_OBS = "#2a78d6", "#6a8d3a", "#222222", "#d4d6da"
C_ALS = "#8a5aa6"

fig = plt.figure(figsize=(12.5, 3.0))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, .08, 1.0], wspace=.32)
gs_c = gs[0, 4].subgridspec(3, 1, hspace=.14)

# --- (b) the representative recording at each level ------------------------------
axb = []
for j, s in enumerate(sig):
    r = next(r for r in R if r["sigma_y"] == s and r["seed"] == a.seed)
    tn = r["tonic"]
    cv = tn["curves"]
    t = (np.arange(len(cv["true"])) * tn["curve_step_s"] - tn["curve_inj_s"]) / 60
    ax = fig.add_subplot(gs[0, j], sharey=axb[0] if axb else None)
    axb.append(ax)
    ax.plot(t[:len(cv["data"])], cv["data"], lw=.5, color=C_OBS, zorder=1)
    ax.plot(t, cv["lowpass"], lw=1.0, color=C_LP, zorder=2, label="low-pass 0.1 Hz")
    ax.plot(t, cv["true"], lw=1.3, color=C_TRUE, zorder=3, label="true")
    ax.plot(t, cv["photosvi"], lw=1.4, color=C_PP, zorder=4, ls=(0, (4, 1.5)),
            label="PhotoSVI")
    ax.axvline(0, color="#999999", lw=.7, ls=":")
    ax.set_title(f"SNR {EA / s:.2f}", fontsize=9, color="#555555")
    ax.set_xlabel("time from injection (min)")
    if j == 0:
        ax.set_ylabel("dF/F (simulated units)")
        ax.text(-.22, 1.04, "b", transform=ax.transAxes, fontsize=12, fontweight="bold")
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
lo = np.percentile(np.concatenate([r["tonic"]["curves"]["data"] for r in R
                                   if r["seed"] == a.seed]), 1)
hi = max(max(max(r["tonic"]["curves"][k]) for k in ("true", "photosvi", "lowpass"))
         for r in R if r["seed"] == a.seed)
axb[0].set_ylim(lo, hi + .15 * (hi - lo))
axb[0].legend(fontsize=7.5, frameon=False, loc="upper right")

# --- (c) error against SNR ---------------------------------------------------------
snr = np.array([EA / s for s in sig])


def one_err(r, m, key):
    """Error of one recording: height and area relative (%), peak time in minutes."""
    sm = r["tonic"]["summary"]
    if key == "height":
        return 100 * (sm[m]["peak_height"] / sm["true"]["peak_height"] - 1)
    if key == "time":
        return (sm[m]["peak_time_s"] - sm["true"]["peak_time_s"]) / 60
    return area_rel_err(r["tonic"], m)


def err(m, key):
    e = [[one_err(r, m, key) for r in R if r["sigma_y"] == s] for s in sig]
    return np.array([np.mean(v) for v in e]), np.array([np.std(v, ddof=1) for v in e])


axc = [fig.add_subplot(gs_c[0])]
axc += [fig.add_subplot(gs_c[i], sharex=axc[0]) for i in (1, 2)]
for ax, key, lab in ((axc[0], "height", "peak height\nerror (%)"),
                     (axc[1], "time", "peak time\nerror (min)"),
                     (axc[2], "area", "area\nerror (%)")):
    for m, col, dx, name in (("photosvi", C_PP, -.018, "PhotoSVI"),
                             ("lowpass", C_LP, 0.0, "low-pass"),
                             ("als", C_ALS, .018, "ALS")):
        mu, sd = err(m, key)
        ax.errorbar(snr * (1 + dx), mu, yerr=sd, color=col, marker="o", ms=3.5, lw=1.1,
                    capsize=2, label=name)
    ax.axhline(0, color="#999999", lw=.7)
    ax.set_ylabel(lab, fontsize=8.5)
axc[0].text(-.30, 1.08, "c", transform=axc[0].transAxes, fontsize=12, fontweight="bold")
axc[0].legend(fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=3,
              handlelength=1.4, columnspacing=.9, handletextpad=.4)
for x in axc[:2]:
    plt.setp(x.get_xticklabels(), visible=False)
axc[2].set_xlabel("SNR")
axc[2].set_xticks(snr)
axc[2].set_xticklabels([f"{v:.2f}" for v in snr])

for x in fig.axes:
    x.grid(alpha=.15, lw=.7)
    x.set_axisbelow(True)
    for sp in x.spines.values():
        sp.set_color("#cccccc")
    x.tick_params(labelsize=8.5, colors="#555555")
_out = Path(a.out)
_out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(_out, transparent=True, bbox_inches="tight")  # PDF: no white patch over LaTeX labels
fig.savefig(_out.with_suffix(".png"), dpi=200, facecolor="white", bbox_inches="tight")
print(f"wrote {_out} and {_out.with_suffix('.png')}  ({len(R)} cells, seed {a.seed} in b)")
