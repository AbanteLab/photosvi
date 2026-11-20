"""Paper: the tonic contrast of the cohort benchmark, PhotoSVI against the low-pass and ALS.

In the cohort benchmark the drug changes the rate of transients (by Delta on the log scale)
and not the tonic response, which is the same in both conditions, so every estimate of the
tonic contrast between conditions should be zero. Per cohort:

  * hierarchical -- the posterior means of the two conditions' tonic peak (hier_*.json);
  * two-stage    -- the single-trace tonic fits (ts_*.json), paired within animal;
  * low-pass, ALS -- the slow components of the same recordings (tb_*.json, from
    scripts/cohort_tonic_baselines.py), paired within animal.

Two contrasts, treated minus vehicle: the log ratio of peak heights and the difference of
peak times (min). Writes a two-row figure in the layout of Fig. 2c (mean +- s.d. over
cohorts against Delta) and the macros \\ppTonicCoh<Arm><Height|Time><Bias|Sd><Delta>.
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True)
ap.add_argument("--out", required=True, help="figure (.pdf; a .png beside it)")
ap.add_argument("--macros", required=True)
a = ap.parse_args()

VEH, TRT = 0, 1
ARMS = (("hier", "hierarchical", "#2a78d6"), ("twostage", "two-stage", "#7fb2ea"),
        ("lowpass", "low-pass", "#6a8d3a"), ("als", "ALS", "#8a5aa6"))


def bateman_peak_s(kr, kd):
    return math.log(kr / kd) / (kr - kd)


def paired(rows):
    """Mean over animals of treated minus vehicle, for (animal, drug, height, time) rows."""
    by = defaultdict(dict)
    for an, dr, h, t in rows:
        by[an][dr] = (h, t)
    dh = [math.log(v[TRT][0] / v[VEH][0]) for v in by.values() if VEH in v and TRT in v]
    dt = [(v[TRT][1] - v[VEH][1]) / 60 for v in by.values() if VEH in v and TRT in v]
    return float(np.mean(dh)), float(np.mean(dt))


est = defaultdict(list)          # (arm, delta) -> [(height contrast, time contrast)]
for f in sorted(glob.glob(f"{a.cell}/hier_*.json")):
    d, s = re.match(r"hier_([\d.]+)_(\d+)\.json", os.path.basename(f)).groups()
    d, s = float(d), int(s)
    tb = f"{a.cell}/tb_{d:.2f}_{s:04d}.json"
    ts = sorted(glob.glob(f"{a.cell}/ts_{d:.2f}_{s:04d}_*.json"))
    if not os.path.exists(tb) or len(ts) < 2:
        continue
    p = json.load(open(f))["tonic_posterior"]
    est["hier", d].append((math.log(p["treated"]["peak"] / p["vehicle"]["peak"]),
                           (p["treated"]["peak_s"] - p["vehicle"]["peak_s"]) / 60))
    T = [json.load(open(x)) for x in ts]
    est["twostage", d].append(paired([(r["animal"], r["drug"], r["tonic_peak"],
                                       bateman_peak_s(r["tonic_k_rise"], r["tonic_k_decay"]))
                                      for r in T]))
    B = json.load(open(tb))["sessions"]
    for arm in ("lowpass", "als"):
        est[arm, d].append(paired([(b["animal"], b["drug"], b[arm]["peak_height"],
                                    b[arm]["peak_time_s"]) for b in B]))

deltas = sorted({d for _, d in est})
n = {d: len(est["hier", d]) for d in deltas}
stat = {(arm, d, j): (float(np.mean([e[j] for e in est[arm, d]])),
                      float(np.std([e[j] for e in est[arm, d]], ddof=1)))
        for arm, *_ in ARMS for d in deltas for j in (0, 1)}

# ------------------------------------------------------------------ figure (Fig. 2c layout)
fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True,
                        gridspec_kw=dict(hspace=.12))
step = (deltas[-1] - deltas[0]) / max(len(deltas) - 1, 1)
for ax, j, lab in ((axs[0], 0, "height contrast\n(log ratio)"), (axs[1], 1, "time contrast\n(min)")):
    for i, (arm, name, col) in enumerate(ARMS):
        dx = (i - 1.5) * .06 * step
        mu = [stat[arm, d, j][0] for d in deltas]
        sd = [stat[arm, d, j][1] for d in deltas]
        ax.errorbar(np.array(deltas) + dx, mu, yerr=sd, color=col, marker="o", ms=3.5, lw=1.1,
                    capsize=2, label=name)
    ax.axhline(0, color="#999999", lw=.7)
    ax.set_ylabel(lab, fontsize=8.5)
for ax, letter in zip(axs, "ab"):
    ax.text(-.13, 1.02, letter, transform=ax.transAxes, fontsize=12, fontweight="bold")
axs[0].legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=4,
              handlelength=1.4, columnspacing=.9, handletextpad=.4)
axs[1].set_xlabel(r"effect on the log transient rate, $\Delta$")
axs[1].set_xticks(deltas)
for x in axs:
    x.grid(alpha=.15, lw=.7)
    x.set_axisbelow(True)
    for sp in x.spines.values():
        sp.set_color("#cccccc")
    x.tick_params(labelsize=8.5, colors="#555555")
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, transparent=True, bbox_inches="tight")  # PDF: no white patch over LaTeX labels
fig.savefig(out.with_suffix(".png"), dpi=200, facecolor="white", bbox_inches="tight")

# ------------------------------------------------------------------ macros
Arm = {"hier": "Hier", "twostage": "Twostage", "lowpass": "Lowpass", "als": "Als"}
# LaTeX macro names cannot hold digits: the effect sizes are named in order (One = the
# smallest), and the mapping is written into the file's header.
_ORD = ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten")
tag = lambda d: "Delta" + _ORD[deltas.index(d)]  # noqa: E731
lines = ["% Generated by scripts/paper_cohort_tonic_compare.py -- do not edit by hand.",
         f"% {a.cell}: cohorts per delta {n}; true tonic contrast 0 in every cohort.",
         "% Delta" + ", Delta".join(f"{_o}={d:g}" for _o, d in zip(("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"), deltas))]
for arm, *_ in ARMS:
    for d in deltas:
        for j, q in ((0, "Height"), (1, "Time")):
            m, s = stat[arm, d, j]
            fmt = "{:+.3f}" if j == 0 else "{:+.1f}"
            lines.append(f"\\newcommand{{\\ppTonicCoh{Arm[arm]}{q}Bias{tag(d)}}}{{{fmt.format(m)}}}")
            lines.append(f"\\newcommand{{\\ppTonicCoh{Arm[arm]}{q}Sd{tag(d)}}}{{{abs(s):.3f}}}"
                         if j == 0 else
                         f"\\newcommand{{\\ppTonicCoh{Arm[arm]}{q}Sd{tag(d)}}}{{{abs(s):.1f}}}")
        # height bias as a percentage, at the largest effect, for the prose
    lines.append(f"\\newcommand{{\\ppTonicCoh{Arm[arm]}HeightPctMax}}"
                 f"{{{100 * (math.exp(stat[arm, deltas[-1], 0][0]) - 1):+.0f}}}")
# How many times more precise PhotoSVI's peak-time contrast is than the better slow-component
# method's, at the main effect size (the worse PhotoSVI arm against the better baseline).
_dm = min(deltas, key=lambda d: abs(d - 0.5))
_ratio = (min(stat["lowpass", _dm, 1][1], stat["als", _dm, 1][1])
          / max(stat["hier", _dm, 1][1], stat["twostage", _dm, 1][1]))
lines.append(f"\\newcommand{{\\ppTonicCohTimePrecRatio}}{{{_ratio:.0f}}}")
lines.append(f"\\newcommand{{\\ppTonicCohDeltaMax}}{{{deltas[-1]:g}}}")
lines.append(f"\\newcommand{{\\ppTonicCohN}}{{{n[deltas[0]]}}}")
Path(a.macros).write_text("\n".join(lines) + "\n")
print(f"wrote {out} and {a.macros}")
for arm, *_ in ARMS:
    print(f"  {arm:9s} " + "  ".join(
        f"D={d:g}: h {stat[arm, d, 0][0]:+.3f}+-{stat[arm, d, 0][1]:.3f} t {stat[arm, d, 1][0]:+.1f}+-{stat[arm, d, 1][1]:.1f}"
        for d in deltas))
