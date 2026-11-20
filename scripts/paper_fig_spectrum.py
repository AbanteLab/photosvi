#!/usr/bin/env python3
"""Supplementary figure: the simulator's spectrum against the recordings'.

    python scripts/paper_fig_spectrum.py --out .../fig_spectrum.pdf

The claim the simulated benchmark rests on is that its recordings sit at the difficulty the
method will actually meet. The calibration behind that is six summary statistics
(\\nameref{sec:preprocessing}); this is the check those statistics cannot make, because a
spectrum is the whole second-order structure and they are four numbers from it.

**Both sides are divided by their own robust scale before transforming.** A dF/F recording
and a simulated trace are in different units -- the recordings' residual scale is 0.013 and
the simulator's is 3.0 -- so an unnormalised comparison would show that difference and
nothing else. What is being compared is the SHAPE: where the power sits, not how much of it
there is.

The real traces are the ten pre-injection windows, which is where the noise is measured and
the only stretch free of the drug.
"""
from __future__ import annotations

import argparse, copy, pathlib, sys

import matplotlib
import numpy as np
import yaml
from scipy.signal import detrend, welch

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from photosvi.data.bonaventura3 import FS, PRE_S, load_sessions  # noqa: E402
from photosvi.experiments.sweep import generate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(REPO / "configs/benchmark_guppy_48.yaml"))
ap.add_argument("--out", default="diagnostics/fig_spectrum.pdf")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--macros", default=None,
                help="write the band ratios the caption states to this .tex")
ap.add_argument("--nperseg-s", type=float, default=160.0,
                help="Welch segment in SECONDS, not samples: the same literal at two "
                     "sampling rates is two different resolutions")
a = ap.parse_args()

NPERSEG = int(a.nperseg_s * FS)


def psd(x):
    """Welch PSD of a trace standardised by its own robust scale, detrended first."""
    x = detrend(np.asarray(x, float), type="linear")
    s = np.median(np.abs(x - np.median(x))) / 0.6745
    f, p = welch(x / max(s, 1e-12), fs=FS, nperseg=NPERSEG)
    k = f > 0
    return f[k], p[k]


real = []
for sess in load_sessions(verbose=False, als_lambda_per_fs=1e4):
    f, p = psd(sess.y[:int(PRE_S * FS)])
    real.append(p)
real = np.array(real)
print(f"  {len(real)} recordings")

cfg = yaml.safe_load(open(a.config))
mid = sorted(float(v) for v in cfg["grid"]["sigma_y"])[len(cfg["grid"]["sigma_y"]) // 2]
sim = []
for seed in range(a.seeds):
    c = copy.deepcopy(cfg)
    c["gt"] = dict(cfg["gt"], sigma_y=mid)
    y, _ = generate(c, seed)
    _, p = psd(y[0].numpy()[:int(PRE_S * FS)])
    sim.append(p)
sim = np.array(sim)
print(f"  {len(sim)} simulated recordings at sigma_y = {mid:g}")

# Lower-order noise for comparison: white (p = 0) and AR(1), its
# coefficient the mean of per-session Yule-Walker fits as for the AR(2), so the figure shows
# how the simulated spectrum approaches the recordings' as the order grows.
from photosvi.models.points import estimate_ar_coeffs  # noqa: E402
phi1 = float(np.mean([estimate_ar_coeffs(s.y[:int(PRE_S * FS)], None, p=1)[0]
                      for s in load_sessions(verbose=False, als_lambda_per_fs=1e4)]))
lower = {}
for order, phi in ((0, []), (1, [phi1])):
    ps = []
    for seed in range(a.seeds):
        c = copy.deepcopy(cfg)
        c["gt"] = dict(cfg["gt"], sigma_y=mid)
        c["emission"] = dict(cfg.get("emission", {}), ar_coeffs=phi)
        y, _ = generate(c, seed)
        ps.append(psd(y[0].numpy()[:int(PRE_S * FS)])[1])
    lower[order] = np.array(ps)
print(f"  lower orders: white and AR(1) with phi = {phi1:.3f}")

# One panel per noise order against the recordings, each with its distance
# from them: the root-mean-square of log10(simulated / recorded) median power over 40
# log-spaced frequency bins from 0.01 Hz to Nyquist, so every decade weighs the same rather
# than the high frequencies, which hold most of the Welch bins.
edges = np.logspace(np.log10(0.01), np.log10(FS / 2), 41)


def log_dist(sim_p, real_p):
    r = np.log10(np.median(sim_p, 0)) - np.log10(np.median(real_p, 0))
    vals = [r[(f >= lo) & (f < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])
            if ((f >= lo) & (f < hi)).any()]
    return float(np.sqrt(np.mean(np.square(vals))))


orders = ((0, "white noise ($p$ = 0)", lower[0]),
          (1, f"AR(1) ($p$ = 1, $\\phi$ = {phi1:.3f})", lower[1]),
          (2, "AR(2) ($p$ = 2), as in the benchmark", sim))
dist = {o: log_dist(p_, real) for o, _, p_ in orders}
fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.6), sharex=True, sharey=True)
fig.patch.set_facecolor("white")
C_REAL, C_SIM = "#4a4a4a", "#eb6834"
for ax, (o, name, p_), letter in zip(axes, orders, "abc"):
    ax.fill_between(f, real.min(0), real.max(0), color=C_REAL, alpha=.18, lw=0)
    ax.plot(f, np.median(real, 0), color=C_REAL, lw=1.6, label=f"recordings (n={len(real)})")
    ax.fill_between(f, p_.min(0), p_.max(0), color=C_SIM, alpha=.18, lw=0)
    ax.plot(f, np.median(p_, 0), color=C_SIM, lw=1.6, label=f"simulated, {name}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.03, 0.04, f"distance {dist[o]:.2f} (log$_{{10}}$ units)", transform=ax.transAxes,
            fontsize=9, color=INK if False else "#2b2b2b")
    ax.legend(fontsize=8, framealpha=.95, loc="upper right")
    ax.grid(alpha=.15, lw=.7, which="both")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.tick_params(labelsize=9, colors="#555555")
    ax.text(-0.11, 1.04, letter, transform=ax.transAxes, fontsize=14, weight="bold",
            color="#2b2b2b", va="top")
axes[-1].set_xlabel("frequency (Hz)")
for ax in axes:
    ax.set_ylabel("standardized power")
print("  distance to the recordings: " + ", ".join(f"p={o} {d:.3f}" for o, d in dist.items()))

# What the text claims, measured here rather than asserted: the ratio of the two medians
# over the band the transients occupy.
# The caption states these four ratios; they are written out so it cannot drift from them.
BANDS = ((0.01, 0.03, "Low", "below the tonic's own band"),
         (0.03, 0.3, "Mid", "the band the tonic occupies"),
         (0.3, 2.0, "Trans", "the band the transients occupy"),
         (2.0, 20.0, "High", "above them"))
macros = []
for lo, hi, name, what in BANDS:
    b = (f >= lo) & (f < hi)
    r = float(np.median(np.median(sim, 0)[b] / np.median(real, 0)[b]))
    print(f"  {lo:>5}-{hi:<5} Hz  sim/real {r:.2f}   {what}")
    macros.append(f"\\newcommand{{\\ppSpec{name}}}{{{r:.0%}}}".replace("%", r"\%"))
if a.macros:
    pathlib.Path(a.macros).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.macros).write_text(
        "% Generated by scripts/paper_fig_spectrum.py -- do not edit by hand.\n"
        "% Ratio of the simulator's median power to the recordings', by band.\n"
        + "\n".join(macros) + "\n"
        + f"\\newcommand{{\\ppSpecSeeds}}{{{a.seeds}}}\n"
        + f"\\newcommand{{\\ppSpecPhiOne}}{{{phi1:.3f}}}\n"
        + "".join(f"\\newcommand{{\\ppSpecDist{n}}}{{{dist[o]:.2f}}}\n"
                  for o, n in ((0, "Zero"), (1, "One"), (2, "Two"))))
    print(f"  wrote {a.macros}")

out = pathlib.Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out, transparent=True, bbox_inches="tight")  # PDF: no white patch over LaTeX labels
fig.savefig(out.with_suffix(".png"), dpi=150, facecolor="white", bbox_inches="tight")
print(f"  wrote {out}")
