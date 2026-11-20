#!/usr/bin/env python3
"""Macros for the methods paragraph on how the simulated recordings were calibrated.

    python scripts/paper_calibration_numbers.py --out ../../papers/photosvi-paper/generated/calibration_numbers.tex

Every number the paragraph states is recomputed here from the recordings and the simulator
config, so the paragraph cannot drift from what the simulator actually does. Three of them
need care:

* The transient-to-noise ratio is measured with the SAME detector on the recordings and on
  a generated trace. The tempting shortcut -- comparing the recordings' measured ratio
  against exp(mu_a)/sigma_y from the config -- compares two different quantities: the
  config's is the median mark over every event, while a detector run at a prominence
  threshold of three local sigma only ever reports peaks above three, so its median is
  bounded below by the threshold. Measured the same way on both, the numbers are
  comparable; measured the two different ways they differ by more than twofold for reasons
  that have nothing to do with calibration.

* The autoregressive coefficients are read FROM the config rather than refitted. They are
  what the simulator used, which is what the paragraph is describing; refitting here would
  silently report one thing while the benchmark ran another. Their per-session spread is
  computed from the recordings, since that is the paragraph's separate point.

* The rate ceiling is measured, not asserted: the true rate is swept and the detected rate
  recorded, and the ceiling is where the detected rate stops following.
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import numpy as np
import yaml
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, filtfilt, find_peaks

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from photosvi.data.bonaventura3 import FS, PRE_S, load_sessions  # noqa: E402
from photosvi.experiments.sweep import generate  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--config", default=str(REPO / "configs/benchmark_guppy_48.yaml"),
                help="the simulator config the benchmark ran")
ap.add_argument("--seeds", type=int, default=3, help="generated traces per measurement")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
BP = butter(3, [0.1 / (FS / 2), 10.0 / (FS / 2)], "band")
PRE_BINS = int(PRE_S * FS)


def detect(y):
    """Band-pass, then the prominence detector, returning peaks and the local noise scale.

    Width is given in seconds converted to bins rather than in bins directly: the same
    literal at two sampling rates is two different durations, which is how five
    rate-dependent comparisons went wrong earlier in this work.
    """
    z = filtfilt(*BP, np.asarray(y, float))
    sd = float(np.median(np.abs(z)) / 0.6745)
    pk, pr = find_peaks(z, prominence=3 * sd, width=max(1, int(0.04 * FS)))
    return z, sd, pk, pr


def ampl_over_noise(y):
    """Median detected prominence in units of the local noise scale."""
    _, sd, pk, pr = detect(y)
    return float(np.median(pr["prominences"]) / sd) if len(pk) else float("nan")


def peaks_per_min(y):
    _, _, pk, _ = detect(y)
    return 60.0 * len(pk) / (len(y) / FS)


def yule_walker(y, p=2):
    y = np.asarray(y, float) - np.mean(y)
    r = np.array([np.dot(y[:len(y) - k], y[k:]) / len(y) for k in range(p + 1)])
    return solve_toeplitz((r[:p], r[:p]), r[1:p + 1])


sessions = load_sessions(verbose=False, als_lambda_per_fs=1e4)
animals = sorted({s.animal for s in sessions})
baselines = [s.y[:PRE_BINS] for s in sessions]
# The simulator was calibrated against ONE session per animal -- the nociceptin arm, which
# was the whole of the dataset at the time, before the saline recordings turned out to be
# the same five animals. So the kernel's transient count and the reference coefficients
# describe those five sessions, and reporting the ten-session figures here would state a
# number the kernel was never fitted to. The per-session SPREAD below does use all ten,
# because its point is how much the animals differ, and that is a property of the cohort
# the paper analyses rather than of the calibration.
fit_baselines = [s.y[:PRE_BINS] for s in sessions if s.cond == "noci"]

# --- the recordings -------------------------------------------------------------------
ampl = np.array([ampl_over_noise(b) for b in baselines])
rate = np.array([peaks_per_min(b) for b in baselines])
phi1 = np.array([yule_walker(b)[0] for b in baselines])

# Isolated transients behind the kernel fit: a stricter threshold and a four-second
# exclusion, so the peak-triggered average is not a sum of overlapping transients.
#
# This selection finds a few more than the one the shipped kernel was fitted to, which
# detrended the ten-minute slice itself rather than the whole analysis window. Refitting on
# this selection was checked rather than assumed: it lowers the residual by 4% and moves the
# kernel's half-width from 0.500 s to 0.479 s, which is not worth a new kernel and a new
# benchmark, so the shipped parameters stand and this is the count of what the current
# pipeline isolates.
n_transients = 0
for b in fit_baselines:
    z, sd, _, _ = detect(b)
    pk, _ = find_peaks(z, prominence=4 * sd, distance=int(4 * FS))
    pre_w, post_w = int(1.0 * FS), int(4.0 * FS)
    n_transients += sum(1 for p in pk if pre_w <= p < len(z) - post_w)

# --- the simulator -------------------------------------------------------------------
# Measured on generated traces with the detector above, at the middle of the grid the
# benchmark swept, and averaged over seeds so one draw's peaks do not set the number.
grid = sorted(float(s) for s in cfg["grid"]["sigma_y"]) if "grid" in cfg else None
sigma_mid = grid[len(grid) // 2] if grid else float(cfg["gt"]["sigma_y"])
sim_cfg = copy.deepcopy(cfg)
sim_cfg["gt"] = dict(cfg["gt"], sigma_y=sigma_mid)
sim_ampl, sim_rate = [], []
for seed in range(a.seeds):
    y, _ = generate(sim_cfg, seed)
    trace = y[0].numpy()[:PRE_BINS]
    sim_ampl.append(ampl_over_noise(trace))
    sim_rate.append(peaks_per_min(trace))

# --- what a peak detector reports, and what it is reporting ---------------------------
# The sweep runs the true rate from nothing to many times the recordings' level and
# measures the detected rate the same way as on the recordings. The point of the extra
# point at the bottom is the one that matters: a trace generated with NO events at all
# still returns peaks, because a prominence detector run on correlated noise finds
# prominent excursions in the noise. That floor is what the detected rate mostly consists
# of here, and it is why the detected rate cannot calibrate beta0 in either direction.
sweep = []
for b0 in (-12.0, -8.0, -5.0, float(cfg["gt"]["beta0"]), -1.0, 0.0, 1.0):
    c = copy.deepcopy(sim_cfg)
    # beta_inj is zeroed so the whole trace sits at one rate and the window carries no
    # step: the sweep is about the rate level, not about the injection.
    c["gt"] = dict(sim_cfg["gt"], beta0=float(b0), beta_inj=0.0)
    det, true = [], []
    for seed in range(a.seeds):
        y, onsets = generate(c, seed)
        det.append(peaks_per_min(y[0].numpy()[:PRE_BINS]))
        true.append(60.0 * int((onsets < PRE_BINS).sum()) / (PRE_BINS / FS))
    sweep.append((float(b0), float(np.mean(true)), float(np.mean(det))))
fp_floor = sweep[0][2]                      # detected rate with no events at all
assert sweep[0][1] < 0.1, f"the floor point is not event-free: {sweep[0][1]:.2f}/min"
at_shipped = next(r for r in sweep if r[0] == float(cfg["gt"]["beta0"]))
# How much of the detected rate at the shipped beta0 is the floor rather than the events.
fp_share = 100.0 * fp_floor / at_shipped[2]

L = [
    "% Generated by scripts/paper_calibration_numbers.py -- do not edit by hand.",
    "%",
    "% The transient-to-noise ratio is measured with the same detector on the recordings",
    "% and on generated traces. Comparing the measured ratio against exp(mu_a)/sigma_y",
    "% instead would compare the median DETECTED prominence, bounded below by the",
    "% detector's threshold, against the median mark over ALL events.",
    "",
    f"\\newcommand{{\\ppCalNAnimals}}{{{len(animals)}}}",
    f"\\newcommand{{\\ppCalNSessions}}{{{len(sessions)}}}",
    f"\\newcommand{{\\ppCalNFitSessions}}{{{len(fit_baselines)}}}",
    f"\\newcommand{{\\ppCalNTransients}}{{{n_transients}}}",
    f"\\newcommand{{\\ppCalAmplSdReal}}{{{ampl.mean():.1f}}}",
    f"\\newcommand{{\\ppCalAmplSdRealSd}}{{{ampl.std(ddof=1):.1f}}}",
    f"\\newcommand{{\\ppCalAmplSdSim}}{{{np.mean(sim_ampl):.1f}}}",
    f"\\newcommand{{\\ppCalSigmaMid}}{{{sigma_mid:g}}}",
    f"\\newcommand{{\\ppCalRateReal}}{{{rate.mean():.0f}}}",
    f"\\newcommand{{\\ppCalRateRealSd}}{{{rate.std(ddof=1):.0f}}}",
    f"\\newcommand{{\\ppCalRateSim}}{{{at_shipped[2]:.0f}}}",
    f"\\newcommand{{\\ppCalTrueRateSim}}{{{at_shipped[1]:.1f}}}",
    f"\\newcommand{{\\ppCalFpFloor}}{{{fp_floor:.0f}}}",
    f"\\newcommand{{\\ppCalFpShare}}{{{fp_share:.0f}}}",
    f"\\newcommand{{\\ppCalSweepHi}}{{{sweep[-1][1]:.0f}}}",
    f"\\newcommand{{\\ppCalSweepHiDet}}{{{sweep[-1][2]:.0f}}}",
    f"\\newcommand{{\\ppCalBetaZero}}{{{cfg['gt']['beta0']:g}}}",
    f"\\newcommand{{\\ppCalPhiLo}}{{{phi1.min():.2f}}}",
    f"\\newcommand{{\\ppCalPhiHi}}{{{phi1.max():.2f}}}",
    f"\\newcommand{{\\ppCalPhi}}{{[{', '.join(f'{v:g}' for v in cfg['emission']['ar_coeffs'])}]}}",
]
pathlib.Path(a.out).write_text("\n".join(L) + "\n")
print(f"  {a.out}")
print(f"  ampl/sd real {ampl.mean():.2f} +- {ampl.std(ddof=1):.2f} | "
      f"sim {np.mean(sim_ampl):.2f} at sigma_y {sigma_mid:g}")
print(f"  peaks/min real {rate.mean():.1f} +- {rate.std(ddof=1):.1f} | "
      f"sim {at_shipped[2]:.1f} at a TRUE rate of {at_shipped[1]:.1f} | "
      f"floor with no events at all {fp_floor:.1f} ({fp_share:.0f}% of the detected rate)")
print(f"  phi_1 over {len(sessions)} sessions: {phi1.min():.2f} to {phi1.max():.2f} | "
      f"{n_transients} isolated transients over the {len(fit_baselines)} the kernel "
      f"was fitted to")
print("  sweep, true -> detected peaks/min: "
      + ", ".join(f"{t:.0f}->{d:.0f}" for _, t, d in sweep))
