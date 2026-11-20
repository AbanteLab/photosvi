"""Paper figure: detection accuracy against GuPPy.

Four panels: a representative trace with both detectors' calls, F1 and PR-AUC
against noise level, and the count of events each detector returns. The matched
filter is computed by the harness but deliberately not plotted -- the comparison
the paper makes is against the tool the field actually uses.
"""
import argparse
import glob
import json
import math
import os
import pathlib

import matplotlib
import numpy as np
import yaml

from photosvi.benchmark import match_events, nms, peaks_from_score
from photosvi.benchmark.harness import guppy_detect, zpeaks_detect

matplotlib.use("Agg")
# TrueType, not Type 3, so the PDF's text stays selectable and editable in a vector
# editor -- and so journals that reject Type 3 fonts accept it.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
from pathlib import Path

import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
# runs_benchmark_48 is the 48 Hz pass: the recordings' own sampling rate, the kernel and
# the autoregressive emission fitted to them, and non-maximum suppression applied to every
# detector. Earlier directories ran at 25 Hz with none of those and their numbers are not
# comparable -- see the config header and
ap.add_argument("--runs", default="runs_benchmark_48")
# The representative trace, kept with the runs rather than in a scratch directory: the
# figure is committed to the paper repo and has to be reproducible after the session that
# made it is gone.
ap.add_argument("--trace",
                default=os.path.join(os.environ.get("PHOTOSVI_RUNS", "runs"), "traces/fd48_3.00.json"))
ap.add_argument("--config", default=str(
    pathlib.Path(__file__).resolve().parent.parent / "configs/benchmark_guppy_48.yaml"),
    help="where the mark distribution comes from, for the transient-to-noise axis")
ap.add_argument("--out", default="diagnostics/fig_detection.pdf",
                help="vector output; a .png of the same stem is written alongside")
ap.add_argument("--t0", type=float, default=980.0)
ap.add_argument("--dur", type=float, default=70.0)
a = ap.parse_args()

R = [json.load(open(f)) for f in glob.glob(f"{a.runs}/bm_*.json")]
D = json.load(open(a.trace))
sig = sorted({r["sigma_y"] for r in R})
C_GUP, C_PP, C_TRUE, C_OBS = "#eb6834", "#2a78d6", "#1baf7a", "#b9bcc2"
# E[a], the typical transient height, from the config the benchmark ran rather than inlined:
# it sets the transient-to-noise axis of two panels and a caption, and a reparameterisation
# that changed the marks would otherwise relabel the figure wrongly and silently.
_cfg = yaml.safe_load(open(a.config))
_gt = _cfg["gt"]
# Both in BINS, converted from the config's seconds at the config's rate. Inlining either
# is how a detector ends up with a 0.32 s refractory at one sampling rate and 0.17 s at
# another while the code looks unchanged.
REFRACTORY = int(_cfg["detection"]["refractory"])
TOL = int(round(float(_cfg["detection"]["tol_s"]) * float(_cfg["data"]["fs"])))
EA = math.exp(float(_gt["mu_a"]) + 0.5 * float(_gt["sigma_a"]) ** 2)

def agg(det, key):
    m = [np.mean([r["detectors"][det][key] for r in R if r["sigma_y"] == s]) for s in sig]
    e = [np.std([r["detectors"][det][key] for r in R if r["sigma_y"] == s], ddof=1) for s in sig]
    return np.array(m), np.array(e)

fig = plt.figure(figsize=(12.5, 3.4))
gs = fig.add_gridspec(1, 1)
fig.patch.set_facecolor("white")

# --- A: a representative trace, spanning both panels of the top row ----------
ax = fig.add_subplot(gs[0, 0])
dt, INJ = D["dt"], D["inj"]
i0, i1 = int(a.t0 / dt), int((a.t0 + a.dur) / dt)
t = np.arange(i0, i1) * dt
y = np.array(D["y"])[i0:i1]
ax.plot(t, y, lw=.7, color=C_OBS, zorder=1)
ax.plot(t, np.array(D["x_fit"])[i0:i1], lw=1.6, color="#5a5a5a", zorder=2)
hi, lo = y.max(), y.min()
pad = .10 * (hi - lo)
# `calls` from the harness are *candidate* peaks, thousands of them, before any
# threshold, so each method has to be shown at an operating point. GuPPy's rule already
# includes its own thresholds and is drawn as it comes.
#
# For photosvi the operating point is the one panel B reports -- the threshold that
# maximises F1 -- so that the trace shows what the accuracy panel measures. A fixed
# cut at q > 0.5 is what this used to draw and it is the wrong question to ask of a
# calibrated posterior over a sparse process: with 466 onsets in 115200 bins the base
# rate is 0.004, the posterior peaks at 0.53 over the whole recording, exactly one bin
# clears 0.5, and panel A came out with an empty photosvi row while panels B to D showed
# it winning at every noise level.
q_arr = np.array(D["q"])
_pk, _ps = nms(*peaks_from_score(q_arr), refractory=REFRACTORY)


def best_f1_calls(name, pk, ps):
    """The candidates above the threshold that maximizes this method's F1 on this trace."""
    lag = int(D["metrics"][name].get("lag", 0))
    order = np.argsort(-np.asarray(ps))
    tp = match_events(np.asarray(pk), np.asarray(ps), np.asarray(D["onsets"]),
                      TOL, lag)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    prec, rec = ctp / (ctp + cfp), ctp / max(len(D["onsets"]), 1)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    cut = float(np.asarray(ps)[order][int(np.argmax(f1))])
    calls = [int(v) for v, sc in zip(pk, ps) if sc >= cut]
    print(f"  {name} at its best-F1 threshold {cut:.4f}: {len(calls)} of {len(pk)} "
          f"candidates, F1 {f1.max():.3f} (stored {D['metrics'][name]['best_f1']:.3f})")
    return calls


pp_calls = best_f1_calls("photosvi", _pk, _ps)
# GuPPy and z-peaks at their own best-F1 thresholds too, as in Table 1: the
# trace stores only their default-rule calls, so their scored candidates are recomputed here
# with the benchmark's own detectors and suppression, which are deterministic.
_y_full = np.asarray(D["y"], dtype=float)
gp_calls = best_f1_calls("guppy", *nms(*guppy_detect(_y_full, fs=D["fs"]), refractory=REFRACTORY))
zp_calls = best_f1_calls("z-peaks", *nms(*zpeaks_detect(_y_full, D["inj"], fs=D["fs"]),
                                         refractory=REFRACTORY))

# All three rows stacked above the trace, GuPPy immediately under photosvi. Putting one
# method above the trace and the other below it meant comparing them across the recording,
# which is the comparison the panel exists to make easy.
rows = [("true onsets", D["onsets"], C_TRUE, hi + pad * 4.5, hi + pad * 5.3),
        ("PhotoSVI", pp_calls, C_PP, hi + pad * 3.2, hi + pad * 4.0),
        ("GuPPy", gp_calls, C_GUP, hi + pad * 1.9, hi + pad * 2.7),
        ("z-peaks", zp_calls, "#6a8d3a", hi + pad * 0.6, hi + pad * 1.4)]
for lab, ev, col, y0, y1 in rows:
    for e in [v * dt for v in ev if i0 <= v < i1]:
        ax.plot([e, e], [y0, y1], color=col, lw=1.3)
    ax.text(t[0] - .012 * a.dur, (y0 + y1) / 2, lab, ha="right", va="center",
            fontsize=9, color=col)
ax.set_xlim(t[0] - .13 * a.dur, t[-1])
ax.set_ylim(lo - pad * 0.6, hi + pad * 5.9)
ax.set_ylabel("dF/F (simulated units)"); ax.set_xlabel("time (s)")
# No title: the caption says what the panel shows.
# The accuracy panels are gone; the summary line stays, since it is what the caller
# reads to check the figure matches the table it now sits beside.
snr = [EA / s for s in sig]

# Panels B, C and D were removed on F1, PR-AUC and recall at 90%
# precision against the noise level are exactly the three columns of Table 1, and a
# figure that redraws a table earns nothing. The trace is what the table cannot show --
# what the two methods do to the same recording, event by event.
for x in fig.axes:
    x.grid(alpha=.15, lw=.7); x.set_axisbelow(True)
    for sp in x.spines.values(): sp.set_color("#cccccc")
    x.tick_params(labelsize=9, colors="#555555")
# PDF is the deliverable -- editable vector art, and LaTeX prefers it. The PNG is
# written alongside for quick viewing.
_out = Path(a.out)
fig.savefig(_out, transparent=True, bbox_inches="tight")  # PDF: no white patch over LaTeX labels
fig.savefig(_out.with_suffix(".png"), dpi=200, facecolor="white", bbox_inches="tight")
print(f"wrote {_out} and {_out.with_suffix('.png')}")
print(a.out)
for s in sig:
    g = np.mean([r["detectors"]["guppy"]["best_f1"] for r in R if r["sigma_y"] == s])
    p = np.mean([r["detectors"]["photosvi"]["best_f1"] for r in R if r["sigma_y"] == s])
    print(f"  SNR {EA/s:5.1f}:  GuPPy {g:.3f}   photosvi {p:.3f}")
