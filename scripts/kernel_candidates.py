"""Candidate biphasic transient kernels -- impulse responses for shortlisting.

Every candidate has sum(h) = 0 (the mean-free basis: E[c_t] = 0 for any intensity,
so the event rate cannot compete with the tonic for the signal mean) and peak 1
(so a mark still reads as the transient's height). What differs is HOW the zero
area is bought:

  DC-centred      h - mean(h). Cheap, but the negative part is a flat shelf that
                  runs to the end of the support -- the shape we are moving away from.
  biphasic        a fast positive lobe minus a scaled slower lobe, the scale set so
                  the areas cancel. Both lobes decay, so h returns to zero and an
                  event's influence ends.

Writes <out>.pdf, .png and .json (the params, for the configs); --out defaults
into the repo, but the paper build points it at the run directory like the others.
"""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
# TrueType, not Type 3, so the PDF stays editable in a vector editor -- and so
# journals that reject Type 3 fonts accept it.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

import argparse  # noqa: E402
import pathlib  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--out", default="diagnostics/kernel_candidates.pdf")
OUT = pathlib.Path(_ap.parse_args().out)
OUT.parent.mkdir(parents=True, exist_ok=True)

DT, SUP = 1/25, 5.0
# The axis runs longer than SUP so K7, four times slower than K5, is not truncated.
# SUP stays 5.0 and K1 is pinned to it explicitly, so every candidate that existed
# before 2026-09-30 is drawn exactly as it was.
AXIS = 8.0
t = np.arange(0, AXIS, DT)

def lobe(kr, kd):
    h = np.exp(-kd*t) - np.exp(-kr*t)
    return h/h.max()

def dc(kr, kd, sup=SUP):
    h = lobe(kr, kd).copy(); h[t >= sup] = 0
    h = h - h[t < sup].mean()*(t < sup)
    return h/h.max()

def bigamma(sf, rf, ss, rs):
    """The gamma family, built the same way: a lobe minus a scaled slower lobe."""
    def lobe(shape, rate):
        lg = (shape - 1)*np.log(np.maximum(t, 1e-12)) - rate*t
        lg[0] = -np.inf
        return np.exp(lg - lg.max())
    f, sl = lobe(sf, rf), lobe(ss, rs)
    h = f - (f.sum()/sl.sum())*sl
    return h/h.max()

def biphasic(krf, kdf, krs, kds):
    f, s = lobe(krf, kdf), lobe(krs, kds)
    h = f - (f.sum()/s.sum())*s
    return h/h.max()

CAND = [
 ("K1", "current mean-free (DC shelf)", dict(kr=2.0, kd=1.3), dc(2.0, 1.3, 5.0)),
 ("K2", "fast, DC-centred, 2 s support", dict(kr=8.0, kd=3.0, sup=2.0), dc(8.0, 3.0, 2.0)),
 ("K3", "biphasic, moderate",  dict(krf=6,  kdf=2.5, krs=2.5, kds=1.0), biphasic(6, 2.5, 2.5, 1.0)),
 ("K4", "biphasic, fast",      dict(krf=12, kdf=5.0, krs=5.0, kds=2.0), biphasic(12, 5, 5, 2)),
 ("K5", "biphasic, very fast", dict(krf=20, kdf=9.0, krs=8.0, kds=3.5), biphasic(20, 9, 8, 3.5)),
 ("K6", "biphasic, fast + shallow long tail", dict(krf=12, kdf=5.0, krs=2.0, kds=0.7), biphasic(12, 5, 2, 0.7)),
 # K7 -- K5's rate constants scaled by 0.25, added 2026-09-30, when raw .doric files
 # made it possible to check a kernel against data for the first time. K5 was chosen
 # on simulation-side criteria alone and puts 1.16% of its power above 10 Hz; the
 # recordings carry 0.024% there. That band holds neither transients nor drift, so it
 # fixes the fast end without decomposing anything, and K7 lands at 0.022%.
 #
 # Its peak is 0.52 s wide against an inter-event interval of 4.7 to 6 s, so events
 # stay ten half-widths apart and do not merge. What changes is the undershoot, which
 # runs to about 6 s at the 1% level: the trace no longer returns fully to baseline
 # between events. That bears on how the tonic is identified rather than on counting,
 # and is the thing to watch if this is adopted.
 ("K7", "biphasic, matched to the recordings", dict(krf=5.0, kdf=2.25, krs=2.0, kds=0.875),
  biphasic(5.0, 2.25, 2.0, 0.875)),
 # K8 -- least squares against the peak-triggered average of 476 isolated transients
 # from the two morphine .doric files, both kernel and data through the same 0.1-10 Hz
 # band-pass. Beats K7 (RMS 0.116/0.137 against 0.127/0.185) mostly on the rise and the
 # tail. Read it as a TIMESCALE, not four constants: profiling kdf/krf from 0.4 to 0.9
 # and refitting the rest moves the loss by 5% while the 50%-to-peak time stays at
 # 0.21-0.23 s and the FWHM at 0.60-0.69 s, against 0.21/0.29 s and 0.54/0.67 s
 # measured. These constants are the best-conditioned point on that flat ridge.
 #
 # Its peak is 0.67 s wide, so with an inter-event interval of 4.7 to 6 s events are
 # seven half-widths apart and still separable. What chooses between K7 and K8 is not
 # this residual but whether the rates are recovered in simulation.
 ("K8", "biphasic, fitted to the recordings", dict(krf=3.30, kdf=1.65, krs=1.65, kds=0.825),
  biphasic(3.30, 1.65, 1.65, 0.825)),
 # K9 -- a different FAMILY, not another point in this one: a gamma lobe minus a scaled
 # slower gamma lobe. Least squares against the same peak-triggered average as K8 and it
 # halves the residual (RMS 0.062/0.071 against 0.116/0.137), because a gamma's onset is
 # smooth where a difference of exponentials is kinked, and that kink is the largest term
 # in K8's residual. Its parameters are also identified, which K8's are not: profiling
 # the leading shape from 3 to 48 moves the loss from 0.0097 through 0.00446 to 0.0073.
 #
 # The catch is the support. Truncating before the slow lobe has decayed forces a bigger
 # subtraction and the undershoot swallows the transient -- 2.7x the peak at 2.1 s -- all
 # while sum(h) stays at zero, so mean-freeness gives no warning. bigamma_kernel refuses
 # a length that does this; here the axis is 8 s, which is past the 5.7 s it needs.
 # Its peak also sits at 1.23 s rather than 0.32 s, so event times carry a fixed lag.
 ("K9", "two gammas, fitted to the recordings", dict(sf=20.34, rf=15.65, ss=15.40, rs=3.95),
  bigamma(20.34, 15.65, 15.40, 3.95)),
]

def props(h):
    pk = int(h.argmax()); mn = int(h.argmin())
    pos = h >= 0.5*h.max()
    nz = np.where(np.abs(h) > 0.01*h.max())[0]
    zc = np.where(np.diff(np.sign(h)) != 0)[0]
    return dict(peak_t=pk*DT, fwhm=pos.sum()*DT, undershoot=float(h.min()),
                min_t=mn*DT, eff_sup=(nz[-1]+1)*DT if len(nz) else 0.0,
                area=float(h.sum()), zc1=(zc[0]+1)*DT if len(zc) else np.nan)

rows = [(n, d, props(h)) for n, d, h, in [(c[0], c[1], c[3]) for c in CAND]]
print(f"{'':3} {'kernel':36} {'peak@':>7} {'FWHM':>7} {'under':>7} {'min@':>7} {'|h|<1%':>8} {'Σh':>8}")
for n, d, p in rows:
    print(f"{n:3} {d:36} {p['peak_t']:6.2f}s {p['fwhm']:6.2f}s {p['undershoot']:+7.3f} "
          f"{p['min_t']:6.2f}s {p['eff_sup']:7.2f}s {p['area']:+8.4f}")

fig, ax = plt.subplots(2, 4, figsize=(19, 8.4))
fig.patch.set_facecolor("white")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#4a3aa7", "#eda100", "#7a5195", "#00857e", "#c2185b"]
for i, ((n, d, _, h), a) in enumerate(zip(CAND, ax.ravel())):
    p = props(h)
    a.axhline(0, color="#9a9a9a", lw=1.1, zorder=1)
    a.fill_between(t, 0, h, where=h >= 0, color=C[i], alpha=.16, lw=0)
    a.fill_between(t, 0, h, where=h < 0, color=C[i], alpha=.07, lw=0)
    a.plot(t, h, color=C[i], lw=2.1, zorder=3)
    a.plot(p["peak_t"], 1, "o", ms=5.5, color=C[i], mec="white", mew=1.4, zorder=4)
    a.set_title(f"{n} — {d}", fontsize=11.5, fontweight="bold", loc="left", pad=9)
    a.set_xlabel("time since impulse (s)", fontsize=9.5)
    a.set_ylabel("h(t)", fontsize=9.5)
    a.set_ylim(-1.05, 1.12); a.set_xlim(0, SUP)
    a.grid(alpha=.16, lw=.7); a.set_axisbelow(True)
    for s in a.spines.values(): s.set_color("#cccccc")
    a.tick_params(labelsize=9, colors="#555555")
    a.text(.975, .045, f"peak {p['peak_t']:.2f}s · FWHM {p['fwhm']:.2f}s\n"
                       f"undershoot {p['undershoot']:+.2f} @ {p['min_t']:.2f}s\n"
                       f"|h|<1% after {p['eff_sup']:.2f}s",
           transform=a.transAxes, ha="right", va="bottom", fontsize=8.6,
           family="monospace", color="#444444",
           bbox=dict(fc="white", ec="#dddddd", boxstyle="round,pad=.42"))

for a, (xl, ttl) in zip(ax.ravel()[6:], [(SUP, "all candidates"), (1.5, "all candidates — first 1.5 s")]):
    a.axhline(0, color="#9a9a9a", lw=1.1)
    for i, (n, _, _, h) in enumerate(CAND):
        a.plot(t, h, color=C[i], lw=1.9, label=n, alpha=.9)
    a.set_title(ttl, fontsize=11.5, fontweight="bold", loc="left", pad=9)
    a.set_xlim(0, xl); a.set_ylim(-1.05, 1.12); a.grid(alpha=.16, lw=.7); a.set_axisbelow(True)
    a.set_xlabel("time since impulse (s)", fontsize=9.5); a.set_ylabel("h(t)", fontsize=9.5)
    a.legend(fontsize=9, ncol=2, framealpha=.95, edgecolor="#dddddd")
    for s in a.spines.values(): s.set_color("#cccccc")
    a.tick_params(labelsize=9, colors="#555555")

fig.suptitle("Candidate transient kernels — all mean-free (Σh = 0), all peak-normalised to 1",
             fontsize=14, fontweight="bold", x=.5, y=.985)
fig.tight_layout(rect=[0, 0, 1, .965])
# PDF is the deliverable; the PNG is written alongside for quick viewing.
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor="white",
            bbox_inches="tight")
json.dump({n: dict(desc=d, params=c[2], **props(h))
           for (n, d, p), c, h in zip(rows, CAND, [c[3] for c in CAND])},
          open(OUT.with_suffix(".json"), "w"), indent=2, default=float)
print(f"\nwrote {OUT} + .png + .json")
