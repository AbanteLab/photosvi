"""Paper figure: the nociceptin cohort at both time scales.

    python scripts/paper_fig_realcohort.py --fit runs_real_final/full_noDL56sali.json \
        --out .../fig_realcohort.pdf --macros .../realcohort_fig_numbers.tex

Three panels, all from the fit's own output (the JSON and its _arrays.npz), so the figure
shows exactly the recordings and window the fit saw -- both batches, after any exclusion
(DL56's morphine-only session is excluded; that animal has one trace).
(a) every recording around its morphine injection, robustly standardized per session, the
RO64-6198 session of each animal above its vehicle session, the excluded window shaded.
(b) the mean dF/F of each condition in 30-s bins, each session referenced to its
pre-injection level, with the mean tonic component the fit inferred for the same sessions.
(c) the event rate in time bins, as the posterior expected number of events per minute,
mean +- s.e.m. over the sessions of each condition, with the excluded window left out.
"""
import argparse, json, os, pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
# TrueType, not Type 3: the text stays selectable, and journals that reject Type 3 accept it.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fit", default="runs_real_final/full_noDL56sali.json")
ap.add_argument("--out", default="diagnostics/fig_realcohort.pdf")
ap.add_argument("--bins", type=int, default=10, help="time bins for the event rate")
ap.add_argument("--macros", default=None, help="write the caption's macros here")
ap.add_argument("--window", type=float, default=240.0,
                help="seconds either side of the injection to draw in a")
a = ap.parse_args()

d = json.load(open(a.fit))
A = np.load(a.fit.replace(".json", "_arrays.npz"))
FS = float(d["fs"])
Y, X, Q, M = A["y"], A["x_session"], A["q"].astype(float), A["mask"].astype(bool)
names = [str(n) for n in A["group_names"]]
grp = A["drug"]
S_, T_ = Y.shape
inj0 = int(A["inj_t"][0])
sess = d["sessions"]
assert len(sess) == S_, "sessions and arrays disagree"
C_NOCI, C_SALI, C_MASK = "#2a78d6", "#8a8f98", "#b3453a"
cols = {"nociceptin": C_NOCI, "saline": C_SALI}
lab = {"nociceptin": "RO64-6198", "saline": "vehicle"}
half = int(a.window * FS)

fig = plt.figure(figsize=(12.0, 6.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], wspace=0.30, hspace=0.42)
fig.patch.set_facecolor("white")

# --- a: the recordings, each around its own injection ---------------------------------
# Grouped by animal: one band per animal, its vehicle session above its
# RO64-6198 session, the animal named once; condition is carried by colour (one legend for
# the figure). An animal without one of the two sessions keeps the empty row, labelled, so
# the pairs stay aligned. The whole recording is drawn, in minutes from the morphine
# injection like b and c: the fitted arrays hold only the analysed window, so the traces are
# reloaded with the same preprocessing (not detrended: a line fitted over the whole
# recording would tilt it). Each row is shaded from the start of the recording to the
# pretreatment injection, which is marked; the analysed window starts at the dotted line.
ax = fig.add_subplot(gs[:, 0])
from photosvi.data import bonaventura3 as _b3  # noqa: E402
import h5py as _h5  # noqa: E402
import os as _os  # noqa: E402
import re as _re2  # noqa: E402


def _inj1_rel_s():
    """Pretreatment injection, seconds from the morphine injection, per (animal, cond)."""
    out = {}
    for _batch, cond, f, matref in _b3._session_files(("batch1", "batch2")):
        an = _re2.search(r"DL(\d+)", _os.path.basename(f), _re2.I).group(0).upper()
        with _h5.File(f, "r") as h:
            ev = _b3.doric_events(h) if _b3.doric_v6(h) else None
        if ev and "inj 2" in ev and "inj 1" in ev:
            i1, i2 = ev["inj 1"][0], ev["inj 2"][0]
        else:
            t = (_b3.injections(matref, an) if _os.path.isdir(matref)
                 else _b3._mat_injections(matref))
            i1, i2 = t[0], t[2]
        out[(an, cond)] = float(i1) - float(i2)
    return out


INJ1 = _inj1_rel_s()
PRE_FULL = min(-v for v in INJ1.values()) + 600.0       # covers every pretreatment baseline
_full = {(s_.animal, s_.cond): s_ for s_ in _b3.load_sessions(
    min(PRE_FULL, 1500.0), verbose=False, detrend_window=False, batches=("batch1", "batch2"))}
_key = lambda an: int("".join(ch for ch in an if ch.isdigit()) or 0)  # noqa: E731
animals = sorted({s_["animal"] for s_ in sess}, key=_key)
# Animals are shown anonymized, numbered in the order of their IDs, with
# their sex once it is recorded here (e.g. {"DL4": "\u2640"}).
ANIMAL_SEX: dict[str, str] = {}
fitted = {(s_["animal"], s_["cond"]) for s_ in sess}
ROW, GAP = 20.0, 14.0
y0, ticks, labels = 0.0, [], []
t_lo = None
mask_end = ((np.where(~M[0])[0][-1] - inj0) / FS / 60) if (~M[0]).any() else 0.0
ax.axvspan(0, mask_end, color=C_MASK, alpha=0.14, lw=0, zorder=1)
for k, an in enumerate(animals):
    top = y0
    for cond, g in (("sali", "saline"), ("noci", "nociceptin")):
        if (an, cond) not in fitted:
            ax.text(5.0, y0, f"{lab[g]} session excluded", ha="center", va="center",
                    fontsize=7.5, color="#9a9a9a", style="italic", zorder=3)
        else:
            se = _full[(an, cond)]
            y = se.y.astype(float)
            y = (y - np.median(y)) / (np.median(np.abs(y - np.median(y))) / 0.6745)
            t = (np.arange(len(y)) - se.inj_bin) / FS / 60
            t_lo = t[0] if t_lo is None else max(t_lo, t[0])
            # clipped to its own row: a handling artefact must not cross into the next one
            ax.plot(t, np.clip(y, -0.48 * ROW, 0.48 * ROW) + y0, lw=0.3, color=cols[g], zorder=2)
            t1 = INJ1[(an, cond)] / 60
            ax.fill_between([t[0], t1], y0 - 0.45 * ROW, y0 + 0.45 * ROW, color="#e9dfc6",
                            alpha=0.55, lw=0, zorder=1)
            ax.plot([t1], [y0 + 0.42 * ROW], marker="v", ms=4, color="#333", zorder=4)
        y0 -= ROW
    bottom = y0 + ROW
    if k % 2 == 0:
        ax.axhspan(bottom - 0.5 * ROW, top + 0.5 * ROW, color="#eef0f3", lw=0, zorder=0)
    ticks.append(0.5 * (top + bottom))
    labels.append(f"{k + 1}{ANIMAL_SEX.get(an, '')}")
    y0 -= GAP
ax.axvline(0, color="#333", lw=1.0, zorder=3)
ax.axvline(-d["pre_s"] / 60, color="#333", lw=0.8, ls=(0, (2, 2)), zorder=3)
ax.set_yticks(ticks)
ax.set_yticklabels(labels, fontsize=9)
ax.tick_params(axis="y", length=0)
ax.set_ylim(y0 + GAP - 0.5 * ROW, 0.5 * ROW + 4)
ax.set_xlabel("time from the morphine injection (min)")
ax.set_xlim(t_lo, tmin_end := (T_ - inj0) / FS / 60)
ax.set_title("a", loc="left", fontweight="bold", fontsize=12, x=-0.08)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)

# --- b and c ----------------------------------------------------------------------------
axb = fig.add_subplot(gs[0, 1])
axc = fig.add_subplot(gs[1, 1])
tmin = (np.arange(T_) - inj0) / FS / 60
masked = ~M
mask_hi = max(float(tmin[np.where(masked[i])[0]].max()) for i in range(S_) if masked[i].any())
for ax_ in (axb, axc):
    ax_.axvspan(0, mask_hi, color=C_MASK, alpha=0.18, lw=0, zorder=0)
w = int(30 * FS)
nb = T_ // w
for gi, gname in enumerate(names):
    rows_y, rows_x = [], []
    for i in np.where(grp == gi)[0]:
        y = np.where(M[i], Y[i], np.nan) - np.nanmedian(Y[i, :inj0])
        rows_y.append(np.nanmean(y[:nb * w].reshape(nb, w), axis=1))
        rows_x.append(X[i] - X[i, :inj0].mean())
    tb = tmin[:nb * w].reshape(nb, w).mean(axis=1)
    axb.plot(tb, np.nanmean(rows_y, axis=0), lw=0.9, color=cols[gname], alpha=0.55, zorder=2)
    axb.plot(tmin, np.mean(rows_x, axis=0), lw=2.6, color=cols[gname], zorder=4,
             label=f"{lab[gname]} (n = {len(rows_x)})")
axb.axvline(0, color="#333", lw=1.0, zorder=3)
axb.axhline(0, color="#bbb", lw=0.6, zorder=1)
axb.set_ylabel("$\\Delta F/F$ from pre-injection")
axb.set_xlim(tmin[0], tmin[-1])
edges = np.linspace(0, T_, a.bins + 1).astype(int)
centre = ((edges[:-1] + edges[1:]) / 2 - inj0) / FS / 60
width = (edges[1] - edges[0]) / FS / 60
for k, (gi, gname) in enumerate(enumerate(names)):
    rates = []
    for i in np.where(grp == gi)[0]:
        rates.append([Q[i, lo:hi][M[i, lo:hi]].sum() / (M[i, lo:hi].sum() / FS / 60)
                      for lo, hi in zip(edges[:-1], edges[1:])])
    rates = np.asarray(rates)
    mu, se = rates.mean(0), rates.std(0, ddof=1) / np.sqrt(len(rates))
    off = (k - 0.5) * 0.18 * width
    axc.errorbar(centre + off, mu, yerr=se, fmt="o-", ms=4, lw=1.4, capsize=2,
                 color=cols[gname], label=lab[gname])
axc.axvline(0, color="#333", lw=1.0, zorder=3)
axc.set_ylabel("events per minute")
axc.set_xlabel("time from the morphine injection (min)")
axc.set_xlim(tmin[0], tmin[-1])
axc.set_ylim(bottom=0)
axb.set_xticklabels([])
for ax_, l in ((axb, "b"), (axc, "c")):
    ax_.set_title(l, loc="left", fontweight="bold", fontsize=12, x=-0.16)
    for side in ("top", "right"):
        ax_.spines[side].set_visible(False)
    ax_.grid(axis="y", color="#eef1f4", lw=1)
    ax_.set_axisbelow(True)

# One legend for the three panels: the colours mean the same in a, b and c,
# and n is the number of sessions of each condition.
from matplotlib.lines import Line2D  # noqa: E402
_n = {g: int(sum(names[grp[i]] == g for i in range(S_))) for g in ("saline", "nociceptin")}
fig.legend(handles=[Line2D([], [], color=cols[g], lw=2.2, label=f"{lab[g]} (n = {_n[g]})")
                    for g in ("saline", "nociceptin")],
           loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=2, frameon=False, fontsize=9.5)

if a.macros:
    pathlib.Path(a.macros).write_text(
        "% Generated by scripts/paper_fig_realcohort.py -- do not edit.\n"
        f"\\newcommand{{\\ppRealBins}}{{{a.bins}}}\n")
out = pathlib.Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, transparent=True, bbox_inches="tight")  # PDF: no white patch over LaTeX labels
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"  wrote {out} and {out.with_suffix('.png')}")
