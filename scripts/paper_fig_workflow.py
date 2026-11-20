#!/usr/bin/env python3
"""Figure 1: from a recording to the estimand.

    python scripts/paper_fig_workflow.py --out diagnostics/fig_workflow.pdf

Replaces the box-and-arrow schematic. The traces are real output of the generative
model rather than drawn squiggles, so what the reader sees is the thing the method
is claimed to decompose -- at the two time scales that make the problem hard, forty
minutes of pharmacokinetics and tenths of a second of transient.

The plate diagram that used to share this figure moved to the supplement
(`paper_fig_pgm.py`): the main text needs the workflow, and a reader who wants the
conditional structure wants it complete rather than compressed into half a figure.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
)
from matplotlib.path import Path as MPath  # noqa: E402

INK, MUTED = "#2b2b2b", "#8a8a8a"
TONIC, PHASIC, DATA = "#c2622d", "#3b6ea5", "#4a4a4a"
FUR, FUR_D = "#e6e0d8", "#c9c1b6"
FIBER = "#2f8f6f"

ap = argparse.ArgumentParser()
# The trace is drawn from the paper's final cohort model (48 Hz, AR(2) noise, marks in noise
# units at medium SNR; 2026-10-28). It was drawn from cohort_comparison.yaml before, at 25 Hz
# with white noise and an SNR near 50, which made the noise vanish on every transient.
ap.add_argument("--data", default="diagnostics/fig1_trace_final.npz")
ap.add_argument("--out", default="diagnostics/fig_workflow.pdf")
ap.add_argument("--dpi", type=int, default=200)
ap.add_argument("--config", default="configs/final_cohort_48.yaml")
# The kernel comes from its own config so that refitting it does not mean touching this
# figure. Panel D measures what it draws rather than asserting it, so a reparameterisation
# propagates into the annotation without a caption edit.
ap.add_argument("--kernel-config", default="configs/final_detection_48.yaml",
                help="config whose `kernel` block supplies the basis drawn in panel D")
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--snr", type=float, default=None,
                help="median mark over sigma_y for the drawn trace; default the config's")
ap.add_argument("--zoom-t", type=float, default=2052.8,
                help="zoom start, s: the 25-s window with the most events in the final trace")
ap.add_argument("--zoom-dur", type=float, default=25.0)
a = ap.parse_args()

# Self-contained: cache the representative session if it is not already there, so
# the figure regenerates from a clean checkout without a separate preparation step.
if not Path(a.data).exists():
    import torch
    import yaml

    from photosvi.experiments.cohort import (
        OPIOID1,
        simulate_cohort,
        spec_from_config,
    )
    from photosvi.models.points import (
        AdditiveEmission,
        PhasicEventProcess,
        TonicBaseline,
        build_kernel,
    )

    cfg = yaml.safe_load(open(a.config))
    cfg["cohort"]["n_animals"] = 1
    spec = spec_from_config(cfg, 0.40)
    d_ = OPIOID1                       # the slow opioid: a carrier you can see
    coh = simulate_cohort(spec, a.seed)
    inj_ = coh["inj_t"][d_]
    kcfg = yaml.safe_load(open(a.kernel_config))
    kspec = dict(kcfg["kernel"]["fit"] if "fit" in kcfg["kernel"] else kcfg["kernel"])
    klen = int(kcfg["kernel"].get("length", spec.kernel_length))
    # `length` in that config is in bins at ITS sampling rate; convert to this one so the
    # support in seconds is preserved whatever rate either config uses.
    kfs = float(kcfg.get("data", {}).get("fs", 1.0 / spec.dt))
    klen = max(8, int(round(klen / kfs * (1.0 / spec.dt))))
    kern_ = build_kernel(kspec, klen, spec.dt)
    x_ = TonicBaseline(dt=spec.dt, b=spec.tonic_b, peak=spec.tonic_peak[d_],
                       k_rise=spec.tonic_k_rise[d_], k_decay=spec.tonic_k_decay[d_],
                       drift=spec.tonic_drift).baseline(spec.T, inj_)
    g_ = torch.Generator().manual_seed(a.seed * 10001)
    ex_ = PhasicEventProcess(kern_, beta0=coh["beta0_true"][d_],
                             beta_inj=coh["beta_inj_true"][d_], mu_a=spec.mu_a,
                             sigma_a=spec.sigma_a, dt=spec.dt,
                             history=None).sample(spec.T, inj_, 1, g_)
    y_ = AdditiveEmission(sigma_y=(math.exp(spec.mu_a) / a.snr if a.snr else spec.sigma_y),
                          ar_coeffs=spec.ar_coeffs).sample(x_ + ex_["c"], generator=g_)["y"][0]
    Path(a.data).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.data, y=y_.numpy(), tonic=x_.detach().numpy(), kernel=kern_.numpy(),
             phasic=ex_["c"][0].numpy(), s=ex_["s"][0].numpy(), a=ex_["a"][0].numpy(),
             inj=inj_, fs=spec.fs, dt=spec.dt, T=spec.T)
    print(f"cached {a.data}")

D = np.load(a.data)
y, tonic, kern = D["y"], D["tonic"], D["kernel"]
phasic, s_true, a_true = D["phasic"], D["s"], D["a"]
fs, dt, inj = float(D["fs"]), float(D["dt"]), int(D["inj"])
t = np.arange(len(y)) * dt


def arrow(ax, p, q, *, color=INK, lw=1.1, rad=0.0, shrink=3.0, ms=12):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=shrink, shrinkB=shrink, zorder=5,
                                 clip_on=False))


fig = plt.figure(figsize=(10.6, 6.9))
gs = fig.add_gridspec(
    3, 3, height_ratios=[0.78, 0.92, 0.86], width_ratios=[0.90, 1.55, 1.05],
    left=0.058, right=0.982, top=0.965, bottom=0.07, hspace=0.48, wspace=0.32)

# Panel A, a schematic of the rig, was removed on 2026-10-05: the recording
# below is what the paper argues about, and the mouse said nothing the
# methods do not. The whole-recording panel takes the row.
ax_full = fig.add_subplot(gs[0, :])
ax_zoom = fig.add_subplot(gs[1, :2])
ax_kern = fig.add_subplot(gs[1, 2])
ax_dec = fig.add_subplot(gs[2, :2])
ax_out = fig.add_subplot(gs[2, 2])

for ax in (ax_out,):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
for ax in (ax_full, ax_zoom, ax_kern, ax_dec):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bbbbbb")
    ax.tick_params(labelsize=8, colors="#555555")

# ------------------------------------------------- A: the whole recording
ax_full.text(-0.055, 1.12, "a", transform=ax_full.transAxes, fontsize=14,
             weight="bold", color=INK, va="top")
step = 10
ax_full.plot(t[::step] / 60, y[::step], lw=0.35, color=DATA, alpha=0.75)
ax_full.plot(t / 60, tonic, lw=1.9, color=TONIC)
ax_full.axvline(inj * dt / 60, lw=1.1, color=INK, ls=(0, (4, 3)))
ax_full.text(inj * dt / 60 + 0.4, ax_full.get_ylim()[1] * 0.92, "injection",
             fontsize=7.6, color=INK, va="top")
z0, z1 = a.zoom_t / 60, (a.zoom_t + a.zoom_dur) / 60
ax_full.axvspan(z0 - 0.12, z1 + 0.12, color=PHASIC, alpha=0.22, lw=0)
ax_full.annotate("b, d", xy=((z0 + z1) / 2, ax_full.get_ylim()[0]),
                 xytext=((z0 + z1) / 2, ax_full.get_ylim()[0] * 1.35),
                 ha="center", va="top", fontsize=7.6, color=PHASIC,
                 arrowprops=dict(arrowstyle="-", color=PHASIC, lw=0.9))
ax_full.set_xlabel("time (min)", fontsize=8.5)
ax_full.set_ylabel(r"$\Delta F/F$", fontsize=8.5)

# --------------------------------------------------------- B: the transient scale
ax_zoom.text(-0.035, 1.12, "b", transform=ax_zoom.transAxes, fontsize=14,
             weight="bold", color=INK, va="top")
i0, i1 = int(a.zoom_t * fs), int((a.zoom_t + a.zoom_dur) * fs)
tz = t[i0:i1] - t[i0]
ax_zoom.plot(tz, y[i0:i1], lw=0.9, color=DATA)
ax_zoom.plot(tz, tonic[i0:i1], lw=1.8, color=TONIC)
ax_zoom.set_xlabel("time within the shaded window (s)", fontsize=8.5)
ax_zoom.set_ylabel(r"$\Delta F/F$", fontsize=8.5)

# ------------------------------------------------------------ C: the kernel
ax_kern.text(-0.16, 1.12, "c", transform=ax_kern.transAxes, fontsize=14,
             weight="bold", color=INK, va="top")
tk = np.arange(len(kern)) * dt
ax_kern.plot(tk, kern, lw=1.8, color=PHASIC)
ax_kern.axhline(0, lw=0.8, color=MUTED)
ax_kern.set_xlabel("time (s)", fontsize=8.5)
ax_kern.set_ylabel(r"$h_k$", fontsize=8.5)
# Measured from the kernel being drawn, not written down: half-width at half maximum, the
# time of the peak, and the depth of the undershoot. Refitting the basis moves these
# numbers in the figure without anyone having to remember to edit a caption.
_pk = int(np.argmax(kern))
_hm = np.flatnonzero(kern >= 0.5 * kern.max())
_fwhm = (_hm[-1] - _hm[0] + 1) * dt
ax_kern.annotate("", xy=(_hm[0] * dt, 0.5 * kern.max()), xytext=(_hm[-1] * dt, 0.5 * kern.max()),
                 arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8, shrinkA=0, shrinkB=0))
ax_kern.text(_hm[-1] * dt + 0.2, 0.5 * kern.max(), f"FWHM {_fwhm * 1e3:.0f} ms",
             ha="left", va="center", fontsize=7.2, color=MUTED)
ax_kern.text(0.97, 0.06,
             f"peak {_pk * dt:.2f} s\nundershoot ${kern.min():.2f}$",
             transform=ax_kern.transAxes, ha="right", va="bottom", fontsize=7.0,
             color=MUTED, linespacing=1.4)

# ----------------------------------------- D: the latent process underneath it
# Not the trace minus the carrier -- over 25 s the carrier is flat, so that would
# repeat panel C shifted. What C does not show is the thing the model estimates:
# a sparse train of events, each with an amplitude, convolved with the kernel.
ax_dec.text(-0.035, 1.12, "d", transform=ax_dec.transAxes, fontsize=14,
            weight="bold", color=INK, va="top")
idx = np.nonzero(s_true[i0:i1])[0]
ax_dec.vlines(idx * dt, 0, a_true[i0:i1][idx], lw=1.4, color=INK, zorder=4)
ax_dec.scatter(idx * dt, a_true[i0:i1][idx], s=13, color=INK, zorder=5)
ax_dec.plot(tz, phasic[i0:i1], lw=1.0, color=PHASIC, alpha=0.85, zorder=3)
ax_dec.axhline(0, lw=0.8, color=MUTED)
ax_dec.set_xlabel("time within the shaded window (s)", fontsize=8.5)
ax_dec.set_ylabel("mark  " + r"$a_t$", fontsize=8.5)

# --------------------------------------------------------------- E: the estimand
ax_out.set_xlim(0, 1); ax_out.set_ylim(-0.04, 1.0)
ax_out.text(0.0, 1.12, "e", transform=ax_out.transAxes, fontsize=14, weight="bold",
            color=INK, va="top")


def obox(x, y, w, h, txt, ec=INK, fs=8.0):
    ax_out.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc="white", ec=ec, lw=1.1))
    ax_out.text(x, y, txt, ha="center", va="center", fontsize=fs, color=INK,
                linespacing=1.5)


obox(0.5, 0.90, 0.92, 0.16, "stochastic variational\ninference")
# Three readouts of one posterior (the tonic contrasts too).
obox(0.5, 0.62, 0.92, 0.13, "per-bin events  " + r"$q(s_t)$", ec=PHASIC)
obox(0.5, 0.36, 0.92, 0.19,
     "rate ratio  " + r"$e^{\beta_{\rm inj}}$" + "\ncohort contrast  "
     + r"$\mu_{d_1}\!-\!\mu_{d_2}$", ec=PHASIC)
obox(0.5, 0.08, 0.92, 0.19,
     "tonic response  " + r"$P, k_{\rm rel}, k_{\rm cl}$" + "\ncohort contrast  "
     + r"$\bar\psi_{d_1}\!-\!\bar\psi_{d_2}$", ec=TONIC)
arrow(ax_out, (0.5, 0.82), (0.5, 0.69), ms=9)
arrow(ax_out, (0.5, 0.555), (0.5, 0.455), ms=9)
arrow(ax_out, (0.5, 0.265), (0.5, 0.175), ms=9)


out = Path(a.out)
fig.savefig(out, transparent=True, bbox_inches="tight")  # PDF: no white patch over LaTeX labels
fig.savefig(out.with_suffix(".png"), dpi=a.dpi, facecolor="white", bbox_inches="tight")
print(f"wrote {out} and {out.with_suffix('.png')}")
