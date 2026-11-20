#!/usr/bin/env python3
"""The third Bonaventura batch: five animals, two conditions, fitted as a cohort.

    DESIGN=full      python scripts/cohort_bonaventura3.py --out runs_real/full.json
    DESIGN=partial   python scripts/cohort_bonaventura3.py --out runs_real/partial.json

Five animals each recorded twice -- once with nociceptin before the morphine and once with
saline -- so the condition contrast is **within animal** and the between-animal spread
cancels out of it. That is the design the hierarchical model exists for, and the point of
running it two ways is to measure what the crossing buys on real recordings rather than in
simulation.

`DESIGN=partial` hides sessions so that each animal carries ONE condition instead of two,
which turns the same contrast into a between-animal comparison. It drops recordings rather
than animals, so the two arms differ in design and not in how much signal there is to fit:
the full arm has ten recordings from five animals, the partial arm has five from five. That
is not a matched comparison on recording count -- with five animals it cannot be, since
hiding to five recordings is the only way to break the crossing -- and the difference in
interval width therefore mixes the design change with the halved data. Both are reported.

**The analysis window.** Ten minutes before the last injection to the end of the recording,
which is 10 + 30.6 min. The last injection is always morphine; the first is the condition.
The handling window, 15 to 80 s depending on the session, is MASKED rather than trimmed --
the model's only way to explain a large fast excursion is events, and that window sits
exactly where the rate contrast is read.

**The recordings are loaded by `photosvi.data.bonaventura3`**, which is where the channel
discovery, the injection times and the window live, so that this script and the paper's
calibration generator cannot end up describing two different sets of ten windows.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os

import numpy as np
import pyro
import torch
import yaml
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from scipy.signal import butter, detrend, filtfilt, find_peaks

from photosvi.data.bonaventura3 import FS, load_sessions
from photosvi.experiments.cohort import build_cohort_model, spec_from_config
from photosvi.inference.points import EstimandPlateau, prepare, resolve_device
from photosvi.inference.points.fitquality import FitQuality, block_spread
from photosvi.models.points.emission import estimate_log_noise_scale
from photosvi.models.points.points import _MARK_SCALE_MAX

PRE_S = 600.0

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--config", default="configs/cohort_comparison.yaml")
ap.add_argument("--kernel-config", default="configs/simulator_2026_09_30.yaml")
ap.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", 40000)))
a = ap.parse_args()
DESIGN = os.environ.get("DESIGN", "full")
DEVICE = os.environ.get("DEVICE", "cpu")
# The random-effect levels, as in scripts/cohort_hierarchical.py, so the real cohort can
# be fitted under the same structure as the simulated ones. Empty means the model's own
# default. LEVELS=animal,cell drops the per-session baseline variance.
LEVELS = tuple(x for x in os.environ.get("LEVELS", "").split(",") if x) or None
# Overrides the config's floor on the gated mark divergence. Here so the fit can be run at
# several floors and the answer checked for sensitivity to it, which is the test that
# separates settling an identification problem from tuning until a number looks right.
GATE_FLOOR = os.environ.get("GATE_FLOOR")
# Raises the floor below which the plateau rule may not fire. The rule watches the estimand
# and not its uncertainty, and on this cohort the contrast went flat while the reported
# standard deviation was still growing ten percent per few thousand steps.
# For a fit whose product is an interval, that is stopping early.
MIN_STEPS = os.environ.get("MIN_STEPS")
# Seed the global block at its prior scale instead of 0.1. The posterior WIDTH,
# not the estimate, is what this is for: see
BLOCK_PRIOR = os.environ.get("BLOCK_PRIOR")


# DETREND=0 leaves the slow trend to the model's drift; MASK_MIN_S extends the handling
# window (see photosvi.data.bonaventura3.load_sessions).
DETREND = os.environ.get("DETREND", "1") != "0"
MASK_MIN_S = float(os.environ.get("MASK_MIN_S", "0"))
# BATCHES picks the recordings (batch1: the first five animals; batch2: DL4, DL5, 2026-10);
# EXCLUDE drops sessions as ANIMAL:cond pairs, e.g. EXCLUDE=DL56:sali (DL56's
# morphine-only session, excluded for a technical problem). ALS_LAMBDA overrides the loader's ALS smoothness per fs.
BATCHES = tuple(x for x in os.environ.get("BATCHES", "batch1").split(",") if x)
EXCLUDE = {tuple(x.split(":")) for x in os.environ.get("EXCLUDE", "").split(",") if x}
_als = os.environ.get("ALS_LAMBDA")
sessions = load_sessions(PRE_S, detrend_window=DETREND, mask_min_s=MASK_MIN_S,
                         batches=BATCHES,
                         **({"als_lambda_per_fs": float(_als)} if _als else {}))
if EXCLUDE:
    gone = [s for s in sessions if (s.animal, s.cond) in EXCLUDE]
    if len(gone) != len(EXCLUDE):
        raise SystemExit(f"EXCLUDE {sorted(EXCLUDE)} matched {len(gone)} sessions")
    sessions = [s for s in sessions if (s.animal, s.cond) not in EXCLUDE]
    print(f"  EXCLUDE: {sorted(EXCLUDE)} -> {len(sessions)} sessions")
# One T for the whole cohort: the model takes [S, T] and the windows differ by seconds.
T = min(len(s.y) for s in sessions)
print(f"  retallant totes a {T} caselles = {T/FS/60:.1f} min")

if DESIGN == "partial":
    # Hide one condition per animal, alternating which, so the condition is still
    # represented in the cohort but no animal carries both and the contrast is
    # between-animal.
    keep = []
    for i, an in enumerate(sorted({s.animal for s in sessions})):
        want = "noci" if i % 2 == 0 else "sali"
        keep += [s for s in sessions if s.animal == an and s.cond == want]
    sessions = keep
    print(f"  DESIGN=partial: {len(sessions)} sessions, "
          + ", ".join(f"{s.animal}={s.cond}" for s in sessions))

animals = sorted({s.animal for s in sessions})
Y = torch.stack([torch.as_tensor(s.y[:T], dtype=torch.float32) for s in sessions])
coh = dict(y=Y,
           inj_t=[s.inj_bin for s in sessions],
           animal=[animals.index(s.animal) for s in sessions],
           drug=[0 if s.cond == "sali" else 1 for s in sessions])
# Index -> condition, so anything reading the output does not have to know the
# convention above.
CONDS = {0: "saline", 1: "nociceptin"}

# The handling window, masked per session. `observed_bins` widens it under a correlated
# emission; see its docstring for why one bin is not enough.
mask = torch.ones(len(sessions), T, dtype=torch.bool)
for i, s in enumerate(sessions):
    lo, hi = s.mask_bins
    mask[i, max(0, lo):min(T, hi + 1)] = False
print(f"  emmascarades {int((~mask).sum())} caselles de {mask.numel()} "
      f"({100*float((~mask).float().mean()):.2f}%)")

cfg = yaml.safe_load(open(a.config))
kcfg = yaml.safe_load(open(a.kernel_config))
if GATE_FLOOR is not None:
    cfg["fit"]["mark_gate_floor"] = float(GATE_FLOOR)
if BLOCK_PRIOR is not None:
    cfg["fit"]["block_init_at_prior"] = BLOCK_PRIOR not in ("0", "", "false")
cfg["cohort"]["sigma_y"] = 0.0  # replaced below; the real scale is estimated per cohort
spec = spec_from_config(cfg, 0.20)
kspec = dict(kcfg["kernel"]["fit"])
klen = int(round(kcfg["kernel"]["length"] / kcfg["data"]["fs"] * FS))
# sigma_y from the recordings themselves, corrected for the correlation -- the MAD of
# successive differences assumes white noise and reads sigma * sqrt(1 - rho_1) otherwise.
phi = list(kcfg["emission"]["ar_coeffs"])
sig = float(np.exp(np.mean([estimate_log_noise_scale(
    torch.as_tensor(s.y[:int(PRE_S*FS)], dtype=torch.float32).reshape(1, -1),
    ar_coeffs=phi) for s in sessions])))
# mu_a from the recordings too, and NOT from the config. Leaving it at the simulator's
# value while taking sigma_y from the data put the expected transient at 5.85 against a
# noise of 0.013 -- a ratio of 450 -- and the mark posterior diverged to NaN within a few
# thousand steps. The two have to come from the same place: the kernel is peak-normalised,
# so a mark of a produces a peak of a, and the median prominence of the isolated peaks in
# the pre-injection stretch is therefore a direct estimate of exp(mu_a).
# sigma_a has to come from the same place, and leaving it at the simulator's 0.212 is what
# made the second failure. The mark KL is gated by the event probability, so in the bins
# holding no event -- almost all of them -- the gate is at its floor of 1e-6 and nothing
# constrains the posterior's WIDTH. With a prior that narrow against data this variable it
# drifts: measured, q(a)'s scale sat near 1 to step 10000 and then ran 3 -> 17 -> 30 by
# 11128, at which point exp(loc + 4 scale) passed the largest float32 and one infinite mark
# met the gate that was supposed to make it harmless, giving 0 * inf = NaN. Every parameter
# was still finite, which is why a finiteness check on them caught nothing.
#
# The spread is measured the same way as the location, from the same peaks. A truncated
# LogNormal MLE was tried and rejected: the detector only reports peaks above three local
# sigma, so the fit extrapolates the unobserved mass below the threshold from the shape
# assumption alone and returns a median transient an eighth of the noise, which the data
# cannot support. What these recordings measure is the spread of the transients that can be
# seen, and that is what the prior describes.
peaks, spreads = [], []
for sess in sessions:
    pre = detrend(sess.y[:int(PRE_S * FS)], type="linear")
    z = filtfilt(*butter(3, [0.1 / (FS / 2), 10.0 / (FS / 2)], "band"), pre)
    sd = np.median(np.abs(z)) / 0.6745
    _, props = find_peaks(z, prominence=3 * sd, width=max(1, int(0.04 * FS)))
    peaks.append(float(np.median(props["prominences"])))
    spreads.append(float(np.std(np.log(props["prominences"]), ddof=1)))
mu_a = float(np.log(np.median(peaks)))
sigma_a = float(np.median(spreads))
# Noise-unit marks (opt-in, experiments.marks): fixed at a_t = sigma_y r_t with the
# geometric-mean sigma_y above; build_cohort_model holds them there.
from photosvi.experiments.marks import noise_unit_marks, noise_unit_mode  # noqa: E402, I001
if noise_unit_mode(cfg["fit"]) is not None:
    mu_a, sigma_a = noise_unit_marks(sig, *noise_unit_mode(cfg["fit"]))
spec = dataclasses.replace(spec, fs=int(FS), kernel=kspec, kernel_length=klen,
                           ar_coeffs=phi, sigma_y=sig, mu_a=mu_a,
                           sigma_a=sigma_a)
print(f"  sigma_y {sig:.5f}   mu_a {mu_a:.3f} (pic tipic {np.exp(mu_a):.4f}, "
      f"{np.exp(mu_a)/sig:.1f}x el soroll)   sigma_a {sigma_a:.3f}   phi {phi}   "
      f"nucli {kspec['family']} {klen} caselles")

model, Yb, _ = build_cohort_model(cfg, spec, coh, levels=LEVELS)
device = resolve_device(DEVICE, batch=len(sessions))
model, Yb = prepare(model, Yb, device)
mask = mask.to(Yb.device)

pyro.clear_param_store()
pyro.set_rng_seed(0)
f = cfg["fit"]
svi = SVI(model.model, model.guide, ClippedAdam({"lr": f["lr"]}),
          Trace_ELBO(max_plate_nesting=2))
plateau = EstimandPlateau(
    tol=f["stop_tol"], window=f["stop_window"],
    min_steps=min(int(MIN_STEPS) if MIN_STEPS else f["stop_min_steps"], a.steps // 2))
quality = FitQuality()
traj = []


def _mark_reach():
    """How far the mark posterior's tail reaches, and whether exp of it overflows.

    The suspicion this measures: every exp in the model is bounded except the one in the
    mark's LogNormal. `q_a_loc` stays modest, but a LogNormal draw is
    exp(loc + scale * eps), so an unbounded SCALE puts a plausible draw past the largest
    float32 -- exp(89) is already infinite -- and one infinite mark makes the whole mean
    NaN with every parameter still finite, which is exactly the failure seen.
    """
    ps = pyro.get_param_store()
    if "q_a_loc" not in ps:
        return None
    loc, scale = ps["q_a_loc"], ps["q_a_scale"]
    reach = float((loc + 4.0 * scale).max())
    limit = math.log(torch.finfo(loc.dtype).max)
    return dict(loc=float(loc.max()), scale=float(scale.max()), reach=reach,
                limit=round(limit, 1), overflows=reach > limit)


def _magnitudes():
    """The largest magnitude in each parameter, biggest first.

    Printed alongside the contrast because the failure this guards against is not a NaN
    gradient but an OVERFLOW: a mark location of 90 is a perfectly finite float whose
    exponential is not, in single precision, so the parameters can all pass a finiteness
    check one step before the mean becomes NaN. Watching the magnitudes shows the drift
    coming instead of only its arrival.
    """
    m = [(float(v.abs().max()), n) for n, v in _unconstrained().items() if v.numel()]
    return sorted(m, reverse=True)


def _unconstrained():
    """The tensors the optimiser actually updates.

    `items()` hands back constrained values, recomputed through each parameter's
    transform; the optimiser holds the unconstrained ones. A poisoned gradient lands in
    those, and those are what a rollback has to restore.
    """
    return pyro.get_param_store()._params


def _reset_moments(opt):
    """Drop the per-parameter optimiser state that a non-finite gradient has poisoned.

    This is what makes the rollback worth doing. Adam carries exp_avg and exp_avg_sq
    across steps, so a single NaN gradient contaminates them permanently: restoring the
    parameter alone leaves the next step dividing by a NaN second moment, and the fit
    dies again one step later. Pyro rebuilds the optimiser for a parameter whenever its
    entry is missing, so deleting the entry restarts the moments from zero -- the cost is
    that this parameter loses its momentum, which is the point. No lr decay is configured
    here, so the rebuilt optimiser resumes at the same learning rate.
    """
    dropped = []
    for prm, o in list(opt.optim_objs.items()):
        st = o.state.get(prm, {})
        if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in st.values()):
            del opt.optim_objs[prm]
            dropped.append(tuple(prm.shape))
    return dropped


snapshot = None
skipped = 0
first_bad = None
# A running mean of the loss. The ELBO on a single relaxed sample swings across orders of
# magnitude and is useless as a stopping signal, which is why the plateau rule watches the
# estimand instead -- but averaged over a thousand steps it is the only thing that can say
# which of two stationary points is the better fit, and two initialisations of this guide
# can land on very different posterior WIDTHS. Without it a disagreement between
# them has no arbiter.
loss_sum, loss_n = 0.0, 0
for step in range(1, a.steps + 1):
    snapshot = {n: v.detach().clone() for n, v in _unconstrained().items()}
    loss = svi.step(Yb, obs_mask=mask)
    # The parameters, not the loss, are what has to be checked. A non-finite gradient
    # lands in a parameter silently -- this step's loss was computed before it -- and the
    # failure surfaces one step later as a distribution refusing its own argument, from
    # inside svi.step, where no return value can report it. At this cohort's
    # signal-to-noise (a typical transient about 2.5x the residual scale) the mark
    # posterior is weakly identified and this does happen: both arms died this way around
    # step 11000. ClippedAdam is no defence, since clipping the norm of a NaN leaves a
    # NaN. Checking costs a few small reductions; no parameter here exceeds
    # (sessions x windows).
    if math.isfinite(loss):
        loss_sum += float(loss); loss_n += 1
    bad = [n for n, v in _unconstrained().items() if not torch.isfinite(v).all()]
    if bad or not math.isfinite(loss):
        with torch.no_grad():
            for n, v in _unconstrained().items():
                if n in snapshot:
                    v.copy_(snapshot[n])
        dropped = _reset_moments(svi.optim)
        skipped += 1
        if first_bad is None:
            first_bad = dict(step=step, loss=float(loss), params=bad or ["(loss only)"])
            print(f"    pas no finit a {step}: loss {loss}, parametres {bad}, "
                  f"moments reiniciats {dropped}", flush=True)
            print("      magnituds: "
                  + ", ".join(f"{n} {v:.2f}" for v, n in _magnitudes()[:8]), flush=True)
            print(f"      marques: {_mark_reach()}", flush=True)
        # A handful of rolled-back steps out of sixty thousand is a numerical hiccup and
        # the fit is still the fit. A tenth of them is not, and reporting a contrast from
        # a run that spent its time rolling back would be reporting the guard's output
        # rather than the model's.
        if skipped > a.steps // 10:
            raise SystemExit(f"  avortat: {skipped} passos no finits de {step}; "
                             f"el fit no es recupera")
    if step % 1000 == 0:
        # nociceptin minus saline, with the interval the design earns: `drug_contrast`
        # carries the between-animal component, which cancels out of a within-animal
        # comparison and sets the floor when each animal has one condition.
        est, sd = model.drug_contrast(1, 0)
        traj.append(dict(step=step, contrast=est, sd=sd,
                         elbo=-(loss_sum / loss_n) if loss_n else None))
        loss_sum, loss_n = 0.0, 0
        mk = _mark_reach()
        extra = (f"   a_loc {mk['loc']:.2f} a_scale {mk['scale']:.2f} "
                 f"reach {mk['reach']:.1f}/{mk['limit']}"
                 + ("  OVERFLOWS" if mk["overflows"] else "")) if mk else ""
        print(f"    {step:>6}/{a.steps}  contrast {est:+.4f} +- {sd:.4f}{extra}",
              flush=True)
        if plateau.update(step, est):
            print(f"    plateau a {step}", flush=True)
            break

# Whether the mark posterior ended up against its bound, which nothing checked
# before. The bound stops the fit dying; it does not make a fit that reached it
# trustworthy, and the share of bins pinned there is how mismatched the mark prior was.
_sc = pyro.get_param_store()["q_a_scale"].detach()
pinned = float((_sc > 0.99 * _MARK_SCALE_MAX).float().mean())
if pinned > 0.001:
    print(f"  ATENCIO: {100*pinned:.1f}% de q_a_scale al seu limit ({_MARK_SCALE_MAX}); "
          f"el prior de les marques no encaixa amb aquestes dades", flush=True)
else:
    print(f"  q_a_scale max {float(_sc.max()):.2f} de {_MARK_SCALE_MAX} -- lluny del limit",
          flush=True)

# Whether the width was still moving when the fit stopped. The plateau rule settles the
# estimate; this says whether the interval had settled with it, so a width is never reported
# from a fit that was still inflating it.
# The drift is compared against the width's OWN probe-to-probe noise, not against a
# fraction of its level. A fixed 2% threshold was the first version and it is useless here:
# the width wanders by about 6% between probes, so a 2% test fires whether or not anything
# is moving, and a warning that always fires carries no information. Ten probes give enough
# to separate a trend from the wander.
_n = max(f["stop_window"], 10)
_sd = [t["sd"] for t in traj[-_n:]] if len(traj) >= _n else []
if _sd:
    _slope = float(np.polyfit(range(len(_sd)), _sd, 1)[0])      # per probe
    sd_drift = _slope * len(_sd)                                 # over the window
    _noise = float(np.std(np.diff(_sd)))
    sd_settled = abs(sd_drift) <= max(_noise, 0.02 * float(np.mean(_sd)))
else:
    sd_drift, sd_settled = float("nan"), False
if _sd and not sd_settled:
    print(f"  ATENCIO: la desviacio encara es movia ({sd_drift:+.4f} sobre la finestra, "
          f"soroll {_noise:.4f}); l'interval no s'havia assentat", flush=True)
elif _sd:
    print(f"  interval assentat: deriva {sd_drift:+.4f} dins del soroll {_noise:.4f}",
          flush=True)

# The LAST probe is one draw from the orbit the iterate settles into, and on this cohort the
# orbit is +-0.05 wide against an interval of +-0.15 -- a tenth of the reported width thrown
# away for nothing. `EstimandPlateau` says this in its own docstring and the benchmark
# already reports `plateau_mean`; this did not. The width comes from the final fit, since
# averaging scales would understate it.
contrast, contrast_sd = model.drug_contrast(1, 0)
if plateau.converged_at is not None and math.isfinite(plateau.plateau_mean):
    contrast = float(plateau.plateau_mean)
per_drug = model.drug_posterior()
out = dict(design=DESIGN, n_sessions=len(sessions), n_animals=len(animals), T=T,
           pre_s=PRE_S, fs=FS,
           animals=animals, sessions=[dict(animal=s.animal, cond=s.cond,
                                           iso=s.iso, mask_s=s.mask_s,
                                           mask_used_s=(s.mask_bins[1] - s.mask_bins[0]) / FS)
                                      for s in sessions],
           detrend_window=DETREND, mask_min_s=MASK_MIN_S,
           sigma_y=sig, mu_a=mu_a, sigma_a=sigma_a,
           sigma_y_sessions=list(getattr(model, "sigma_y_sessions", []) or []),
           mu_a_sessions=[float(v) for v in model.phasic.mu_a.detach().reshape(-1).cpu()],
           mark_gate_floor=float(cfg["fit"].get("mark_gate_floor", 0.0)),
           block_init_at_prior=bool(cfg["fit"].get("block_init_at_prior", False)),
           ar_coeffs=phi, kernel=kspec, kernel_length=klen,
           steps=step, trajectory=traj, skipped_steps=skipped, first_bad=first_bad,
           mark_scale_max=float(_sc.max()), mark_scale_pinned=pinned,
           sd_drift=sd_drift, sd_settled=bool(sd_settled),
           # The per-condition pharmacokinetics. The model has estimated these all
           # along -- tonic_hierarchy is on by default and the two conditions each
           # have five sessions to pool over -- and this script simply never wrote
           # them down, so the only record of the slow component the fit inferred was
           # inside a param store that is gone when the process exits.
           levels=list(model.levels),
           tonic_posterior={CONDS[d]: v for d, v in model.tonic_posterior().items()},
           contrast=float(contrast), contrast_sd=float(contrast_sd),
           contrast_last=float(model.drug_contrast(1, 0)[0]),
           mu_drug={int(d): [float(v) for v in lo_sd]
                    for d, lo_sd in per_drug.items()},
           block_spread=block_spread())
# The per-condition tonic WITH its uncertainty, on the unconstrained axis the guide uses
# (alpha = log peak, rho = log k_decay, gamma = log(k_rise - k_decay)): the contrast in
# alpha between conditions is the log ratio of the response heights. Added 2026-10-08 so
# the tonic difference between pretreatments can be stated with an interval.
from photosvi.models.points.hierarchical import _TONIC_COMPONENTS  # noqa: E402
if model.tonic_hierarchy:
    _mom = {n_: model._tonic_mu_moments(n_) for n_ in _TONIC_COMPONENTS}
    out["tonic_mu"] = {
        CONDS[d]: {n_: [float(_mom[n_][0][d]), float(_mom[n_][1][d, d].sqrt())]
                   for n_ in _TONIC_COMPONENTS}
        for d in range(model.n_drugs)}
    # Treated minus vehicle in log height, with the covariance the tonic block carries.
    out["tonic_contrast_alpha"] = list(model.tonic_contrast(1, 0, "alpha"))
    out["tonic_block"] = bool(model.tonic_block)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
from photosvi.data.bonaventura3 import ALS_LAMBDA_PER_FS  # noqa: E402
out["batches"] = list(BATCHES)
out["exclude"] = sorted(":".join(e) for e in EXCLUDE)
out["als_lambda_per_fs"] = float(_als) if _als else ALS_LAMBDA_PER_FS
out["sessions_meta"] = [dict(
    animal=s.animal, cond=s.cond, batch=s.batch, source=os.path.basename(s.source),
    fs_raw=round(s.fs_raw, 4), inj_source=s.inj_source,
    mask_s=round((s.mask_bins[1] - s.mask_bins[0]) / FS, 1),
    sigma_y=math.exp(estimate_log_noise_scale(
        torch.as_tensor(s.y[:int(PRE_S * FS)], dtype=torch.float32).reshape(1, -1),
        ar_coeffs=phi))) for s in sessions]
json.dump(out, open(a.out, "w"), indent=1)

# Arrays beside the JSON: the per-bin event probabilities (for event rates over time),
# each session's fitted tonic, and each condition's pooled tonic without the session
# offsets (for the group-mean panel).
with torch.no_grad():
    q_ev = model.event_posterior(Yb).float().cpu().numpy()
    tn = model.tonic
    S_, T_ = Yb.shape
    if model.tonic_hierarchy:
        mu = {n_: model._tonic_mu_moments(n_)[0].to(Yb.device) for n_ in _TONIC_COMPONENTS}
        ses = {}
        for n_ in _TONIC_COMPONENTS:
            v = mu[n_][model.drug]
            if model.tonic_session_level:
                v = v + (pyro.param(f"q_tonic_sigma_{n_}_loc").detach().exp()
                         * model._tonic_z_loc(n_))
            ses[n_] = v
        x_ses = tn.baseline_from(tn.b, ses["alpha"], ses["rho"], ses["gamma"], tn.drift,
                                 T_, model.inj_t)
        # The tonic block is sized per session, so the pooled curve is evaluated on every
        # session with its condition's mean and no offsets, then one row kept per condition.
        zero = torch.zeros(S_, device=Yb.device, dtype=Yb.dtype)
        x_all = tn.baseline_from(zero, mu["alpha"][model.drug], mu["rho"][model.drug],
                                 mu["gamma"][model.drug], zero, T_, model.inj_t)
        x_grp = torch.stack([x_all[(model.drug == d).nonzero()[0, 0]]
                             for d in range(model.n_drugs)])
    else:
        x_ses = tn.baseline(T_, model.inj_t)
        x_grp = torch.zeros(1, T_)
np.savez_compressed(os.path.splitext(a.out)[0] + "_arrays.npz",
                    q=q_ev.astype(np.float16), y=Yb.float().cpu().numpy(),
                    x_session=x_ses.float().cpu().numpy(), x_group=x_grp.float().cpu().numpy(),
                    drug=model.drug.cpu().numpy(), inj_t=model.inj_t.cpu().numpy(),
                    mask=mask.cpu().numpy(), group_names=np.array([CONDS[d] for d in range(model.n_drugs)]))
print(f"  escrit {a.out}")
