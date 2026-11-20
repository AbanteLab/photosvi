"""Fitting a cohort with the hierarchical model: `fit_cohort` and what it returns.

The procedure is the one behind the paper's real-cohort analysis
(`scripts/cohort_bonaventura3.py`), with what was specific to that laboratory's files
taken out: the sessions arrive as `Session` objects, the animal and the condition are read
from their metadata, and the noise and mark scales are measured on the recordings passed
in. On the paper's 13 recordings the two give the same fit, step for step.

Exported through `photosvi.api`, which is where callers should import it from.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

Z90 = 1.6448536269514722
#: The cohort configuration of the paper: priors, guide, learning rate, stopping rule.
COHORT_CONFIG = "final_cohort_48.yaml"
#: The levels the paper fits. Of those the design identifies, only these are switched on;
#: the session-level baseline (`b0_session`) is left out, as in the paper's real cohort
#: and power benchmark.
DEFAULT_LEVELS = ("animal", "cell")


@dataclass
class CohortSettings:
    """What a caller may change in a cohort fit. `None` means the shipped value.

    The defaults are the paper's real-cohort fit: at least 50,000 and at most 100,000 steps,
    the contrast probed every 1,000 steps and the fit stopped once it no longer drifts, a
    window of 10 minutes before the injection, the levels of `DEFAULT_LEVELS` that the
    design identifies, and the kernel of the single-trace model.
    """

    min_steps: int = 50_000
    max_steps: int = 100_000
    probe_every: int = 1_000
    device: str = "auto"
    seed: int = 0
    #: Seconds of recording kept before the injection; None keeps all that every session
    #: has. The window should not reach back to an earlier injection.
    pre_s: float | None = 600.0
    #: Random-effect levels; None -> the levels of DEFAULT_LEVELS the design identifies.
    levels: tuple[str, ...] | None = None
    #: AR coefficients of the noise, shared by the cohort. None -> the mean over sessions of
    #: each one's Yule-Walker AR(2) estimate on its pre-injection segment.
    ar_coeffs: list[float] | None = None
    #: The kernel as a config `kernel:` block ({"fit": {...}, "length": taps at
    #: `kernel_fs`}). None -> the single-trace model's, from `api.PRODUCTION_CONFIG`.
    kernel: dict[str, Any] | None = None
    kernel_fs: float | None = None
    #: Floor on the gated mark divergence; None -> the cohort config's (0.1).
    mark_gate_floor: float | None = None
    config: str = COHORT_CONFIG

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["levels"] = list(self.levels) if self.levels is not None else None
        return d


@dataclass
class CohortResult:
    """What a cohort fit yields.

    `contrast(a, b)` is the estimand: the difference between conditions `a` and `b` in the
    log rate of transients after the injection, with a 90% interval that carries the
    between-animal variability. `tonic` holds each condition's pooled tonic response.
    """

    conditions: list[str]
    reference: str
    animals: list[str]
    sessions: list[dict[str, Any]]
    settings: CohortSettings
    design: dict[str, Any]
    #: (a, b) -> (estimate, sd) on the log-rate axis, for every ordered pair.
    contrasts_: dict[tuple[str, str], tuple[float, float]]
    #: condition -> {"peak", "k_decay", "k_rise", "peak_s"} of the pooled tonic curve.
    tonic: dict[str, dict[str, float]]
    #: (a, b) -> (estimate, sd) of the log ratio of the tonic peak heights.
    tonic_contrasts_: dict[tuple[str, str], tuple[float, float]]
    variance_components: dict[str, float]
    sigma_y: float
    sigma_y_sessions: list[float]
    ar_coeffs: list[float]
    kernel: dict[str, Any]
    kernel_length: int
    fs: float
    inj_t: list[int]
    steps: int
    converged_at: int | None
    sd_settled: bool
    skipped_steps: int
    seconds: float
    device: str
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    #: [S, T] arrays: the fitted traces, the per-bin event probability, each session's
    #: tonic curve and the mask; [D, T]: each condition's pooled tonic curve.
    y: np.ndarray | None = None
    event_prob: np.ndarray | None = None
    tonic_session: np.ndarray | None = None
    tonic_condition: np.ndarray | None = None
    mask: np.ndarray | None = None

    def contrast(self, a: str, b: str | None = None) -> dict[str, float]:
        """Condition `a` minus `b` (default: the reference), with its 90% interval.

        `estimate`, `lo` and `hi` are on the log-rate axis; `ratio`, `ratio_lo` and
        `ratio_hi` are the same as the factor by which the transient rate after the
        injection differs between the two conditions.
        """
        b = self.reference if b is None else b
        est, sd = self.contrasts_[(a, b)]
        return _interval(est, sd)

    def tonic_ratio(self, a: str, b: str | None = None) -> dict[str, float]:
        """Ratio of the tonic peak heights, `a` over `b` (default: the reference), 90%."""
        b = self.reference if b is None else b
        est, sd = self.tonic_contrasts_[(a, b)]
        out = _interval(est, sd)
        return dict(ratio=out["ratio"], lo=out["ratio_lo"], hi=out["ratio_hi"],
                    level=out["level"])

    def to_dict(self) -> dict[str, Any]:
        others = [c for c in self.conditions if c != self.reference]
        return dict(
            conditions=self.conditions, reference=self.reference, animals=self.animals,
            sessions=self.sessions, settings=self.settings.to_dict(), design=self.design,
            contrasts={f"{a} - {b}": self.contrast(a, b)
                       for (a, b) in self.contrasts_ if a != b},
            main=({f"{c} - {self.reference}": self.contrast(c) for c in others}),
            tonic=self.tonic,
            tonic_ratios={f"{a} / {b}": self.tonic_ratio(a, b)
                          for (a, b) in self.tonic_contrasts_ if a != b},
            variance_components=self.variance_components,
            sigma_y=self.sigma_y, sigma_y_sessions=self.sigma_y_sessions,
            ar_coeffs=self.ar_coeffs, kernel=self.kernel, kernel_length=self.kernel_length,
            fs=self.fs, inj_t=self.inj_t, steps=self.steps, converged_at=self.converged_at,
            sd_settled=self.sd_settled, skipped_steps=self.skipped_steps,
            seconds=round(self.seconds, 1), device=self.device, trajectory=self.trajectory,
        )

    def save(self, out: Path | str) -> Path:
        """Write a run directory: result.json and curves.npz (the [S, T] arrays)."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(self.to_dict(), indent=1))
        arrays = {k: getattr(self, k) for k in
                  ("y", "event_prob", "tonic_session", "tonic_condition", "mask")
                  if getattr(self, k) is not None}
        np.savez_compressed(out / "curves.npz", conditions=np.array(self.conditions),
                            session_names=np.array([s["name"] for s in self.sessions]),
                            **arrays)
        return out


def _interval(est: float, sd: float) -> dict[str, float]:
    lo, hi = est - Z90 * sd, est + Z90 * sd
    return dict(estimate=est, sd=sd, lo=lo, hi=hi, ratio=math.exp(est),
                ratio_lo=math.exp(lo), ratio_hi=math.exp(hi), level=0.90)


def _label(meta: dict[str, Any], keys: Sequence[str], what: str, name: str) -> str:
    for k in keys:
        if k in meta and meta[k] not in (None, ""):
            return str(meta[k])
    raise ValueError(f"session {name!r} has no {what} in its metadata "
                     f"(expected one of {list(keys)})")


def _peak_marks(pre_segments: Sequence[np.ndarray], fs: float) -> tuple[float, float]:
    """(mu_a, sigma_a) from the prominences of the isolated pre-injection peaks.

    Only used when the noise-unit mark mode is off. The kernel is peak-normalised, so the
    median prominence estimates exp(mu_a); the spread is that of the visible peaks.
    """
    from scipy.signal import butter, detrend, filtfilt, find_peaks
    peaks, spreads = [], []
    for pre in pre_segments:
        z = filtfilt(*butter(3, [0.1 / (fs / 2), 10.0 / (fs / 2)], "band"),
                     detrend(pre, type="linear"))
        sd = np.median(np.abs(z)) / 0.6745
        _, props = find_peaks(z, prominence=3 * sd, width=max(1, int(0.04 * fs)))
        peaks.append(float(np.median(props["prominences"])))
        spreads.append(float(np.std(np.log(props["prominences"]), ddof=1)))
    return float(np.log(np.median(peaks))), float(np.median(spreads))


def prepare_cohort(sessions: Sequence, settings: CohortSettings, *,
                   reference: str | None = None,
                   condition_keys: Sequence[str] = ("condition", "drug"),
                   animal_keys: Sequence[str] = ("animal",)) -> dict[str, Any]:
    """Everything `fit_cohort` builds before the first step, without fitting.

    Returns the cohort config, the model spec, the index vectors, the traces cut to one
    window and the mask. Separate so that the construction can be checked on its own.
    """
    import yaml

    from photosvi.api import PRODUCTION_CONFIG, _shipped, config_path
    from photosvi.experiments.cohort import spec_from_config
    from photosvi.experiments.marks import noise_unit_marks, noise_unit_mode
    from photosvi.models.points import estimate_ar_coeffs, estimate_log_noise_scale

    if len(sessions) < 2:
        raise ValueError("a cohort needs at least two sessions")
    fs = float(sessions[0].fs)
    if any(abs(float(s.fs) - fs) > 1e-6 for s in sessions):
        raise ValueError("every session must have the same sampling rate; got "
                         f"{sorted({float(s.fs) for s in sessions})} Hz. Resample first.")
    names = [s.name for s in sessions]
    conds = [_label(s.metadata, condition_keys, "condition", s.name) for s in sessions]
    anim = [_label(s.metadata, animal_keys, "animal", s.name) for s in sessions]
    levels_seen = sorted(set(conds))
    if len(levels_seen) < 2:
        raise ValueError(f"a cohort needs at least two conditions; got {levels_seen}")
    if reference is None:
        reference = levels_seen[0]
    if reference not in levels_seen:
        raise ValueError(f"reference {reference!r} is not a condition of this cohort "
                         f"({levels_seen})")
    conditions = [reference] + [c for c in levels_seen if c != reference]
    animals = sorted(set(anim))

    # One window for every session: the model takes [S, T]. Each trace is cut from `pre`
    # bins before its injection to `post` bins after it, the most that all sessions have.
    pre_avail = min(int(s.inj_t) for s in sessions)
    pre = pre_avail if settings.pre_s is None else min(pre_avail,
                                                      int(round(settings.pre_s * fs)))
    post = min(s.n_samples - int(s.inj_t) for s in sessions)
    T = pre + post
    Y = torch.stack([torch.as_tensor(s.trace[int(s.inj_t) - pre:int(s.inj_t) + post],
                                     dtype=torch.float32) for s in sessions])
    mask = torch.ones(len(sessions), T, dtype=torch.bool)
    for i, s in enumerate(sessions):
        if s.mask is not None:
            mask[i] = torch.as_tensor(s.mask[int(s.inj_t) - pre:int(s.inj_t) + post])
    pre_seg = [s.trace[int(s.inj_t) - pre:int(s.inj_t)] for s in sessions]

    cfg = yaml.safe_load(config_path(settings.config).read_text())
    if settings.mark_gate_floor is not None:
        cfg["fit"]["mark_gate_floor"] = float(settings.mark_gate_floor)
    cfg["cohort"]["sigma_y"] = 0.0          # measured on the recordings below
    spec = spec_from_config(cfg, 0.20)      # the effect size only matters to simulation
    if settings.kernel is not None:
        kblock, kfs = settings.kernel, settings.kernel_fs or fs
    else:
        prod = _shipped(PRODUCTION_CONFIG)
        kblock, kfs = prod["kernel"], float(prod["data"]["fs"])
    kspec = dict(kblock["fit"])
    klen = int(round(kblock["length"] / kfs * fs))
    if settings.ar_coeffs is not None:
        phi = [float(x) for x in settings.ar_coeffs]
    else:
        est = [estimate_ar_coeffs(torch.as_tensor(p, dtype=torch.float32)) for p in pre_seg]
        est = [e for e in est if len(e) == 2]
        phi = [float(x) for x in np.mean(est, axis=0)] if est else []
    sig = float(np.exp(np.mean([estimate_log_noise_scale(
        torch.as_tensor(p, dtype=torch.float32).reshape(1, -1), ar_coeffs=phi)
        for p in pre_seg])))
    nu = noise_unit_mode(cfg["fit"])
    mu_a, sigma_a = noise_unit_marks(sig, *nu) if nu is not None else _peak_marks(pre_seg, fs)
    spec = dataclasses.replace(spec, fs=int(round(fs)), kernel=kspec, kernel_length=klen,
                               ar_coeffs=phi, sigma_y=sig, mu_a=mu_a, sigma_a=sigma_a)
    coh = dict(y=Y, inj_t=[pre] * len(sessions),
               animal=[animals.index(x) for x in anim],
               drug=[conditions.index(c) for c in conds])
    return dict(cfg=cfg, spec=spec, coh=coh, mask=mask, fs=fs, T=T, pre=pre,
                conditions=conditions, reference=reference, animals=animals,
                session_info=[dict(name=n, animal=a, condition=c, inj_t=int(s.inj_t),
                                   masked_bins=int((~mask[i]).sum()))
                              for i, (n, a, c, s) in enumerate(zip(names, anim, conds,
                                                                   sessions, strict=True))],
                sigma_y=sig, ar_coeffs=phi, kernel=kspec, kernel_length=klen)


def fit_cohort(study, settings: CohortSettings | None = None, *,
               reference: str | None = None,
               condition_keys: Sequence[str] = ("condition", "drug"),
               animal_keys: Sequence[str] = ("animal",),
               on_progress: Callable[[dict[str, Any]], None] | None = None) -> CohortResult:
    """Fit a cohort jointly with the hierarchical model.

    `study` is a `Study`, or a list of `Session`, one per recording; each session's
    metadata names its animal (`animal`) and its condition (`condition`, or `drug`). The
    contrasts are differences between conditions in the log rate of transients after the
    injection; `reference` is the condition the others are compared against (by default
    the first in alphabetical order), usually the vehicle.

    All sessions must have the same sampling rate. Each is cut to the same window around
    its injection: `settings.pre_s` before it, and as much after it as every session has.
    Bins where `Session.mask` is False (the handling window) are left out of the
    likelihood.

    `on_progress` is called at every probe with the step, the maximum number of steps and
    the current contrasts against the reference. A cohort of a dozen 40-minute recordings
    takes several hours on a GPU.
    """
    import time

    import pyro
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import ClippedAdam

    from photosvi.experiments.cohort import build_cohort_model
    from photosvi.inference.points import EstimandPlateau, prepare, resolve_device
    from photosvi.models.points.hierarchical import HierarchicalModel

    st = settings or CohortSettings()
    sessions = list(getattr(study, "sessions", study))
    c = prepare_cohort(sessions, st, reference=reference, condition_keys=condition_keys,
                       animal_keys=animal_keys)
    cfg, f = c["cfg"], c["cfg"]["fit"]
    conditions, ref = c["conditions"], c["reference"]
    D = len(conditions)

    levels = st.levels
    if levels is None:
        A = torch.as_tensor(c["coh"]["animal"])
        d = torch.as_tensor(c["coh"]["drug"])
        cell, n_cells = HierarchicalModel._cell_index(A, d)
        detected = HierarchicalModel._detect_levels(A, d, cell, n_cells, len(c["animals"]))
        levels = tuple(x for x in DEFAULT_LEVELS if x in detected)

    t0 = time.time()
    model, Yb, _ = build_cohort_model(cfg, c["spec"], c["coh"], levels=levels)
    device = resolve_device(st.device, batch=len(sessions))
    model, Yb = prepare(model, Yb, device)
    mask = c["mask"].to(Yb.device)

    pyro.clear_param_store()
    pyro.set_rng_seed(st.seed)
    svi = SVI(model.model, model.guide, ClippedAdam({"lr": f["lr"]}),
              Trace_ELBO(max_plate_nesting=2))
    plateau = EstimandPlateau(tol=f["stop_tol"], window=f["stop_window"],
                              min_steps=min(st.min_steps, st.max_steps // 2))
    pairs = [(a, b) for a in range(D) for b in range(D) if a != b]
    traj: list[dict[str, Any]] = []
    params = pyro.get_param_store()._params
    skipped = 0
    step = 0
    for step in range(1, st.max_steps + 1):
        snapshot = {n: v.detach().clone() for n, v in params.items()}
        loss = svi.step(Yb, obs_mask=mask)
        # A non-finite gradient lands in a parameter silently and surfaces a step later.
        # Roll back and restart the poisoned optimiser moments; give up if it recurs.
        if not math.isfinite(loss) or any(not torch.isfinite(v).all()
                                          for v in params.values()):
            with torch.no_grad():
                for n, v in params.items():
                    if n in snapshot:
                        v.copy_(snapshot[n])
            for prm, o in list(svi.optim.optim_objs.items()):
                if any(torch.is_tensor(x) and not torch.isfinite(x).all()
                       for x in o.state.get(prm, {}).values()):
                    del svi.optim.optim_objs[prm]
            skipped += 1
            if skipped > st.max_steps // 10:
                raise RuntimeError(f"{skipped} non-finite steps out of {step}; the fit "
                                   "does not recover")
        if step % st.probe_every == 0:
            row = {"step": step}
            for a, b in pairs:
                row[f"{a}-{b}"] = list(model.drug_contrast(a, b))
            traj.append(row)
            vs_ref = [row[f"{a}-0"][0] for a in range(1, D)]
            if on_progress is not None:
                on_progress(dict(step=step, total=st.max_steps,
                                 contrasts={conditions[a]: row[f"{a}-0"]
                                            for a in range(1, D)}))
            if plateau.update(step, vs_ref if D > 2 else vs_ref[0]):
                break

    # The estimate is the mean over the plateau window (the iterate orbits its answer),
    # or the last probe if the fit never settled; the width is the mean over the same
    # number of probes, since it oscillates too. This is how the paper reports it.
    n = f["stop_window"]
    contrasts = {}
    for a, b in pairs:
        rows = [r[f"{a}-{b}"] for r in traj[-n:]] or [list(model.drug_contrast(a, b))]
        est = (float(np.mean([r[0] for r in rows])) if plateau.converged_at is not None
               else float(rows[-1][0]))
        contrasts[(conditions[a], conditions[b])] = (est, float(np.mean([r[1] for r in rows])))

    # Whether the width was still moving when the fit stopped, against its own noise.
    sds = [r["1-0"][1] for r in traj[-max(n, 10):]]
    sd_settled = False
    if len(sds) >= max(n, 10):
        drift = float(np.polyfit(range(len(sds)), sds, 1)[0]) * len(sds)
        sd_settled = abs(drift) <= max(float(np.std(np.diff(sds))), 0.02 * float(np.mean(sds)))

    tonic, tonic_contrasts = {}, {}
    if model.tonic_hierarchy:
        tonic = {conditions[d]: v for d, v in model.tonic_posterior().items()}
        tonic_contrasts = {(conditions[a], conditions[b]): model.tonic_contrast(a, b, "alpha")
                           for a, b in pairs}
    arrays = _arrays(model, Yb, mask, D)
    return CohortResult(
        conditions=conditions, reference=ref, animals=c["animals"],
        sessions=c["session_info"], settings=st,
        design={k: (list(v) if isinstance(v, tuple) else v)
                for k, v in model.design_summary().items()},
        contrasts_=contrasts, tonic=tonic, tonic_contrasts_=tonic_contrasts,
        variance_components=model.variance_components(),
        sigma_y=c["sigma_y"],
        sigma_y_sessions=[float(x) for x in (getattr(model, "sigma_y_sessions", []) or [])],
        ar_coeffs=c["ar_coeffs"], kernel=c["kernel"], kernel_length=c["kernel_length"],
        fs=c["fs"], inj_t=[c["pre"]] * len(sessions), steps=step,
        converged_at=plateau.converged_at, sd_settled=bool(sd_settled),
        skipped_steps=skipped, seconds=time.time() - t0, device=str(device),
        trajectory=[{"step": r["step"], **{f"{conditions[a]} - {conditions[b]}": r[f"{a}-{b}"]
                                           for a, b in pairs}} for r in traj],
        **arrays)


@torch.no_grad()
def _arrays(model, Yb: torch.Tensor, mask: torch.Tensor, D: int) -> dict[str, np.ndarray]:
    """The per-bin event probability, each session's tonic and each condition's pooled one."""
    import pyro

    from photosvi.models.points.hierarchical import _TONIC_COMPONENTS
    q = model.event_posterior(Yb).float().cpu().numpy()
    tn = model.tonic
    S, T = Yb.shape
    if model.tonic_hierarchy:
        mu = {n: model._tonic_mu_moments(n)[0].to(Yb.device) for n in _TONIC_COMPONENTS}
        ses = {}
        for n in _TONIC_COMPONENTS:
            v = mu[n][model.drug]
            if model.tonic_session_level:
                v = v + (pyro.param(f"q_tonic_sigma_{n}_loc").detach().exp()
                         * model._tonic_z_loc(n))
            ses[n] = v
        x_ses = tn.baseline_from(tn.b, ses["alpha"], ses["rho"], ses["gamma"], tn.drift,
                                 T, model.inj_t)
        zero = torch.zeros(S, device=Yb.device, dtype=Yb.dtype)
        x_all = tn.baseline_from(zero, mu["alpha"][model.drug], mu["rho"][model.drug],
                                 mu["gamma"][model.drug], zero, T, model.inj_t)
        x_grp = torch.stack([x_all[(model.drug == d).nonzero()[0, 0]] for d in range(D)])
    else:
        x_ses = tn.baseline(T, model.inj_t)
        x_grp = torch.zeros(D, T)
    return dict(y=Yb.float().cpu().numpy(), event_prob=q.astype(np.float16),
                tonic_session=x_ses.float().cpu().numpy(),
                tonic_condition=x_grp.float().cpu().numpy(), mask=mask.cpu().numpy())
