"""The stable surface: what a caller outside this package should use.

    from photosvi.api import Session, FitSettings, fit_session

    session = Session.from_array(trace, fs=25.0, inj_t=15000, name="m3_fentanyl")
    result = fit_session(session, FitSettings(), on_progress=print)
    result.rate_ratio           # e^beta_inj, with a credible interval
    result.events()             # times and amplitudes
    result.save(Path("run"))    # the same layout the CLI writes

Three consumers are meant to go through here and nowhere else: the command line, the
notebook templates, and the GUI in `photosvi-app`. Everything under `photosvi.models`
and `photosvi.inference` is internal and changes without notice; this module is the
part that is allowed to be depended on.

The unit is a `Session` -- one recording with an injection -- and a `Study` is a
collection of them. Even code that only ever handles one recording should say
`Study([session])` rather than inventing a "the recording" concept, because the
cohort model reads its design off the per-session metadata and that metadata has to
exist before there is a cohort to fit.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "FitSettings",
    "KERNEL_SENSORS",
    "KernelCalibration",
    "Result",
    "Session",
    "Study",
    "calibrate_kernel",
    "config_path",
    "events_from_posterior",
    "fit_session",
    "load_doric_session",
    "load_session",
    "spectrum",
]



@dataclass
class Session:
    """One recording: a trace, when the injection happened, and who it came from.

    `trace` is corrected dF/F, one dimension. Preprocessing -- isosbestic correction,
    bleaching, dF/F -- happens upstream of this package.

    `metadata` carries the experimental design: animal, drug, dose, and whatever else
    the study records. It is free-form here and becomes the index vectors the
    hierarchical model reads, so a pilot that only fits single recordings should still
    fill it in -- it costs nothing now and is the difference between having a cohort
    later and having a pile of unlabelled traces.
    """

    trace: np.ndarray
    fs: float
    inj_t: int
    name: str = "session"
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Bins that count towards the likelihood, or None for all of them. The reason this
    #: exists is the injection: the animal is handled, the fibre moves, and the trace
    #: carries an artefact for 15 to 80 seconds depending on the session. It cannot be
    #: left in -- the model's only way to explain a fast excursion is events, and this one
    #: sits exactly where the rate contrast is read -- and it cannot be trimmed either,
    #: since cutting it would splice two stretches together. Masking leaves the events
    #: under it at their prior, which is the honest statement that we do not know.
    #: Per SESSION rather than per study, because the window is not the same length twice.
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.trace = np.asarray(self.trace, dtype=np.float32).reshape(-1)
        if self.trace.size == 0:
            raise ValueError("trace is empty")
        if not np.isfinite(self.trace).all():
            raise ValueError("trace contains non-finite samples; clean or crop it first")
        if self.fs <= 0:
            raise ValueError(f"fs must be positive, got {self.fs}")
        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool).reshape(-1)
            if self.mask.size != self.trace.size:
                raise ValueError(
                    f"mask has {self.mask.size} bins and the trace has "
                    f"{self.trace.size}; they index the same time axis")
            if not self.mask.any():
                raise ValueError("mask excludes every bin, so there is nothing to fit")
        if not 0 <= self.inj_t < self.trace.size:
            raise ValueError(
                f"inj_t {self.inj_t} is outside the recording (0..{self.trace.size - 1})"
            )

    @classmethod
    def from_array(cls, trace, fs: float, inj_t: int, **kw) -> Session:
        return cls(trace=trace, fs=fs, inj_t=inj_t, **kw)

    @property
    def n_samples(self) -> int:
        return int(self.trace.size)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    @property
    def inj_s(self) -> float:
        return self.inj_t / self.fs

    def summary(self) -> dict[str, Any]:
        return dict(name=self.name, n_samples=self.n_samples, fs=self.fs,
                    inj_t=self.inj_t, inj_s=round(self.inj_s, 2),
                    duration_s=round(self.duration_s, 2), metadata=dict(self.metadata))


@dataclass
class Study:
    """A set of sessions fitted together. One session is a legitimate study."""

    sessions: list[Session]

    def __post_init__(self) -> None:
        if not self.sessions:
            raise ValueError("a study needs at least one session")
        names = [s.name for s in self.sessions]
        if len(set(names)) != len(names):
            raise ValueError("session names must be unique within a study")

    def __len__(self) -> int:
        return len(self.sessions)

    def summary(self) -> dict[str, Any]:
        return dict(n_sessions=len(self.sessions),
                    sessions=[s.summary() for s in self.sessions])


# The paper's single-trace configuration (Table 1 and the single-trace arm of the cohort
# benchmark): dLight1.3b kernel K10, AR(2) emission, tau 0.2, window 10, at 48 Hz. Until
# 2026-10-15 this was the 25 Hz benchmark_guppy.yaml, whose K5 kernel is far narrower than
# dLight1.3b. Every duration in it is in bins at its own `data.fs` and is rescaled to
# the recording's rate.
PRODUCTION_CONFIG = "final_detection_48.yaml"
# Since 2026-10-21 the defaults are the paper's FINAL model (configs/final_detection_48.yaml,
# `fit.architecture: paper_final`): marks fixed in noise units, one gate floor, beta0 ~
# N(anchor, 2^2), beta_inj ~ N(0, 1), tonic refined jointly. The settings before it are
# reachable with FitSettings(mark_noise_units=[]) (mark prior on the visible peaks) and the
# corresponding explicit values.
# The real-data settings the paper adds on top, all from the cohort configuration: the
# mark prior anchored on the recording's own peaks (sd 0.05 on mu_a and log sigma_a) and
# the floor on the gated mark divergence.
REAL_DATA_CONFIG = "cohort_power_48.yaml"


def _shipped(name: str = PRODUCTION_CONFIG) -> dict:
    """The config behind the paper's benchmark, as the source of every default.

    Retyping these values here is exactly how this module first went wrong: it
    reconstructed the model by hand with `warm.gain` at 1.0 instead of 2.0 and
    `window` at 8 instead of 5, which over-detected and compressed the rate ratio
    from 2.0 to 1.26 on data drawn from the model's own generative process. The
    numbers now have one home.
    """
    import yaml
    return yaml.safe_load(config_path(name).read_text())


def config_path(name: str) -> Path:
    """Path of a config shipped with the package, e.g. `config_path("toy.yaml")`.

    The configs live inside the package (`photosvi/configs/`), so they are there after a
    `pip install` as well as in a clone, where `configs/` at the root links to them.
    """
    from importlib.resources import files
    cand = Path(str(files("photosvi") / "configs" / name))
    if not cand.exists():
        raise FileNotFoundError(f"photosvi ships no config named {name!r}")
    return cand


def _noise_units_default(value, fit_cfg: dict):
    """The noise-unit mark parameters a fit uses: explicit, the config's, or off ([])."""
    if value is not None:
        return [float(v) for v in value] or None
    if fit_cfg.get("mark_mode") == "noise_units":
        p = fit_cfg.get("mark_noise_units") or {}
        from photosvi.experiments.marks import NOISE_UNITS_MU, NOISE_UNITS_SIGMA
        return [float(p.get("mu", NOISE_UNITS_MU)), float(p.get("sigma", NOISE_UNITS_SIGMA))]
    return None


@dataclass
class FitSettings:
    """Everything a caller may reasonably want to change, with defended defaults.

    `None` means "use the shipped value" and is the default for every model
    hyperparameter, so a caller that changes nothing gets the paper's single-trace model:
    the configuration of `PRODUCTION_CONFIG`, with the noise scale, the AR coefficients
    and the mark distribution read off the recording as the paper reads them, and the
    mark prior and gate floor of `REAL_DATA_CONFIG`. Only the operational fields -- how
    long to run, how many threads, which device -- have literals here, because those are
    properties of the machine rather than of the model.
    """

    # operational
    steps: int | None = None          # None -> the config's stop_min_steps
    max_steps: int | None = None
    threads: int = 4
    device: str = "auto"
    seed: int = 0
    event_threshold: float | None = None
    refractory: int | None = None
    # model hyperparameters; None -> shipped
    lr: float | None = None
    tau: float | None = None
    window: int | None = None
    sigma_y: float | None = None      # None -> estimated from the trace, AR-corrected
    # AR coefficients of the noise. None -> Yule-Walker AR(2) on the pre-injection trace,
    # the procedure that produced the paper's coefficients; [] fits white noise.
    ar_coeffs: list[float] | None = None
    # Centre of the mark prior. None -> read off the pre-injection peak prominences
    # (`experiments.marks`), as the paper's single-trace and real-data fits do. Taking it
    # from anywhere else while sigma_y comes from the data put the expected transient at
    # 450 times the noise on the opioid recordings, and the fit diverged.
    mu_a: float | None = None
    sigma_a: float | None = None
    # Width of the mark prior on mu_a and on log sigma_a. None -> REAL_DATA_CONFIG (0.05).
    mark_prior_sd: float | None = None
    # [mu_r, sigma_r]: the marks are fixed at a_t = sigma_y r_t, r_t ~ LogNormal(mu_r,
    # sigma_r^2), nothing about them fitted and mu_a/sigma_a/mark_prior_sd ignored. None ->
    # the config's (the paper's final model, [ln 1.95, 0.75]); [] -> the earlier mark prior
    # anchored on the visible peaks.
    mark_noise_units: list[float] | None = None
    # Floor on the gated mark divergence. None -> REAL_DATA_CONFIG (0.1). Gating the mark
    # divergence by q(s_t) leaves q(a)'s width unconstrained wherever the gate is near zero;
    # on the opioid recordings it reached 30 and overflowed single precision.
    mark_gate_floor: float | None = None
    kernel_length: int | None = None
    kernel: dict[str, Any] | None = None

    def resolved(self, fs: float | None = None) -> dict[str, Any]:
        """Every value this fit will actually use, with the shipped defaults filled in.

        Durations the config gives in bins (kernel support, refractory window) are
        rescaled from its sampling rate to `fs`; without `fs` they are quoted at the
        config's own rate. Values read off the recording (sigma_y, ar_coeffs, mu_a,
        sigma_a) stay None here and are filled in by `fit_session`.
        """
        c, rd = _shipped(), _shipped(REAL_DATA_CONFIG)["fit"]
        f, w, d, k = (c["fit"], c["warm"], c["detection"], c["kernel"])
        scale = (fs or c["data"]["fs"]) / c["data"]["fs"]
        mp = rd["mark_prior"]
        if mp["mu_sd"] != mp["log_sigma_sd"]:
            raise ValueError(f"{REAL_DATA_CONFIG}: one mark_prior_sd cannot carry {mp}")
        return dict(
            steps=self.steps if self.steps is not None else f["stop_min_steps"],
            max_steps=self.max_steps if self.max_steps is not None else f["max_steps"],
            threads=self.threads, device=self.device, seed=self.seed,
            event_threshold=(self.event_threshold if self.event_threshold is not None
                             else d["thresh"]),
            refractory=(self.refractory if self.refractory is not None
                        else max(1, int(round(d["refractory"] * scale)))),
            mf_thresh=d["mf_thresh"],
            lr=self.lr if self.lr is not None else f["lr"],
            tau=self.tau if self.tau is not None else f["tau"],
            window=self.window if self.window is not None else int(
                f.get("window", c.get("guide", {}).get("window", 10))),
            sigma_y=self.sigma_y,
            ar_coeffs=None if self.ar_coeffs is None else [float(v) for v in self.ar_coeffs],
            mu_a=self.mu_a, sigma_a=self.sigma_a,
            mark_prior_sd=(self.mark_prior_sd if self.mark_prior_sd is not None
                           else float(mp["mu_sd"])),
            mark_noise_units=_noise_units_default(self.mark_noise_units, f),
            mark_gate_floor=(self.mark_gate_floor if self.mark_gate_floor is not None
                             else float(f.get("mark_gate_floor", rd["mark_gate_floor"]))),
            architecture=f.get("architecture", "legacy"),
            kernel_length=(self.kernel_length if self.kernel_length is not None
                           else int(round(k["length"] * scale))),
            kernel=dict(self.kernel) if self.kernel else dict(k["fit"]),
            warm=dict(w),
            beta0_prior_scale=f["beta0_prior_scale"],
            beta_inj_prior_loc=f["beta_inj_prior_loc"],
            beta_inj_prior_scale=f["beta_inj_prior_scale"],
            beta_prior=f["beta_prior"], beta_guide=f["beta_guide"],
            stop_tol=f["stop_tol"], stop_window=f["stop_window"],
            # The shipped probe interval, untouched, unless the caller asked for a
            # shorter run than it can resolve. The plateau is only checked on a probe,
            # so a probe larger than the minimum makes the minimum meaningless --
            # asking for 30 steps and getting 60000 because the first check lands at
            # 3000. Narrowing it by default would change the validated path, so the
            # default is the config value exactly.
            stop_probe=(f["stop_probe"] if self.steps is None
                        else min(f["stop_probe"],
                                 max(1, self.steps // f["stop_window"]))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FitSettings:
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        return cls(**d)


@dataclass
class Result:
    """What one fitted session yields, and the only thing callers should read.

    `rate_ratio` is the estimand: the factor by which the drug changed the transient
    rate, with a credible interval. `event_prob` is the per-bin posterior, from which
    `events()` picks peaks -- the two are different questions and the interval caveat
    applies only to the first (see MODEL_CARD.md).
    """

    session: Session
    settings: FitSettings
    beta0: tuple[float, float]
    beta_inj: tuple[float, float]
    event_prob: np.ndarray
    tonic: np.ndarray
    stopped_at: int
    seconds: float
    # Which device this fit ran on. A property of the fit, not of the machine: `auto`
    # sends one recording to the CPU and a cohort to the GPU, so two runs on the same
    # host legitimately differ, and only the run itself knows which it was.
    device: str = "cpu"
    # What the fit settled on, enough to run the generative process forward (`simulate`):
    # the joint posterior of (beta0, beta_inj), the mark distribution, the noise and the
    # kernel. Optional so results built before 2026-10-16 still load.
    beta_mean: tuple[float, float] | None = None
    beta_cov: tuple[tuple[float, float], tuple[float, float]] | None = None
    mu_a: float | None = None
    sigma_a: float | None = None
    sigma_y: float | None = None
    ar_coeffs: tuple[float, ...] = ()
    kernel: np.ndarray | None = None

    def simulate(self, n: int = 1, seed: int = 0) -> list[dict[str, Any]]:
        """Recordings the fitted model generates: a posterior predictive check.

        Each draw samples (beta0, beta_inj) from their joint posterior, events from the
        intensity, marks from the fitted distribution, convolves them with the kernel and
        adds the fitted tonic and autoregressive noise. It needs no ground truth, so it
        applies to experimental recordings: a draw that looks unlike the recording says the
        fit has not captured it. Returns one dict per draw with the trace `y` and the
        number of events `n_events`.
        """
        if self.kernel is None or self.mu_a is None or self.sigma_y is None:
            raise ValueError("this result does not carry the fitted model; refit it")
        from photosvi.models.points import AdditiveEmission
        rng = np.random.default_rng(seed)
        T, inj, dt = self.session.n_samples, self.session.inj_t, 1.0 / self.session.fs
        mean = np.asarray(self.beta_mean if self.beta_mean is not None
                          else (self.beta0[0], self.beta_inj[0]), float)
        cov = (np.asarray(self.beta_cov, float) if self.beta_cov is not None
               else np.diag([self.beta0[1] ** 2, self.beta_inj[1] ** 2]))
        post = np.arange(T) >= inj
        em = AdditiveEmission(sigma_y=self.sigma_y, ar_coeffs=list(self.ar_coeffs) or None)
        gen = torch.Generator().manual_seed(int(seed))
        out = []
        for _ in range(n):
            b0, binj = rng.multivariate_normal(mean, cov)
            p = -np.expm1(-np.exp(b0 + binj * post) * dt)
            s = rng.random(T) < p
            a = np.exp(self.mu_a + self.sigma_a * rng.standard_normal(T))
            c = np.convolve(s * a, self.kernel)[:T]
            with torch.no_grad():
                y = em.sample(torch.as_tensor(self.tonic + c, dtype=torch.float32)
                              .reshape(1, -1), generator=gen)["y"]
            out.append(dict(y=y.reshape(-1).numpy(), n_events=int(s.sum())))
        return out

    @property
    def rate_ratio(self) -> dict[str, float]:
        loc, scale = self.beta_inj
        return dict(estimate=math.exp(loc),
                    lo=math.exp(loc - 1.645 * scale),
                    hi=math.exp(loc + 1.645 * scale),
                    level=0.90)

    def events(self, threshold: float | None = None) -> dict[str, np.ndarray]:
        """Peak-picked events above `threshold`, after non-maximum suppression."""
        r = self.settings.resolved(fs=self.session.fs)
        return events_from_posterior(
            self.event_prob, fs=self.session.fs,
            threshold=r["event_threshold"] if threshold is None else threshold,
            refractory=r["refractory"])

    def rate_per_min(self, threshold: float | None = None) -> dict[str, float]:
        """Observed event rate before and after the injection, in events per minute.

        A descriptive counterpart to `rate_ratio`: this counts detections, while
        `rate_ratio` is a parameter of the model. They answer the same question by
        different routes and are not obliged to agree exactly.
        """
        ev = self.events(threshold)["bins"]
        inj, n, fs = self.session.inj_t, self.session.n_samples, self.session.fs
        pre_min, post_min = inj / fs / 60, (n - inj) / fs / 60
        return dict(
            pre=float((ev < inj).sum()) / pre_min if pre_min > 0 else float("nan"),
            post=float((ev >= inj).sum()) / post_min if post_min > 0 else float("nan"),
        )

    def to_dict(self) -> dict[str, Any]:
        ev = self.events()
        return dict(
            session=self.session.summary(),
            settings=self.settings.to_dict(),
            resolved=self.settings.resolved(fs=self.session.fs),
            beta0=list(self.beta0),
            beta_inj=list(self.beta_inj),
            rate_ratio=self.rate_ratio,
            rate_per_min=self.rate_per_min(),
            n_events=int(ev["bins"].size),
            stopped_at=self.stopped_at,
            seconds=round(self.seconds, 1),
            device=self.device,
            fitted=dict(mu_a=self.mu_a, sigma_a=self.sigma_a, sigma_y=self.sigma_y,
                        ar_coeffs=list(self.ar_coeffs),
                        n_masked=(0 if self.session.mask is None
                                  else int((~self.session.mask).sum()))),
        )

    def save(self, out: Path | str) -> Path:
        """Write a run directory: result.json, events.csv, and the fitted curves.

        Files rather than a database, and the same layout the command line writes, so
        a GUI run and a scripted run are interchangeable downstream.
        """
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(self.to_dict(), indent=2))
        ev = self.events()
        lines = ["bin,time_s,probability"]
        lines += [f"{b},{t:.4f},{p:.6f}"
                  for b, t, p in zip(ev["bins"], ev["times_s"], ev["prob"], strict=True)]
        (out / "events.csv").write_text("\n".join(lines) + "\n")
        np.savez_compressed(out / "curves.npz", trace=self.session.trace,
                            event_prob=self.event_prob, tonic=self.tonic)
        return out


def load_doric_session(path: Path | str, *, inj_start_s: float | None = None,
                       inj_end_s: float | None = None, event: str | None = None,
                       mask_min_s: float = 0.0, als_lambda: float | None = None,
                       name: str | None = None, **metadata) -> tuple[Session, dict]:
    """A Doric recording as a Session: isosbestic-corrected dF/F, injection and its mask.

    The injection is the user's `inj_start_s` (and `inj_end_s`, the end of the handling
    window) when given; otherwise the file's injection event -- `event` by name, or the
    last one recorded -- and an error if there is neither. Bins from the injection to the
    later of `inj_end_s` and `inj_start_s + mask_min_s` are excluded from the likelihood:
    handling moves the fibre and produces transient-shaped artefacts at the moment the rate
    is meant to change. Returns the Session and a dict saying where the times came from.
    """
    from photosvi.data.doric import read_doric
    from photosvi.data.preprocess import correct_isosbestic

    path = Path(path)
    rec = read_doric(path)
    if inj_start_s is not None:
        start = float(inj_start_s)
        end = float(inj_end_s) if inj_end_s is not None else start
        source = "user"
    else:
        if not rec.events:
            raise ValueError(f"{path} records no injection events: give the injection "
                             "times (inj_start_s, inj_end_s)")
        key = event if event is not None else list(rec.events)[-1]
        if key not in rec.events:
            raise ValueError(f"no event {key!r} in {path}; it has {sorted(rec.events)}")
        start, end = rec.events[key]
        if inj_end_s is not None:
            end = float(inj_end_s)
        source = f"doric event {key!r}"
    fs = rec.fs
    pp = correct_isosbestic(rec.signal, rec.isosbestic, fs, inj_t=int(round(start * fs)),
                            smooth_green=False, als_lambda=als_lambda)
    trace = np.asarray(pp["dff"], np.float32)
    shift = int(round(start * fs)) - int(pp["inj_t"])   # the samples the trim removed
    inj_t = int(pp["inj_t"])
    lo = inj_t
    hi = int(round(max(end, start + mask_min_s) * fs)) - shift
    mask = None
    if hi > lo:
        mask = np.ones(trace.size, dtype=bool)
        mask[lo:min(hi, trace.size)] = False
    info = dict(source=source, inj_start_s=start, inj_end_s=end, mask_s=(hi - lo) / fs,
                fs=fs, layout=rec.layout, events=rec.events, trim_s=shift / fs)
    return (Session(trace=trace, fs=fs, inj_t=inj_t, name=name or path.stem, mask=mask,
                    metadata=metadata), info)


def load_session(path: Path | str, *, fs: float, inj_t: int | None = None,
                 inj_s: float | None = None, column: str | int | None = None,
                 name: str | None = None, **metadata) -> Session:
    """Read one recording from .csv, .txt, .npy or .npz.

    `inj_s` is accepted as well as `inj_t` because an injection is recorded in
    seconds by whoever ran the experiment, and converting by hand is a good way to be
    off by a factor of the sampling rate.
    """
    path = Path(path)
    if inj_t is None and inj_s is None:
        raise ValueError("give the injection as inj_t (bins) or inj_s (seconds)")
    if inj_t is None:
        inj_t = int(round(inj_s * fs))

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        z = np.load(path)
        if isinstance(column, str):
            key = column
        elif len(z.files) == 1:
            key = z.files[0]
        else:
            raise ValueError(
                f"{path} holds {len(z.files)} arrays ({z.files}); pass `column` to "
                "say which one is the trace"
            )
        arr = z[key]
    elif suffix in (".csv", ".txt", ".tsv"):
        arr = _read_delimited(path, column)
    else:
        raise ValueError(
            f"unsupported file type {suffix!r}; expected .csv, .txt, .tsv, .npy or .npz"
        )
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim != 1:
        raise ValueError(
            f"expected one trace, got an array of shape {arr.shape}; pass `column` "
            "to choose one"
        )
    return Session(trace=arr, fs=fs, inj_t=inj_t, name=name or path.stem,
                   metadata=metadata)


def _read_delimited(path: Path, column: str | int | None) -> np.ndarray:
    """One column of a delimited text file, by header name or index."""
    text = path.read_text().strip().splitlines()
    if not text:
        raise ValueError(f"{path} is empty")
    delim = "\t" if "\t" in text[0] else ("," if "," in text[0] else None)
    head = text[0].split(delim) if delim else [text[0]]
    has_header = any(not _is_number(c) for c in head)
    if isinstance(column, str):
        if not has_header:
            raise ValueError(f"{path} has no header, so column {column!r} cannot be found")
        try:
            idx = [c.strip() for c in head].index(column)
        except ValueError:
            raise ValueError(
                f"column {column!r} not in {[c.strip() for c in head]}"
            ) from None
    elif isinstance(column, int):
        idx = column
    elif len(head) == 1:
        idx = 0
    elif has_header:
        idx = 1                    # conventionally time, value
    else:
        # Several columns and no header: defaulting to the first would usually pick
        # the timestamps and fit a monotonic ramp -- a wrong answer with no warning.
        raise ValueError(
            f"{path} has {len(head)} columns and no header, so which one holds the "
            "signal is a guess; pass `column` as an index"
        )
    rows = text[1:] if has_header else text
    return np.array([float((r.split(delim) if delim else [r])[idx]) for r in rows])


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def events_from_posterior(prob, *, fs: float, threshold: float,
                          refractory: int) -> dict[str, np.ndarray]:
    """Turn a per-bin event posterior into a list of events.

    Separate from `Result` so that re-thresholding a stored run uses the same rule
    the fit reported, rather than a second implementation that agrees today.

    The posterior is per-bin, so one transient contributes a cluster of adjacent
    high-probability bins; suppression is what turns that cluster into one event
    rather than several. The refractory width is a reporting choice, not a
    biological claim, and it should be stated wherever counts are.

    Re-thresholding is cheap and correct: nothing about the fit depends on the
    threshold. It is a readout of the posterior, applied afterwards. The rate ratio,
    by contrast, is a parameter of the model and does not move when this does --
    which is the whole point of estimating it rather than counting detections.
    """
    from photosvi.benchmark.detectors import nms, peaks_from_score

    prob = np.asarray(prob, dtype=float)
    t, p = peaks_from_score(prob)
    keep = p >= threshold
    t, p = nms(t[keep], p[keep], refractory=refractory)
    return dict(bins=t, times_s=t / fs, prob=p)


def spectrum(session: Session, *, n_per_segment: int = 4096,
             band: tuple[float, float] = (1.0, 10.0)) -> dict[str, Any]:
    """Power spectral density of a recording, and whether the transient band survives.

    The single most likely silent failure on real data is a trace that has already
    been low-pass filtered upstream. The model does not error on one: it explains a
    smooth curve with a point process and returns a rate ratio, and the only symptom
    is that few events are found -- which looks like a biological result.

    The spectrum makes it visible before anything is fitted. The two components live
    in separate places: the pharmacokinetic carrier below 0.01 Hz, and the transients
    in a shoulder from about 1 Hz up. `band_ratio` compares the power inside that
    shoulder to the power just below it; a trace whose transients have been filtered
    away has almost none, and one that never had transients looks the same -- so this
    diagnoses the trace, not the biology, and says so.
    """
    from scipy import signal as _sig

    y = session.trace.astype(np.float64)
    nps = int(min(n_per_segment, y.size))
    freq, psd = _sig.welch(y, fs=session.fs, nperseg=nps)
    lo, hi = band
    inband = (freq >= lo) & (freq <= hi)
    below = (freq >= lo / 10) & (freq < lo)
    ratio = (float(psd[inband].mean() / psd[below].mean())
             if inband.any() and below.any() and psd[below].mean() > 0 else float("nan"))
    # A hard floor rather than a soft hint: the point of the check is to stop a fit
    # that would return a number from a trace with nothing in it to fit.
    suspect = ratio == ratio and ratio < 0.02
    return dict(
        freq=freq.tolist(), psd=psd.tolist(), band=[lo, hi], band_ratio=ratio,
        suspect_filtered=bool(suspect),
        message=(
            f"the {lo}-{hi} Hz band carries {ratio:.1%} of the power just below it; "
            "if this recording was low-pass filtered upstream the transients are "
            "gone and a fit will find almost none"
            if suspect else
            f"the {lo}-{hi} Hz band carries {ratio:.1%} of the power just below it"
        ),
    )


def fit_session(session: Session, settings: FitSettings | None = None, *,
                on_progress: Callable[[dict[str, Any]], None] | None = None,
                probe_every: int = 250) -> Result:
    """Fit one recording and return its readouts.

    The model is built through `PointsSetup`, the same path the paper's benchmark
    uses, so this cannot drift from the validated configuration -- and stopping uses
    the same plateau rule, so it converges rather than running a fixed number of
    steps that happens to be too few.

    `on_progress` is called every `probe_every` steps with the step number, the
    running ELBO and the current estimate. A forty-minute recording is ten minutes of
    fitting on four CPU threads, so anything with a user in front of it needs a
    progress channel rather than a return value.
    """
    import time

    import pyro
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import ClippedAdam

    from photosvi.benchmark import matched_filter_score
    from photosvi.experiments.arms import PointsSetup
    from photosvi.experiments.marks import estimate_mark_distribution
    from photosvi.experiments.sweep import mf_anchor
    from photosvi.inference.points import EstimandPlateau, resolve_device
    from photosvi.models.points import (
        AdditiveEmission,
        PhasicEventProcess,
        SingleTraceModel,
        WindowedCategoricalEventGuide,
        build_kernel,
        estimate_ar_coeffs,
        estimate_log_noise_scale,
        fit_tonic_baseline,
    )

    st = settings or FitSettings()
    r = st.resolved(fs=session.fs)
    torch.set_num_threads(max(1, r["threads"]))
    dt = 1.0 / session.fs
    y = torch.as_tensor(session.trace, dtype=torch.float32).reshape(1, -1)
    inj = int(session.inj_t)

    kernel = build_kernel(dict(r["kernel"]), r["kernel_length"], dt)
    mf = matched_filter_score(y, kernel, inj, dt=dt)
    b0a, _ = mf_anchor(mf, inj, dt, thresh=r["mf_thresh"], refractory=r["refractory"])

    # Read off the recording, as the paper reads them: the AR coefficients by Yule-Walker
    # on the pre-injection trace, sigma_y from the successive differences corrected for
    # that correlation (the white-noise estimate is 22 times too small on dLight at 48 Hz),
    # and the centre of the mark prior from the pre-injection peak prominences.
    phi = r["ar_coeffs"] if r["ar_coeffs"] is not None else estimate_ar_coeffs(y, inj)
    sigma_y = r["sigma_y"]
    if sigma_y is None:
        sigma_y = math.exp(estimate_log_noise_scale(y, inj, ar_coeffs=phi or None))
    mu_a, sigma_a = r["mu_a"], r["sigma_a"]
    if r["mark_noise_units"] is not None:
        from photosvi.experiments.marks import noise_unit_marks
        mu_a, sigma_a = noise_unit_marks(sigma_y, *r["mark_noise_units"])
    elif mu_a is None or sigma_a is None:
        m_hat, s_hat = estimate_mark_distribution([session.trace[:inj]], session.fs)
        mu_a = m_hat if mu_a is None else mu_a
        sigma_a = s_hat if sigma_a is None else sigma_a

    setup = PointsSetup(
        dt=dt, kernel_length=r["kernel_length"], kernel_spec=dict(r["kernel"]),
        mu_a=mu_a, sigma_a=sigma_a, sigma_y=sigma_y,
        beta0_prior_loc=b0a, beta0_prior_scale=r["beta0_prior_scale"],
        beta_inj_prior_loc=r["beta_inj_prior_loc"],
        beta_inj_prior_scale=r["beta_inj_prior_scale"],
        tau=r["tau"], warm=dict(r["warm"]), window=r["window"],
        beta_prior=r["beta_prior"], beta_guide=r["beta_guide"],
        mark_gate_floor=r["mark_gate_floor"],
    )

    tonic = fit_tonic_baseline(y, inj, dt=dt)
    # The paper's final model refines the tonic jointly with the rest (as the benchmarks do);
    # earlier versions of this function held it at its least-squares pre-fit.
    if r["architecture"] != "paper_final":
        for prm in tonic.parameters():
            prm.requires_grad_(False)
    emission = AdditiveEmission(sigma_y=setup.sigma_y, ar_coeffs=phi or None)
    emission.log_sigma_y.requires_grad_(False)

    model = SingleTraceModel(
        PhasicEventProcess(kernel, beta0=b0a, beta_inj=0.0, mu_a=setup.mu_a,
                           sigma_a=setup.sigma_a, dt=dt, history=None),
        tonic, emission, tau=setup.tau,
        event_guide=WindowedCategoricalEventGuide(
            torch.as_tensor(mf, dtype=torch.float32), window=setup.window,
            # thresh and gain only: `hi`/`floor` belong to the warm-start guide,
            # not this one. Matching benchmark_guppy.py exactly is the point.
            thresh=setup.warm["thresh"], gain=setup.warm["gain"]),
        analytic_s_kl=True, mark_kl="gated", couple_events=False,
        mark_gate_floor=setup.mark_gate_floor,
        **setup.beta_prior_kwargs(),
    )
    # The amplitude-rate trade-off: the likelihood pins rate x amplitude^2 far better than
    # either, so the mark distribution is anchored on the peaks with a stated width.
    if r["mark_noise_units"] is not None:
        from photosvi.experiments.marks import fix_marks_at
        fix_marks_at(model.phasic, mu_a, sigma_a)
    else:
        model.phasic.set_mark_prior(mu_a, sigma_a, mu_sd=r["mark_prior_sd"],
                                    log_sigma_sd=r["mark_prior_sd"])
    model.train(False)
    device = resolve_device(r["device"], batch=1)
    model, y = model.to(device), y.to(device)
    # Bins outside the session's keep-mask do not enter the likelihood (the injection's
    # handling artefact, typically). Session has carried a mask since 2026-10-01 and the
    # GUI sets one, but until 2026-10-16 it never reached the model.
    obs_mask = (None if session.mask is None
                else torch.as_tensor(session.mask, dtype=torch.bool, device=device).reshape(1, -1))

    pyro.clear_param_store()
    pyro.set_rng_seed(r["seed"])
    svi = SVI(model.model, model.guide, ClippedAdam({"lr": r["lr"]}),
              Trace_ELBO(max_plate_nesting=2))
    plateau = EstimandPlateau(tol=r["stop_tol"], window=r["stop_window"],
                              min_steps=r["steps"])
    t0, stopped_at = time.time(), None
    for step in range(1, r["max_steps"] + 1):
        loss = svi.step(y, inj, obs_mask=obs_mask)
        if on_progress is not None and (step % probe_every == 0 or step == 1):
            bp = model.beta_posterior()
            on_progress(dict(step=step, total=r["max_steps"], loss=float(loss),
                             beta_inj=bp["beta_inj"][0],
                             rate_ratio=math.exp(bp["beta_inj"][0]),
                             min_steps=r["steps"], device=str(device),
                             elapsed=time.time() - t0))
        if step % r["stop_probe"] == 0 and plateau.update(
                step, model.beta_posterior()["beta_inj"][0]):
            stopped_at = step
            break
    seconds = time.time() - t0

    bp = model.beta_posterior()
    with torch.no_grad():
        prob = model.event_posterior(y, inj).reshape(-1).cpu().numpy()
        base = tonic.baseline(session.n_samples, inj).reshape(-1).cpu().numpy()
        b_mean, b_cov = None, None
        if r["beta_guide"] == "fullrank" and r["beta_prior"] != "uniform":
            L = pyro.param("q_beta_tril").detach().cpu()
            b_mean = tuple(float(v) for v in pyro.param("q_beta_loc").detach().cpu())
            b_cov = tuple(tuple(float(v) for v in row) for row in (L @ L.T))
        mu_fit = float(model.phasic.mu_a.detach().cpu())
        sigma_fit = float(model.phasic.sigma_a.detach().cpu())
    return Result(session=session, settings=st, beta0=tuple(bp["beta0"]),
                  beta_inj=tuple(bp["beta_inj"]), event_prob=prob, tonic=base,
                  stopped_at=stopped_at or r["max_steps"], seconds=seconds,
                  device=str(device), beta_mean=b_mean, beta_cov=b_cov,
                  mu_a=mu_fit, sigma_a=sigma_fit, sigma_y=float(sigma_y),
                  ar_coeffs=tuple(phi), kernel=kernel.detach().cpu().numpy().reshape(-1))


def fit_study(study: Study, settings: FitSettings | None = None,
              **kw) -> Sequence[Result]:
    """Fit each session independently.

    Deliberately NOT the hierarchical model. Fitting a cohort jointly is a different
    estimand -- a population contrast rather than a per-session effect -- and giving
    it the same name here would let a caller believe they had one when they had the
    other. The joint fit will arrive as `fit_cohort` with its own return type.
    """
    return [fit_session(s, settings, **kw) for s in study.sessions]


# The kernel calibration lives in its own module because it is a step before a fit rather
# than part of one; it is exported here so that this module stays the one contract callers
# (the GUI among them) import from.
from photosvi.kernel_calibration import SENSORS as KERNEL_SENSORS  # noqa: E402, I001
from photosvi.kernel_calibration import KernelCalibration, calibrate_kernel  # noqa: E402
