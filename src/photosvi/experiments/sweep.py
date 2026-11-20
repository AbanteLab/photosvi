"""The grid driver: one process runs one (arm, seed) cell and writes one JSON file.

That granularity is the design. It gives server-native fan-out with no scheduler
(`xargs -P`, or a job array if one ever exists), resumability for free (a cell whose
output file exists is skipped), and no shared-memory parallelism to debug. The cost is
that the trace is regenerated per cell, which is milliseconds.

Output is JSON, not pickle: it must stay readable without this package installed, and it
must diff. Each file carries its own provenance -- git SHA, host, library versions, wall
time -- because a server run without those is unreproducible three weeks later.

**No figures here.** The sweep produces data; plotting happens on whatever machine reads
the results back. A plotting bug should not cost a compute run.
"""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from photosvi.benchmark import matched_filter_score, nms, peaks_from_score
from photosvi.experiments.arms import ARMS, PointsSetup
from photosvi.experiments.heldout import heldout_probe, null_recon_ll
from photosvi.experiments.trajectory import fit_trajectory
from photosvi.models.points import (
    AdditiveEmission,
    PhasicEventProcess,
    TonicBaseline,
    build_kernel,
    fit_tonic_baseline,
)

# ------------------------------------------------------------------ config & provenance


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def provenance() -> dict:
    """Everything needed to identify what produced a result file."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent, timeout=10,
        ).stdout.strip() or None
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent, timeout=10,
        ).stdout.strip())
    except Exception:
        sha, dirty = None, None
    import pyro
    return dict(
        git_sha=sha, git_dirty=dirty, host=socket.gethostname(),
        python=platform.python_version(), torch=torch.__version__, pyro=pyro.__version__,
        numpy=np.__version__, threads=torch.get_num_threads(),
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


# --------------------------------------------------------------------------- synthetic


def _em(cfg, fit_key, gen_key):
    """`emission.<fit_key>` if given, else `emission.<gen_key>`, else falsy.

    Two keys so the noise can be mis-specified the way the kernel can: what generates
    and what is fitted are separate settings, and a config that gives only the
    generating one fits what it generated.
    """
    e = cfg.get("emission", {})
    return e.get(fit_key, e.get(gen_key, None))


def setup_from_config(cfg: dict, beta0_prior_loc: float) -> PointsSetup:
    k, gt = cfg["kernel"], cfg["gt"]
    return PointsSetup(
        dt=1.0 / cfg["data"]["fs"],
        kernel_length=k["length"], kernel_spec=dict(k["fit"]),
        # `emission.fit_ar_rho` if given, else `emission.ar_rho`, else white. Kept
        # separate from the generating value so a config can mis-specify the noise the
        # same way it can mis-specify the kernel.
        ar_coeffs=_em(cfg, "fit_ar_coeffs", "ar_coeffs")
        or ([_em(cfg, "fit_ar_rho", "ar_rho")] if _em(cfg, "fit_ar_rho", "ar_rho") else None),
        mu_a=gt["mu_a"], sigma_a=gt["sigma_a"], sigma_y=gt["sigma_y"],
        beta0_prior_loc=beta0_prior_loc,
        beta0_prior_scale=cfg["fit"]["beta0_prior_scale"],
        beta_inj_prior_loc=cfg["fit"]["beta_inj_prior_loc"],
        beta_inj_prior_scale=cfg["fit"]["beta_inj_prior_scale"],
        tau=cfg["fit"]["tau"],
        warm=dict(cfg.get("warm", {})),
        window=int(cfg.get("guide", {}).get("window", 10)),
        beta_prior=cfg["fit"].get("beta_prior", "normal"),
        beta_guide=cfg["fit"].get("beta_guide", "fullrank"),
        beta0_halfwidth=float(cfg["fit"].get("beta0_halfwidth", 5.0)),
        beta_inj_halfwidth=float(cfg["fit"].get("beta_inj_halfwidth", 4.0)),
    )


def gen_spec(k: dict) -> dict:
    """The generating kernel's spec: `kernel.gen` if given, else `kernel.fit`.

    Defaulting to the fit spec is the point. A generative model whose phasic carrier
    has a different mean structure from the inference model's has no representable
    truth, so beta recovery would be scoring misspecification rather than the
    estimator. Set `gen` explicitly only to probe that mismatch on purpose.
    """
    return dict(k.get("gen", k["fit"]))


def generate(cfg: dict, seed: int) -> tuple[torch.Tensor, np.ndarray]:
    """One realization of the frozen GT: tonic + phasic + noise, onsets kept.

    The generating kernel is `kernel.gen`, which defaults to `kernel.fit` -- see
    `gen_spec` for why the default is to match rather than to mismatch.
    """
    d, k, gt, ton = cfg["data"], cfg["kernel"], cfg["gt"], cfg["tonic"]
    dt = 1.0 / d["fs"]
    g = torch.Generator().manual_seed(int(seed))
    kernel = build_kernel(gen_spec(k), k["length"], dt)
    x_hat = TonicBaseline(dt=dt, **ton).baseline(d["T"], d["inj_t"])
    # The marks the benchmark itself draws (experiments.marks): under the final configs the
    # config's absolute mu_a with the noise-unit width, so calibration and spectra measure
    # the same simulator as the benchmark cells.
    from photosvi.experiments.marks import noise_unit_generation
    mu_g, sig_g = noise_unit_generation(cfg.get("fit"), gt["sigma_y"], gt["mu_a"], gt["sigma_a"])
    phasic = PhasicEventProcess(kernel, beta0=gt["beta0"], beta_inj=gt["beta_inj"],
                                mu_a=mu_g, sigma_a=sig_g, dt=dt, history=None)
    ex = phasic.sample(d["T"], d["inj_t"], batch=1, generator=g)
    em = cfg.get("emission", {})
    phi = em.get("ar_coeffs") or ([em["ar_rho"]] if em.get("ar_rho") else None)
    y = AdditiveEmission(sigma_y=gt["sigma_y"], ar_coeffs=phi).sample(
        x_hat + ex["c"], generator=g)["y"]
    return y, ex["s"][0].nonzero().reshape(-1).numpy()


def mf_anchor(mf: np.ndarray, inj_t: int, dt: float, *, thresh: float, refractory: int) -> tuple:
    """Matched-filter baseline log-rate: NMS'd pre-injection peaks over pre exposure.

    `estimate_log_baseline_rate` is deliberately not used: it thresholds at 2 *z-units*
    assuming a z-scored dF/F trace, and this GT carries a tonic of amplitude ~10, so it
    counts tonic ripples as events. The matched-filter score is detrended and
    standardized, hence scale-free.

    This is an *anchor* -- it locates the beta0 prior and is drawn as a reference line.
    It never constrains a fit; the prior around it is deliberately wide.
    """
    pk, ps = nms(*peaks_from_score(mf), refractory=refractory)
    n_pre = int(((pk < inj_t) & (ps > thresh)).sum())
    return math.log(max(n_pre, 1) / (inj_t * dt)), n_pre


# -------------------------------------------------------------------------------- cell


def rate_readout(b0: float, bi: float, cfg: dict, n_call: int, n_true: int) -> dict:
    """The fitted intensity in probability space, not on the log axis.

    beta is a log-rate, so a plausible-looking beta can still be an absurd process:
    the sweep's betas span four orders of magnitude in ev/min while looking like
    near-misses in log space. `n_implied` integrates the fitted intensity over the
    trace -- the number of events the *rate* claims. Comparing it against `n_call`
    (what the guide actually flags) exposes the rate block and the event block
    having settled on different processes, which no single-block metric shows.
    """
    d = cfg["data"]
    dt, inj_t, T = 1.0 / d["fs"], d["inj_t"], d["T"]
    mins_pre, mins_post = inj_t * dt / 60.0, (T - inj_t) * dt / 60.0
    lam_pre, lam_post = math.exp(b0), math.exp(b0 + bi)
    n_pre, n_post = lam_pre * 60.0 * mins_pre, lam_post * 60.0 * mins_post
    return dict(
        lam_pre_hz=lam_pre, lam_post_hz=lam_post,
        p_pre=1.0 - math.exp(-lam_pre * dt), p_post=1.0 - math.exp(-lam_post * dt),
        ev_per_min_pre=lam_pre * 60.0, ev_per_min_post=lam_post * 60.0,
        rate_ratio=math.exp(bi),
        n_implied_pre=n_pre, n_implied_post=n_post, n_implied_total=n_pre + n_post,
        implied_over_true=(n_pre + n_post) / max(n_true, 1),
        implied_over_called=(n_pre + n_post) / max(n_call, 1),
    )


def cell_path(outdir: str | Path, arm: str, seed: int) -> Path:
    return Path(outdir) / f"{arm.replace('+', '_plus_')}__seed{seed}.json"


def run_cell(cfg: dict, arm: str, seed: int, *, verbose: bool = True) -> dict:
    """Fit one arm on one seed, then replay the held-out probe over its recorded betas."""
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; known: {sorted(ARMS)}")
    d, det, ho = cfg["data"], cfg["detection"], cfg["heldout"]
    dt, inj_t = 1.0 / d["fs"], d["inj_t"]
    tol = int(round(det["tol_s"] * d["fs"]))

    y, onsets = generate(cfg, seed)
    y_ho, onsets_ho = generate(cfg, seed + ho["seed_offset"])

    k = cfg["kernel"]
    k_fit = build_kernel(dict(k["fit"]), k["length"], dt)
    mf = matched_filter_score(y, k_fit, inj_t, dt=dt)
    mf_ho = matched_filter_score(y_ho, k_fit, inj_t, dt=dt)

    b0_anchor, n_pre_mf = mf_anchor(
        mf, inj_t, dt, thresh=det.get("mf_thresh", 2.5), refractory=det["refractory"]
    )
    setup = setup_from_config(cfg, b0_anchor)

    rec = fit_trajectory(
        arm, y, onsets, inj_t, setup, mf=mf, seed=seed,
        steps=cfg["fit"]["steps"], probe=cfg["fit"]["probe"], lr=cfg["fit"]["lr"],
        tol=tol, refractory=det["refractory"], thresh=det["thresh"], verbose=verbose,
    )

    # held-out: replayed over the recorded betas. Equivalent to running it inline because
    # everything but (beta0, beta_inj) is pinned, and it cannot disturb the fit's store.
    tonic_ho = fit_tonic_baseline(y_ho, inj_t, dt=dt)
    ho_null = null_recon_ll(y_ho, inj_t, setup, tonic_ho)
    ix = list(range(0, len(rec["steps"]), ho["every"]))
    if ix[-1] != len(rec["steps"]) - 1:
        ix.append(len(rec["steps"]) - 1)

    t0 = time.time()
    rec["ho_steps"], rec["ho_recon"], rec["ho_elbo"] = [], [], []
    for i in ix:
        out = heldout_probe(
            rec["b0"][i], rec["bi"][i], y_ho, inj_t, setup, tonic_ho,
            mf_ho=mf_ho, steps=ho["steps"], lr=ho["lr"], seed=seed,
        )
        rec["ho_steps"].append(rec["steps"][i])
        rec["ho_recon"].append(out["recon"] - ho_null)      # delta on the no-event baseline
        rec["ho_elbo"].append(out["elbo"] - ho_null)
        if verbose:
            print(f"  [{arm:10s} s{seed}] held-out @ {rec['steps'][i]:6d}  "
                  f"d-recon {rec['ho_recon'][-1]:+10.5f}  d-elbo {rec['ho_elbo'][-1]:+10.5f}",
                  flush=True)
    rec["heldout_secs"] = time.time() - t0

    rec.update(
        label=ARMS[arm].label,
        guide_family=ARMS[arm].guide,
        coupled=ARMS[arm].couple,
        n_onsets=int(len(onsets)),
        n_onsets_pre=int((onsets < inj_t).sum()),
        n_onsets_post=int((onsets >= inj_t).sum()),
        n_onsets_heldout=int(len(onsets_ho)),
        b0_anchor=b0_anchor,
        n_pre_mf=n_pre_mf,
        rates=rate_readout(rec["b0"][-1], rec["bi"][-1], cfg,
                           n_call=rec["n_call"][-1], n_true=int(len(onsets))),
        gt_rates=rate_readout(cfg["gt"]["beta0"], cfg["gt"]["beta_inj"], cfg,
                              n_call=int(len(onsets)), n_true=int(len(onsets))),
        ar_gen=list(cfg.get("emission", {}).get("ar_coeffs")
                    or ([cfg["emission"]["ar_rho"]]
                        if cfg.get("emission", {}).get("ar_rho") else [])),
        ar_fit=list(setup.ar_coeffs or ()),
        sum_h_gen=float(build_kernel(gen_spec(k), k["length"], dt).sum()),
        kernel_spec_gen=gen_spec(k),
        kernel_spec_fit=dict(k["fit"]),
        sum_h_fit=float(setup.kernel().sum()),
        ho_null=ho_null,
        setup=asdict(setup),
        config=cfg,
        provenance=provenance(),
    )
    return rec


def write_cell(rec: dict, outdir: str | Path) -> Path:
    out = cell_path(outdir, rec["arm"], rec["seed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(out)          # atomic: a killed job never leaves a half-written result
    return out


def pin_threads(n: int = 1) -> None:
    """Pin every math backend to `n` threads.

    With no scheduler, workers inherit torch's default thread count -- 16 or 32 on a
    workstation. Fan out 15 of those and several hundred threads contend over the cores
    and the sweep runs *slower* than serial. This workload is small-tensor convolution
    over one trace, so one thread per worker is the right setting, not a compromise.
    Env vars must be set before the backends initialize; the torch call covers the rest.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n))
    torch.set_num_threads(n)
