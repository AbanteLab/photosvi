#!/usr/bin/env python3
"""Fit one benchmark recording and save everything the fit figure draws.

    SIG=2.0 python scripts/paper_fig_fit_data.py \
        --out $PHOTOSVI_RUNS/traces/fit_2.00.json

The recording is generated from the benchmark's own config, so the figure shows the model
on a cell of the grid the paper reports rather than on a trace made for the picture. The
previous input was a sigma_y of 0.50 -- a transient-to-noise ratio of twelve, far easier
than anything the benchmark or the recordings carry -- sitting in a scratch directory that
does not survive the session that made it.

Beyond the fit it writes the posterior-predictive draws, which are the figure's third panel
and the part that cannot be recovered from the fit alone: traces generated ancestrally from
the FITTED globals, which is the check that the fitted rate reproduces the recording rather
than merely scoring well on it.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import pyro
import torch
import yaml
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from photosvi.benchmark import matched_filter_score, run_benchmark  # noqa: E402
from photosvi.experiments.sweep import gen_spec, mf_anchor, setup_from_config  # noqa: E402
from photosvi.inference.points import EstimandPlateau, prepare, resolve_device  # noqa: E402
from photosvi.models.points import (  # noqa: E402
    AdditiveEmission,
    PhasicEventProcess,
    SingleTraceModel,
    TonicBaseline,
    WindowedCategoricalEventGuide,
    build_kernel,
    fit_tonic_baseline,
)
from photosvi.models.points.emission import estimate_log_noise_scale  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(REPO / "configs/benchmark_guppy_48.yaml"))
ap.add_argument("--out", required=True)
ap.add_argument("--sigma-y", type=float, default=float(os.environ.get("SIG", "2.0")),
                help="noise level; the benchmark grid's easiest is 2.0")
ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
ap.add_argument("--n-predictive", type=int, default=4)
ap.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
d, k, gt, ton, det = cfg["data"], cfg["kernel"], cfg["gt"], cfg["tonic"], cfg["detection"]
dt, T, INJ, fs = 1 / d["fs"], d["T"], d["inj_t"], d["fs"]
phi = cfg["emission"]["ar_coeffs"]

g = torch.Generator().manual_seed(a.seed)
x_true = TonicBaseline(dt=dt, **ton).baseline(T, INJ)
gen_kernel = build_kernel(gen_spec(k), k["length"], dt)
# Under the final configs the simulator's marks come from experiments.marks, exactly as in
# scripts/benchmark_guppy.py, so this trace is the benchmark cell with the same seed.
from photosvi.experiments.marks import noise_unit_generation  # noqa: E402
from photosvi.experiments.single_trace import build_single_trace, is_final  # noqa: E402

MU_GEN, SIG_GEN = noise_unit_generation(cfg.get("fit"), a.sigma_y, gt["mu_a"], gt["sigma_a"])
ex = PhasicEventProcess(gen_kernel, beta0=gt["beta0"], beta_inj=gt["beta_inj"],
                        mu_a=MU_GEN, sigma_a=SIG_GEN, dt=dt,
                        history=None).sample(T, INJ, 1, g)
y = AdditiveEmission(sigma_y=a.sigma_y, ar_coeffs=phi).sample(x_true + ex["c"],
                                                              generator=g)["y"]
onsets = ex["s"][0].nonzero().reshape(-1).numpy()
print(f"  generated: sigma_y {a.sigma_y}, seed {a.seed}, {len(onsets)} true onsets")

kf = build_kernel(dict(k["fit"]), k["length"], dt)
if is_final(cfg):
    model, _info = build_single_trace(
        y, INJ, kernel=kf, dt=dt, fit=dict(cfg["fit"], window=int(cfg.get("guide", {}).get(
            "window", cfg["fit"].get("window", 10)))), warm=cfg["warm"], detection=det,
        ar_coeffs=phi)
    setup = type("S", (), {"sigma_y": _info["sigma_y"], "mu_a": _info["mu_a"],
                           "sigma_a": _info["sigma_a"]})()
else:
    mf = matched_filter_score(y, kf, INJ, dt=dt)
    b0a, _ = mf_anchor(mf, INJ, dt, thresh=det["mf_thresh"], refractory=det["refractory"])
    setup = setup_from_config(cfg, b0a)
    setup.sigma_y = float(np.exp(estimate_log_noise_scale(y, INJ, ar_coeffs=phi)))
    em = AdditiveEmission(sigma_y=setup.sigma_y, ar_coeffs=phi)
    em.log_sigma_y.requires_grad_(False)
    model = SingleTraceModel(
        PhasicEventProcess(setup.kernel(), beta0=b0a, beta_inj=0.0, mu_a=setup.mu_a,
                           sigma_a=setup.sigma_a, dt=dt, history=None),
        fit_tonic_baseline(y, INJ, dt=dt), em, tau=setup.tau,
        beta0_prior_loc=b0a, beta_inj_prior_loc=0.0,
        event_guide=WindowedCategoricalEventGuide(
            torch.as_tensor(mf, dtype=torch.float32), window=setup.window,
            thresh=setup.warm["thresh"], gain=setup.warm["gain"]),
        couple_events=False, mark_gate_floor=cfg["fit"].get("mark_gate_floor", 0.0),
        **setup.beta_prior_kwargs())
model.train(False)
md, yd = prepare(model, y, resolve_device(a.device, batch=1))
pyro.clear_param_store()
pyro.set_rng_seed(a.seed)
f = cfg["fit"]
svi = SVI(md.model, md.guide, ClippedAdam({"lr": f["lr"]}), Trace_ELBO(max_plate_nesting=2))
pl = EstimandPlateau(tol=f["stop_tol"], window=f["stop_window"],
                     min_steps=f["stop_min_steps"])
for step in range(1, f["max_steps"] + 1):
    svi.step(yd, INJ)
    if step % f["stop_probe"] == 0 and pl.update(
            step, float(md.beta_posterior()["beta_inj"][0])):
        print(f"  plateau at {step}")
        break
    if step % 5000 == 0:
        print(f"    {step}", flush=True)

q = md.event_posterior(yd)[0].detach().cpu().numpy()
rec = md.reconstruct(yd, INJ)
res = run_benchmark(y.cpu(), onsets, kf, inj_t=INJ, fs=fs, tol_s=det["tol_s"],
                    refractory=det["refractory"], extra={"photosvi": q})

# --- posterior predictive -------------------------------------------------------------
# Drawn from the FITTED globals, not the true ones: each trace samples (beta0, beta_inj)
# from their posterior and then runs the generative process forward. A fit that scores well
# on detection can still imply a recording that looks nothing like the one it was given,
# and this is the only panel that would show it.
post = md.beta_posterior()
pp, pp_n = [], []
gpp = torch.Generator().manual_seed(a.seed + 1000)
tonic_fit = rec["tonic"][0].detach().cpu()
for i in range(a.n_predictive):
    b0 = float(np.random.default_rng(a.seed + i).normal(*post["beta0"]))
    bi = float(np.random.default_rng(a.seed + 100 + i).normal(*post["beta_inj"]))
    draw = PhasicEventProcess(kf, beta0=b0, beta_inj=bi, mu_a=setup.mu_a,
                              sigma_a=setup.sigma_a, dt=dt, history=None).sample(
                                  T, INJ, 1, gpp)
    yi = AdditiveEmission(sigma_y=setup.sigma_y, ar_coeffs=phi).sample(
        tonic_fit.unsqueeze(0) + draw["c"], generator=gpp)["y"]
    pp.append([round(float(v), 5) for v in yi[0].numpy()])
    pp_n.append(int(draw["s"][0].sum()))

out = dict(seed=a.seed, sigma_y=a.sigma_y, dt=dt, T=T, inj=INJ, fs=fs,
           y=[round(float(v), 5) for v in yd[0].cpu().numpy()],
           x_true=[round(float(v), 5) for v in x_true.detach().numpy()],
           x_fit=[round(float(v), 5) for v in tonic_fit.numpy()],
           q=[round(float(v), 5) for v in q],
           onsets=[int(v) for v in onsets],
           calls={nm: [int(v) for v in o.times] for nm, o in res.outcomes.items()},
           metrics={nm: o.metrics for nm, o in res.outcomes.items()},
           beta0=float(post["beta0"][0]), beta_inj=float(post["beta_inj"][0]),
           beta_inj_sd=float(post["beta_inj"][1]), sigma_y_est=setup.sigma_y,
           stopped_at=step, pp=pp, pp_n=pp_n)
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(a.out, "w"))
print(f"  wrote {a.out}: F1 "
      + ", ".join(f"{nm} {o.metrics['best_f1']:.3f}" for nm, o in res.outcomes.items()))
print(f"  true events {len(onsets)}, posterior-predictive counts {pp_n}")
