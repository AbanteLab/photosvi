"""One benchmark cell: simulate at a given noise level, fit, and score against GuPPy.

    mkdir -p runs_benchmark
    python - <<'P' | xargs -P 24 -n 2 sh -c 'OMP_NUM_THREADS=1 python scripts/benchmark_guppy.py $0 $1'
    import yaml; c=yaml.safe_load(open("configs/benchmark_guppy.yaml"))
    for s in c["grid"]["sigma_y"]:
        for d in c["grid"]["seeds"]: print(d, s)
    P

Fairness notes, because a benchmark is only as good as what it hands each side:

  * GuPPy runs at its published defaults (15 s chunks, 2.0/3.0 MAD thresholds). It
    is not tuned on these traces; tuning ours and not theirs would settle the
    comparison before it started.
  * sigma_y is **estimated from the trace** rather than taken from the simulation.
    The guide sweep hands the model its true noise level, which is fine when every
    arm gets the same favour, but GuPPy estimates its own noise per chunk and would
    be conceding an advantage it never had.
  * All three detectors are scored by one routine, at one tolerance, after the same
    non-maximum suppression. GuPPy is a *detector* and makes no rate estimate: the
    comparison is on detection alone, and the rate columns are reported for our
    model as something GuPPy does not attempt rather than as a contest it loses.
"""
import json
import math
import os
import sys

import pyro
import torch
import yaml
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from photosvi.benchmark import matched_filter_score, run_benchmark
from photosvi.experiments.marks import (
    fix_marks_at,
    noise_unit_generation,
    noise_unit_marks,
    noise_unit_mode,
)
from photosvi.experiments.single_trace import build_single_trace, is_final
from photosvi.experiments.sweep import gen_spec, mf_anchor, setup_from_config
from photosvi.inference.points import EstimandPlateau, prepare, resolve_device
from photosvi.models.points import (
    AdditiveEmission,
    PhasicEventProcess,
    SingleTraceModel,
    TonicBaseline,
    WindowedCategoricalEventGuide,
    build_kernel,
    estimate_log_noise_scale,
    fit_tonic_baseline,
)

SEED, SIGMA_Y = int(sys.argv[1]), float(sys.argv[2])
# Resumable at cell granularity, like the cohort runners. The grid gained two noise
# levels on 2026-09-16 (sigma_y 4 and 6, i.e. transient-to-noise 1.5 and 1.0, the
# regime the real recordings actually sit in) and re-running the 50 cells that were
# already done would cost a day and produce the same numbers.
OUT = os.environ.get("OUT", "runs_benchmark")
_dest = f"{OUT}/bm_{SIGMA_Y:.2f}_{SEED:02d}.json"
if os.path.exists(_dest):
    print(f"bm sigma={SIGMA_Y} seed={SEED}: already done")
    raise SystemExit(0)
# Device: argv[3], else $DEVICE, else "cpu". NOT "auto" -- this script is one cell of a
# 50-way CPU fan-out, and auto would pick CPU here anyway (see inference/points/device.py:
# at B = 1 the fan-out beats the GPU ~12x on throughput). Asking for cuda is honoured.
DEVICE = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("DEVICE", "cpu")
# Config: $BM_CONFIG, else the 25 Hz original. A path rather than a hard-coded name
# because there are now two -- the 25 Hz benchmark every earlier result came from, and
# benchmark_guppy_48.yaml at the recordings' own rate. Defaulting to the old one keeps
# every existing invocation reproducing exactly what it did before.
cfg = yaml.safe_load(open(os.environ.get("BM_CONFIG", "configs/benchmark_guppy.yaml")))
d, k, gt, ton, det = cfg["data"], cfg["kernel"], cfg["gt"], cfg["tonic"], cfg["detection"]
dt, T, INJ, fs = 1 / d["fs"], d["T"], d["inj_t"], d["fs"]

g = torch.Generator().manual_seed(SEED)
x_true = TonicBaseline(dt=dt, **ton).baseline(T, INJ)
# Noise-unit marks (opt-in, MARK_MODE=noise_units or fit.mark_mode): the marks are drawn
# from a fixed distribution in units of this cell's sigma_y, and the fit assumes the same
# distribution at its own estimate of sigma_y, with nothing about the marks fitted.
NU = noise_unit_mode(cfg.get("fit"))
MU_GEN, SIG_GEN = noise_unit_generation(cfg.get("fit"), SIGMA_Y, gt["mu_a"], gt["sigma_a"])
ex = PhasicEventProcess(
    build_kernel(gen_spec(k), k["length"], dt), beta0=gt["beta0"], beta_inj=gt["beta_inj"],
    mu_a=MU_GEN, sigma_a=SIG_GEN, dt=dt, history=None,
).sample(T, INJ, 1, g)
# The emission's colour, for generating and for fitting, kept separate so the noise can
# be mis-specified the way the kernel can. Absent, both are white and every result from
# before 2026-09-30 reproduces.
_em_cfg = cfg.get("emission", {})
PHI_GEN = _em_cfg.get("ar_coeffs") or None
PHI_FIT = _em_cfg.get("fit_ar_coeffs", PHI_GEN) or None
y = AdditiveEmission(sigma_y=SIGMA_Y, ar_coeffs=PHI_GEN).sample(
    x_true + ex["c"], generator=g)["y"]
onsets = ex["s"][0].nonzero().reshape(-1).numpy()
npre, npost = int((onsets < INJ).sum()), int((onsets >= INJ).sum())
mle = math.log((npost / ((T - INJ) * dt)) / (npre / (INJ * dt))) if npre and npost else float("nan")

k_fit = build_kernel(dict(k["fit"]), k["length"], dt)
device = resolve_device(DEVICE, batch=1)
pyro.set_rng_seed(SEED)
if is_final(cfg):
    # The paper's final single architecture (experiments.single_trace), shared with the
    # two-stage cohort arm and photosvi.api, so the detection benchmark fits the same model.
    model, _info = build_single_trace(
        y, INJ, kernel=k_fit, dt=dt, fit=dict(cfg["fit"], window=int(cfg.get("guide", {}).get(
            "window", cfg["fit"].get("window", 10)))), warm=cfg["warm"], detection=det,
        ar_coeffs=PHI_FIT)
    mf, b0a = _info["mf"], _info["b0a"]
    setup = type("S", (), {"sigma_y": _info["sigma_y"]})()
else:
    mf = matched_filter_score(y, k_fit, INJ, dt=dt)
    b0a, _ = mf_anchor(mf, INJ, dt, thresh=det["mf_thresh"], refractory=det["refractory"])
    setup = setup_from_config(cfg, b0a)
    # Estimated from the trace, not given -- GuPPy estimates its own noise per chunk and
    # handing ours the true value would concede an advantage it never had. `ar_coeffs` is
    # passed because the estimator's /sqrt(2) assumes white differences: without it a trace
    # with rho_1 = 0.998 reads as sigma * 0.045, a 22-fold underestimate.
    setup.sigma_y = math.exp(estimate_log_noise_scale(y, INJ, ar_coeffs=PHI_FIT))

    device = resolve_device(DEVICE, batch=1)
    pyro.set_rng_seed(SEED)
    em = AdditiveEmission(sigma_y=setup.sigma_y, ar_coeffs=PHI_FIT)
    em.log_sigma_y.requires_grad_(False)
    model = SingleTraceModel(
        PhasicEventProcess(setup.kernel(), beta0=b0a, beta_inj=0.0, mu_a=setup.mu_a,
                           sigma_a=setup.sigma_a, dt=dt, history=None),
        fit_tonic_baseline(y, INJ, dt=dt), em, tau=setup.tau,
        beta0_prior_loc=b0a, beta_inj_prior_loc=0.0,
        event_guide=WindowedCategoricalEventGuide(
            torch.as_tensor(mf, dtype=torch.float32), window=setup.window,
            thresh=setup.warm["thresh"], gain=setup.warm["gain"]),
        couple_events=False,
        # From the config, so a run that turns the floor on is identified by its config and
        # not by an argument somebody remembered. Absent means 0.0, the objective every
        # finished benchmark used.
        mark_gate_floor=cfg["fit"].get("mark_gate_floor", 0.0),
        **setup.beta_prior_kwargs(),
    )
    if NU is not None:
        fix_marks_at(model.phasic, *noise_unit_marks(setup.sigma_y, *NU))
model.train(False)
model, y = prepare(model, y, device)
pyro.clear_param_store()
pyro.set_rng_seed(SEED)
svi = SVI(model.model, model.guide, ClippedAdam({"lr": cfg["fit"]["lr"]}),
          Trace_ELBO(max_plate_nesting=2))
# Early stopping on the estimand rather than a fixed step count. The detection
# metrics read `q` at whatever step the fit lands on, so the stopping rule is part of
# the benchmark: a run cut short reports a posterior that had not finished moving.
f = cfg["fit"]
plateau = EstimandPlateau(tol=f["stop_tol"], window=f["stop_window"],
                          min_steps=f["stop_min_steps"])
stopped_at = None
for step in range(1, f["max_steps"] + 1):
    svi.step(y, INJ)
    if step % f["stop_probe"] == 0 and plateau.update(
            step, float(model.beta_posterior()["beta_inj"][0])):
        stopped_at = step
        break

q = model.event_posterior(y)[0].detach().cpu().numpy()
bp = model.beta_posterior()
# The fit's own check (2026-10-21): sd(y - E[y]) / sigma_y, one when the residual is the
# emission noise the model claims.
try:
    from photosvi.inference.points.fitquality import residual_ratio
    with torch.no_grad():
        RESID = float(residual_ratio(model, y, INJ)[0])
except Exception:  # noqa: BLE001 -- a diagnostic, never a reason to lose the cell
    RESID = float("nan")
# The tonic (2026-10-27): read after the fit, so it touches neither the simulation's nor
# the fit's random stream and the detection columns reproduce the 10-seed runs exactly.
from detection_tonic import tonic_record  # noqa: E402
with torch.no_grad():
    _tn = model.tonic
    _y = y.detach().cpu().reshape(-1, y.shape[-1])[0].double().numpy()
    _xt = x_true.detach().cpu().reshape(-1)[:_y.size].double().numpy()
    _xf = _tn.baseline(_y.size, INJ).detach().cpu().double().numpy()
    _tt = TonicBaseline(dt=dt, **ton)
    TONIC = tonic_record(
        _y, _xt, _xf,
        true_params=dict(P=float(ton["peak"]), k_rel=float(ton["k_rise"]),
                         k_cl=float(ton["k_decay"]), b=float(ton["b"]),
                         drift=float(ton.get("drift", 0.0)),
                         peak_time_s=float(_tt.peak_time)),
        fit_params=dict(P=float(_tn.peak), k_rel=float(_tn.k_rise), k_cl=float(_tn.k_decay),
                        b=float(_tn.b), drift=float(_tn.drift),
                        peak_time_s=float(_tn.peak_time)),
        inj=INJ, fs=fs)
res = run_benchmark(y.cpu(), onsets, k_fit, inj_t=INJ, fs=fs, tol_s=det["tol_s"],
                    refractory=det["refractory"],
                    extra={"photosvi": q})
out = dict(seed=SEED, mark_mode="noise_units" if NU is not None else "default",
           mu_a_gen=MU_GEN, sigma_a_gen=SIG_GEN,
           mu_a_fit=float(model.phasic.mu_a), sigma_a_fit=float(model.phasic.sigma_a),
           mark_gate_floor=float(cfg["fit"].get("mark_gate_floor", 0.0)),
           sigma_y=SIGMA_Y, sigma_y_est=setup.sigma_y, device=str(device),
           n_true=len(onsets), n_pre=npre, n_post=npost, mle=mle,
           stopped_at=stopped_at, plateau_mean=plateau.plateau_mean,
           beta_inj=bp["beta_inj"][0], beta0=bp["beta0"][0],
           beta_inj_sd=bp["beta_inj"][1], beta0_sd=bp["beta0"][1], residual_ratio=RESID,
           architecture=cfg["fit"].get("architecture", "legacy"),
           config=os.environ.get("BM_CONFIG", "configs/benchmark_guppy.yaml"),
           tonic=TONIC,
           detectors={n: {kk: vv for kk, vv in o.metrics.items() if kk != "perbin_roc"}
                      | {"perbin_roc": o.metrics.get("perbin_roc")}
                      for n, o in res.outcomes.items()})
json.dump(out, open(f"{OUT}/bm_{SIGMA_Y:.2f}_{SEED:02d}.json", "w"))
line = "  ".join(f"{n}: F1={o.metrics.get('best_f1', float('nan')):.3f}"
                 for n, o in res.outcomes.items())
print(f"sigma_y {SIGMA_Y:.2f} s{SEED:02d} | stopped {stopped_at} | {line}")
