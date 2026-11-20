"""One coverage replicate: simulate at a known beta_inj, fit the reference model,
and record a properly transformed credible interval.

    mkdir -p runs_coverage
    for g in 0.00 1.02 2.00; do for s in $(seq 0 29); do echo "$s $g"; done; done \
      | xargs -P 24 -n 2 sh -c 'OMP_NUM_THREADS=1 python scripts/coverage_replicate.py $0 $1'

The interval is built from the transformed quantiles rather than loc +- z*scale:
under the uniform prior the posterior is a Normal on an unconstrained axis pushed
through sigmoid+affine, and the delta-method scale that `beta_posterior` returns is
a local linearisation that gets the tails wrong.
"""
import sys, json, math, yaml, torch, pyro, numpy as np
from scipy.stats import norm
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from photosvi.benchmark import matched_filter_score
from photosvi.experiments.sweep import setup_from_config, mf_anchor, gen_spec
from photosvi.inference.points import EstimandPlateau
from photosvi.models.points import (build_kernel, TonicBaseline, PhasicEventProcess,
    AdditiveEmission, SingleTraceModel, WindowedCategoricalEventGuide, fit_tonic_baseline)

SEED, GT_BI = int(sys.argv[1]), float(sys.argv[2])
cfg = yaml.safe_load(open("configs/points_guide_sweep_k5.yaml"))
d,k,gt,ton,det = cfg["data"],cfg["kernel"],cfg["gt"],cfg["tonic"],cfg["detection"]
dt,T,INJ = 1/d["fs"], d["T"], d["inj_t"]
gt = dict(gt, beta_inj=GT_BI)

g = torch.Generator().manual_seed(SEED)
x_true = TonicBaseline(dt=dt, **ton).baseline(T, INJ)
ex = PhasicEventProcess(build_kernel(gen_spec(k), k["length"], dt), beta0=gt["beta0"],
    beta_inj=gt["beta_inj"], mu_a=gt["mu_a"], sigma_a=gt["sigma_a"], dt=dt, history=None).sample(T,INJ,1,g)
y = AdditiveEmission(sigma_y=gt["sigma_y"]).sample(x_true + ex["c"], generator=g)["y"]
ons = ex["s"][0].nonzero().reshape(-1).numpy()
npre, npost = int((ons < INJ).sum()), int((ons >= INJ).sum())
mle = math.log((npost/((T-INJ)*dt))/(npre/(INJ*dt))) if npre and npost else float("nan")

mf = matched_filter_score(y, build_kernel(dict(k["fit"]), k["length"], dt), INJ, dt=dt)
b0a,_ = mf_anchor(mf, INJ, dt, thresh=det["mf_thresh"], refractory=det["refractory"])
setup = setup_from_config(cfg, b0a)
torch.set_num_threads(1); pyro.set_rng_seed(SEED)
em = AdditiveEmission(sigma_y=setup.sigma_y); em.log_sigma_y.requires_grad_(False)
m = SingleTraceModel(
    PhasicEventProcess(setup.kernel(), beta0=b0a, beta_inj=0.0, mu_a=setup.mu_a,
                       sigma_a=setup.sigma_a, dt=dt, history=None),
    fit_tonic_baseline(y, INJ, dt=dt), em, tau=setup.tau,
    beta0_prior_loc=b0a, beta_inj_prior_loc=0.0,
    event_guide=WindowedCategoricalEventGuide(torch.as_tensor(mf, dtype=torch.float32),
        window=setup.window, thresh=setup.warm["thresh"], gain=setup.warm["gain"]),
    couple_events=False, **setup.beta_prior_kwargs())
m.train(False)
pyro.clear_param_store(); pyro.set_rng_seed(SEED)
svi = SVI(m.model, m.guide, ClippedAdam({"lr": cfg["fit"]["lr"]}), Trace_ELBO(max_plate_nesting=2))
# Early stopping on the estimand. The reported value is the plateau mean, not the
# last probe: once the iterate is orbiting its answer the final draw is one sample
# from the orbit.
fit = cfg["fit"]
plateau = EstimandPlateau(tol=fit["stop_tol"], window=fit["stop_window"],
                          min_steps=fit["stop_min_steps"])
probe, stopped_at = fit["stop_probe"], None
for step in range(1, fit["max_steps"] + 1):
    svi.step(y, INJ)
    if step % probe == 0:
        if plateau.update(step, float(m.beta_posterior()["beta_inj"][0])):
            stopped_at = step
            break

# The posterior is a Normal on an unconstrained axis pushed through sigmoid+affine,
# so the credible interval must be the transformed quantiles, NOT loc +- z*scale:
# the delta-method scale is a local linearisation and gets the tails wrong.
z_loc = float(pyro.param("q_beta_inj_loc").detach())
z_sc = float(pyro.param("q_beta_inj_scale").detach())
lo, hi = m.beta_inj_low, m.beta_inj_high
xf = lambda z: lo + (hi - lo) / (1.0 + math.exp(-z))
q = {p: xf(z_loc + z_sc * norm.ppf(p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
bp = m.beta_posterior()
json.dump(dict(seed=SEED, gt_bi=GT_BI, mle=mle, n_pre=npre, n_post=npost,
               stopped_at=stopped_at, plateau_mean=plateau.plateau_mean,
               fit=bp["beta_inj"][0], delta_scale=bp["beta_inj"][1],
               q05=q[0.05], q25=q[0.25], q50=q[0.5], q75=q[0.75], q95=q[0.95],
               b0=bp["beta0"][0]),
          open(f"runs_coverage_v2/cov_{GT_BI:.2f}_{SEED:03d}.json","w"))
print(f"gt {GT_BI:+.2f} s{SEED:03d} | plateau {plateau.plateau_mean:+.3f} | last {bp['beta_inj'][0]:+.3f} "
      f"| stopped {stopped_at} | MLE {mle:+.3f}")
