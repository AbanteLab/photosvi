"""Fit a cohort and read a drug contrast off its posterior.

Runnable as-is and executed by the test suite. Two animals, three drugs, two minutes
per recording -- small enough to finish in seconds, and far too small to say anything
about a drug. The point is the shape of the call, not the answer.
"""

import pyro
import torch
import yaml
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from photosvi.api import config_path
from photosvi.benchmark import matched_filter_score
from photosvi.experiments.cohort import (
    DRUG_NAMES,
    OPIOID1,
    OPIOID2,
    simulate_cohort,
    spec_from_config,
)
from photosvi.inference.points import prepare, resolve_device
from photosvi.models.points import (
    AdditiveEmission,
    HierarchicalModel,
    PhasicEventProcess,
    WindowedCategoricalEventGuide,
    build_kernel,
    fit_tonic_baseline,
    stack_tonic_baselines,
)

cfg = yaml.safe_load(config_path("cohort_comparison.yaml").read_text())
cfg["cohort"].update(n_animals=2, T=3000, inj_t=750, inj_jitter=50)
spec = spec_from_config(cfg, delta=0.40)          # opioid2 is 0.40 above opioid1

# --- 1. a cohort. Replace with your own traces plus the index vectors describing
#        which animal and which drug each recording belongs to.
coh = simulate_cohort(spec, seed=0)
Y, animal, drug, inj = coh["y"], coh["animal"], coh["drug"], coh["inj_t"]
S, dt = spec.n_sessions, spec.dt

# --- 2. per-session anchors. Each recording gets its own matched filter and its own
#        tonic pre-fit: a cohort is S different recordings, not one repeated.
kernel = build_kernel(dict(spec.kernel), spec.kernel_length, dt)
mf = torch.stack([
    torch.as_tensor(matched_filter_score(Y[s:s + 1], kernel, inj[s], dt=dt),
                    dtype=torch.float32).reshape(-1) for s in range(S)])
tonic = stack_tonic_baselines(
    [fit_tonic_baseline(Y[s:s + 1], inj[s], dt=dt) for s in range(S)])

emission = AdditiveEmission(sigma_y=spec.sigma_y)
emission.log_sigma_y.requires_grad_(False)
model = HierarchicalModel(
    PhasicEventProcess(kernel, beta0=spec.beta0_mean, beta_inj=0.0, mu_a=spec.mu_a,
                       sigma_a=spec.sigma_a, dt=dt, history=None),
    tonic, emission,
    animal=animal, drug=drug, inj_t=inj, beta0_anchor=spec.beta0_mean,
    event_guide=WindowedCategoricalEventGuide(mf, window=10),
)
model.train(False)

# --- 3. the model reads the design off the indices and reports what it can identify.
design = model.design_summary()
print("design:", {k: design[k] for k in
                  ("n_sessions", "n_animals", "n_drugs", "levels", "within_animal")})

# --- 4. fit. `resolve_device` puts a cohort on the GPU when there is one, because
#        the per-step cost there is flat in batch size.
device = resolve_device("auto", batch=S)
model, Y = prepare(model, Y, device)
pyro.clear_param_store()
pyro.set_rng_seed(0)
svi = SVI(model.model, model.guide, ClippedAdam({"lr": 0.01}),
          Trace_ELBO(max_plate_nesting=2))
for _ in range(400):
    svi.step(Y)

# --- 5. the estimand is a contrast, not a level.
loc, scale = model.drug_contrast(OPIOID2, OPIOID1)
print(f"{DRUG_NAMES[OPIOID2]} - {DRUG_NAMES[OPIOID1]}: {loc:+.3f} +- {scale:.3f} "
      f"(simulated {spec.true_contrast():+.3f})")
print("variance components:",
      {k: round(v, 3) for k, v in model.variance_components().items()})

assert set(model.drug_posterior()) == set(range(spec.n_drugs))
