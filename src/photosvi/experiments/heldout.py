"""The held-out probe: freeze the globals, refit the locals, score unseen data.

Ordinary train/test scoring assumes everything learned is global -- fit coefficients,
apply them to a new row. This model is not like that. It has

  * **globals** -- beta0, beta_inj, sigma_y, the mark hyperparameters. Learned once,
    shared by the whole trace. These are the hypothesis.
  * **locals** -- the per-bin q(s) and marks. One per bin. These belong to the data.

So a fitted model handed an unseen trace cannot score it: it has no idea where the
events are there. The free per-bin family is the extreme case -- its logits on unseen
bins were simply never trained, so they sit at the prior and score badly for reasons
that have nothing to do with whether beta_inj is any good.

The fix is the standard one (the "fold-in" of held-out perplexity in topic models):
freeze the globals, infer the locals on the held-out trace under a fixed budget, then
score. Step 3 looks like cheating and is not -- locating events in unseen data is
*inference*, not *learning*, and the shared globals never budge. Compare a hierarchical
model: predicting for a new subject requires estimating that subject's random effect
from their own data, and that is not leakage.

Two design commitments make the comparison fair across arms:

  * the local family is **common** (plain free per-bin, matched-filter warm start),
    not the arm's own guide -- so the score reflects the globals handed over rather
    than the arm's guide expressiveness;
  * the budget is **fixed and identical**. Cutting it is the one economy that could
    manufacture the result, since an under-converged refit depresses the score in
    exactly the shape we are looking for.

What comes back is an ELBO -- a *bound*, not the likelihood. Bound tightness can differ,
so cross-arm levels are soft and the shape over training within an arm is the solid part.
"""

from __future__ import annotations

import numpy as np
import pyro
import torch
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from photosvi.experiments.arms import PointsSetup, frozen_local_model
from photosvi.models.points import SingleTraceModel, TonicBaseline


def warm_logits(
    mf: np.ndarray | torch.Tensor,
    *,
    thresh: float = 2.5,
    gain: float = 2.0,
    hi: float = 6.0,
    floor: float = -6.0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """The classical detector init used to start the probe's locals: clamp(g(m - t))."""
    m = torch.as_tensor(mf, dtype=dtype or torch.get_default_dtype())
    if m.dim() == 2:
        m = m[0]
    return (gain * (m - thresh)).clamp(floor, hi)


@torch.no_grad()
def recon_ll(model: SingleTraceModel, y: torch.Tensor, inj_t: int) -> float:
    """Per-element reconstruction LL at the posterior-mean drive (events x marks)."""
    x_hat = model.tonic.baseline(y.shape[-1], inj_t)
    u = model.event_posterior(y) * pyro.param("q_a_loc").exp()
    mean = x_hat.unsqueeze(0) + model.phasic.convolve(u)
    return float(model.emission.log_prob(y, mean).sum()) / y.numel()


@torch.no_grad()
def null_recon_ll(
    y: torch.Tensor, inj_t: int, setup: PointsSetup, tonic: TonicBaseline
) -> float:
    """The same trace scored with **zero drive** -- tonic only.

    The baseline every held-out number is reported against. Absolute LL here is
    dominated by the tonic and the noise floor; the events move the fourth decimal,
    so the raw curve is unreadable and the delta is the part that depends on the betas.
    """
    m = frozen_local_model(0.0, inj_t, setup, tonic)
    mean = m.tonic.baseline(y.shape[-1], inj_t).unsqueeze(0).expand_as(y)
    return float(m.emission.log_prob(y, mean).sum()) / y.numel()


def heldout_probe(
    beta0: float,
    beta_inj: float,
    y_ho: torch.Tensor,
    inj_t: int,
    setup: PointsSetup,
    tonic: TonicBaseline,
    *,
    mf_ho: np.ndarray | torch.Tensor,
    steps: int = 2000,
    lr: float = 5e-3,
    lrd: float = 0.9998,
    seed: int = 0,
) -> dict[str, float]:
    """Score one (beta0, beta_inj) pair on held-out data. Returns {"recon", "elbo"}.

    `poutine.condition` pins the two betas in the model and `poutine.block` hides them
    from the guide, so SVI can only move the locals -- nothing the held-out trace says
    can reach a global.

    The caller's parameter store is saved and restored, so this is safe to call at any
    point. Note the restore is *necessary but not sufficient* for calling it mid-fit:
    clearing the store recreates the param tensors, which orphans a live optimizer's
    Adam state. Drive it as a replay over recorded betas instead (`sweep.py` does).
    """
    saved = pyro.get_param_store().get_state()
    try:
        pyro.clear_param_store()
        m = frozen_local_model(beta0, inj_t, setup, tonic)
        m.guide(y_ho, inj_t)                                  # materialize the locals
        pyro.param("q_s_logits").data.copy_(
            warm_logits(mf_ho, dtype=y_ho.dtype, **setup.warm).expand_as(
                pyro.param("q_s_logits")
            )
        )
        cond = poutine.condition(
            m.model,
            data={
                "beta0": torch.as_tensor(float(beta0), dtype=y_ho.dtype),
                "beta_inj": torch.as_tensor(float(beta_inj), dtype=y_ho.dtype),
            },
        )
        guide = poutine.block(m.guide, hide=["beta0", "beta_inj"])
        pyro.set_rng_seed(seed)
        svi = SVI(cond, guide, ClippedAdam({"lr": lr, "lrd": lrd}),
                  Trace_ELBO(max_plate_nesting=2))
        best = float("inf")
        for _ in range(steps):
            best = min(best, svi.step(y_ho, inj_t) / y_ho.numel())
        return {"recon": recon_ll(m, y_ho, inj_t), "elbo": -best}
    finally:
        pyro.clear_param_store()
        pyro.get_param_store().set_state(saved)
