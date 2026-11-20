"""The single-trace model as the paper fits it, built in one place.

The detection benchmark, the two-stage arm of the cohort benchmark and `photosvi.api`
used to construct `SingleTraceModel` each in their own way, and they drifted: different
prior scales on the coefficients (3.0 / 2.0 on beta0, 2.0 / 1.0 on beta_inj), a gate floor
of 0 in one and 0.1 in another, the tonic frozen in one and refined in the others. The
paper's final model (2026-10-21) is ONE architecture, and a config carrying
`fit.architecture: paper_final` builds it through this function wherever it is fitted.
Configs without that key keep the code paths they were run with, so earlier results
reproduce.

Everything the model needs comes from the config's `fit`, `warm` and `detection` blocks;
the only quantities read off the trace are the matched-filter anchor of beta0 and, unless
given, the noise scale sigma_y (successive differences, corrected for the AR(p)
correlation). Under the noise-unit mark mode the marks are fixed at a_t = sigma_y r_t.
"""
from __future__ import annotations

import math

import torch

FINAL = "paper_final"


def is_final(cfg: dict | None) -> bool:
    """Whether a config asks for the paper's final single architecture."""
    return ((cfg or {}).get("fit") or {}).get("architecture") == FINAL


def build_single_trace(y: torch.Tensor, inj: int, *, kernel: torch.Tensor, dt: float,
                       fit: dict, warm: dict, detection: dict, ar_coeffs=None,
                       sigma_y: float | None = None, mu_a: float | None = None,
                       sigma_a: float | None = None):
    """`(model, info)` for one recording `y` ([1, T]) with the injection at bin `inj`.

    `info` carries what was read off the trace: the matched-filter score `mf`, the beta0
    anchor `b0a`, `sigma_y` and the mark distribution `(mu_a, sigma_a)` the fit holds.
    `mu_a`/`sigma_a` are only used when the noise-unit mode is off (old configs).
    """
    from photosvi.benchmark import matched_filter_score
    from photosvi.experiments.marks import fix_marks_at, noise_unit_marks, noise_unit_mode
    from photosvi.experiments.sweep import mf_anchor
    from photosvi.models.points import (
        AdditiveEmission,
        PhasicEventProcess,
        SingleTraceModel,
        WindowedCategoricalEventGuide,
        estimate_log_noise_scale,
        fit_tonic_baseline,
    )

    phi = list(ar_coeffs) if ar_coeffs else None
    mf = matched_filter_score(y, kernel, inj, dt=dt)
    b0a, _ = mf_anchor(mf, inj, dt, thresh=detection["mf_thresh"],
                       refractory=detection["refractory"])
    if sigma_y is None:
        sigma_y = math.exp(estimate_log_noise_scale(y, inj, ar_coeffs=phi))
    nu = noise_unit_mode(fit)
    if nu is not None:
        mu_a, sigma_a = noise_unit_marks(sigma_y, *nu)
    elif mu_a is None or sigma_a is None:
        raise ValueError("without the noise-unit mark mode, mu_a and sigma_a must be given")

    em = AdditiveEmission(sigma_y=sigma_y, ar_coeffs=phi)
    em.log_sigma_y.requires_grad_(False)
    prior = dict(beta_prior=fit["beta_prior"], beta_guide=fit["beta_guide"],
                 beta0_prior_scale=fit["beta0_prior_scale"],
                 beta_inj_prior_scale=fit["beta_inj_prior_scale"])
    if fit["beta_prior"] == "uniform":
        prior.update(beta_low=b0a - fit["beta0_halfwidth"], beta_high=b0a + fit["beta0_halfwidth"],
                     beta_inj_low=-fit["beta_inj_halfwidth"],
                     beta_inj_high=fit["beta_inj_halfwidth"])
    model = SingleTraceModel(
        PhasicEventProcess(kernel, beta0=b0a, beta_inj=0.0, mu_a=mu_a, sigma_a=sigma_a,
                           dt=dt, history=None),
        # Refined jointly with the rest of the model as point estimates (Methods), not frozen.
        fit_tonic_baseline(y, inj, dt=dt), em, tau=fit["tau"],
        beta0_prior_loc=b0a, beta_inj_prior_loc=float(fit.get("beta_inj_prior_loc", 0.0)),
        event_guide=WindowedCategoricalEventGuide(
            torch.as_tensor(mf, dtype=torch.float32), window=int(fit["window"]),
            thresh=warm["thresh"], gain=warm["gain"]),
        couple_events=False, analytic_s_kl=True, mark_kl="gated",
        mark_gate_floor=float(fit["mark_gate_floor"]),
        **prior,
    )
    if nu is not None:
        fix_marks_at(model.phasic, mu_a, sigma_a)
    model.train(False)
    return model, dict(mf=mf, b0a=float(b0a), sigma_y=float(sigma_y), mu_a=float(mu_a),
                       sigma_a=float(sigma_a))
