"""The arm registry: names for points of the (guide family x coupling) product.

An "arm" here is one variational family over the *same* generative points model.
Two axes generate all of them, and keeping them as axes rather than as a list of
subclasses is the point of this module:

  * **guide family** -- which event-logit block is installed (`event_guides.py`):
    `free` (per-bin, prior-initialized), `warm` (per-bin, matched-filter
    initialized), `mf` (amortized, two scalars on the filter).
  * **coupling** -- whether `SingleTraceModel.guide` rebuilds its event anchor at the
    betas sampled in the guide (`couple_events=True`) or at a detached constant.

`free` vs `mf` alone confounds family with initialization, because the amortized
block *is* the matched filter. `warm` exists so the two contrasts separate:
`free -> warm` is initialization at fixed family, `warm -> mf` is family at fixed
initialization.

Adding an arm is a dict entry, not a class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyro
import torch

from photosvi.models.points import (
    AdditiveEmission,
    FreeEventGuide,
    MatchedFilterEventGuide,
    PhasicEventProcess,
    SingleTraceModel,
    TonicBaseline,
    WarmFreeEventGuide,
    WindowedCategoricalEventGuide,
    build_kernel,
    fit_tonic_baseline,
)

GUIDE_FAMILIES = ("free", "warm", "mf", "cat")


@dataclass(frozen=True)
class ArmSpec:
    """One arm: which event-guide block, and whether the rate is coupled into it."""

    guide: str
    couple: bool
    label: str

    def __post_init__(self):
        if self.guide not in GUIDE_FAMILIES:
            raise ValueError(f"guide must be one of {GUIDE_FAMILIES} (got {self.guide!r})")


#: The frozen production configuration. Everything else in ARMS is either the
#: status quo it replaced or a control that isolates one axis of the comparison.
#: Frozen 2026-09-08 on synthetic recordings; revisit if the real data behaves
#: differently, which is what the pluggable guide seam exists for.
REFERENCE_ARM = "cat"

ARMS: dict[str, ArmSpec] = {
    "free": ArmSpec("free", False, "free per-bin (cold)"),
    "free-warm": ArmSpec("warm", False, "free per-bin (MF-warm)"),
    "amort": ArmSpec("mf", False, "amortized (MF)"),
    "free+c": ArmSpec("free", True, "free per-bin + coupling"),
    "amort+c": ArmSpec("mf", True, "amortized + coupling"),
    "cat": ArmSpec("cat", False, "windowed categorical (MF-warm)"),   # REFERENCE_ARM
    "cat+c": ArmSpec("cat", True, "windowed categorical + coupling"),
}


@dataclass
class PointsSetup:
    """Everything shared by every arm: the model hyperparameters and the fit kernel.

    Held apart from `ArmSpec` on purpose -- these must be *identical* across arms or
    the comparison measures the setup rather than the guide. `sigma_y` is pinned
    (not fitted) so nothing hides in it, and the tonic is fit once per trace from
    the same routine for every arm.
    """

    dt: float
    kernel_length: int
    kernel_spec: dict           # {"family": "biexp"|"biphasic", ...}; see build_kernel
    mu_a: float
    sigma_a: float
    sigma_y: float
    beta0_prior_loc: float
    # Lag-one correlation of the emission residual, ASSERTED like sigma_y and the
    # kernel rather than estimated. Zero, the default, is the white emission every
    # config before 2026-09-30 used and reproduces those results exactly. It exists
    # because the real recordings are not white: see
    ar_coeffs: tuple | list | None = None
    beta0_prior_scale: float = 2.0
    beta_inj_prior_loc: float = 0.0
    beta_inj_prior_scale: float = 1.0
    tau: float = 0.5
    warm: dict = field(default_factory=dict)   # thresh/gain/hi/floor for the warm block
    window: int = 10           # bins per window for the categorical block (~0.2 s at 48 Hz)
    beta_prior: str = "normal"     # "normal" | "uniform"
    # Floor on the gated mark divergence. Zero is the objective everything in the paper was
    # fitted with; see photosvi.models.points.points._MARK_GATE_FLOOR for what it guards
    # against and why it is not on by default.
    mark_gate_floor: float = 0.0
    # The joint block over (beta0, beta_inj). Their posterior correlation is -0.88 --
    # a higher baseline with a smaller injection effect explains the same
    # post-injection rate -- and a factorised guide must call it zero. Measured
    # against HMC on the same recordings, factorised is 2.2x narrower than exact and
    # the block brings it to 0.97x.
    beta_guide: str = "fullrank"   # "fullrank" | "meanfield"
    beta0_halfwidth: float = 5.0   # uniform bounds: anchor +/- this, on the log-rate axis
    beta_inj_halfwidth: float = 4.0  # uniform bounds: 0 +/- this, on the log-ratio axis

    def beta_prior_kwargs(self) -> dict:
        """Prior and variational-family arguments for `SingleTraceModel`.

        The default is an unbounded Normal. The bounded Uniform is what forced the
        guide onto a transformed axis, and the scale it reported back was a
        delta-method linearisation: measured on the same recordings it read 0.1525
        where the unbounded fits read 0.1711, a 12% artificial narrowing of the
        interval. Shrinkage from the Normal's widths is 0.77% of the estimate,
        matching the analytic prediction, so it is quantified rather than argued away.


        The bounds are derived, not written down: beta0 is a log-rate whose plausible
        range sits around the matched filter's estimate for *this* trace, and hard
        numbers in a config would silently put a real animal's baseline outside the
        support -- where the Uniform scores -inf and the fit fails without saying so.
        beta_inj is a log-ratio and stays centred on zero.
        """
        if self.beta_prior == "normal":
            return dict(
                beta_prior="normal",
                beta_guide=self.beta_guide,
                beta0_prior_scale=self.beta0_prior_scale,
                beta_inj_prior_scale=self.beta_inj_prior_scale,
            )
        return dict(
            beta_prior="uniform",
            beta_guide=self.beta_guide,
            beta_low=self.beta0_prior_loc - self.beta0_halfwidth,
            beta_high=self.beta0_prior_loc + self.beta0_halfwidth,
            beta_inj_low=-self.beta_inj_halfwidth,
            beta_inj_high=self.beta_inj_halfwidth,
        )

    def kernel(self) -> torch.Tensor:
        return build_kernel(self.kernel_spec, self.kernel_length, self.dt)


def build_arm(
    name: str,
    y: torch.Tensor,
    inj_t: int,
    setup: PointsSetup,
    *,
    seed: int,
    mf: np.ndarray | torch.Tensor | None = None,
) -> SingleTraceModel:
    """Construct one arm on one trace.

    The RNG is seeded **before** construction, not after: an event-guide block that
    draws weights at init would otherwise land in an uncontrolled basin, and a fit
    that looks good for that reason is a lottery win, not a result. Neither block
    shipped here draws at init, but the ordering is the contract.

    `mf` must be the matched filter of *this* `y`, precomputed by the caller
    (`benchmark.matched_filter_score`) -- `event_guides` cannot compute it without
    closing an import cycle. Required for the `warm` and `mf` families.
    """
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {sorted(ARMS)}")
    spec = ARMS[name]

    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)

    if spec.guide in ("warm", "mf", "cat") and mf is None:
        raise ValueError(f"arm {name!r} needs the matched-filter score; pass mf=")

    if spec.guide == "free":
        event_guide = FreeEventGuide()
    elif spec.guide == "cat":
        event_guide = WindowedCategoricalEventGuide(
            torch.as_tensor(mf, dtype=torch.float32),
            window=setup.window,
            thresh=setup.warm.get("thresh", 2.5),
            gain=setup.warm.get("gain", 2.0),
        )
    elif spec.guide == "warm":
        event_guide = WarmFreeEventGuide(torch.as_tensor(mf, dtype=torch.float32), **setup.warm)
    else:
        # coupled: the live prior anchor already supplies the sparse baseline, so the
        # block starts at zero offset instead of the sparse -5 it needs when decoupled.
        event_guide = MatchedFilterEventGuide(
            torch.as_tensor(mf, dtype=torch.float32),
            bias_init=0.0 if spec.couple else -5.0,
        )

    phasic = PhasicEventProcess(
        setup.kernel(), beta0=setup.beta0_prior_loc, beta_inj=0.0,
        mu_a=setup.mu_a, sigma_a=setup.sigma_a, dt=setup.dt, history=None,
    )
    emission = AdditiveEmission(sigma_y=setup.sigma_y, ar_coeffs=setup.ar_coeffs)
    emission.log_sigma_y.requires_grad_(False)          # pinned: nothing hides in sigma_y

    model = SingleTraceModel(
        phasic,
        fit_tonic_baseline(y, inj_t, dt=setup.dt),      # same routine for every arm
        emission,
        tau=setup.tau,
        beta0_prior_loc=setup.beta0_prior_loc,
        beta_inj_prior_loc=setup.beta_inj_prior_loc,
        **setup.beta_prior_kwargs(),
        event_guide=event_guide,
        couple_events=spec.couple,
    )
    model.train(False)          # sigma_y is pinned; silences the param-store warning
    return model


def frozen_local_model(
    beta0: float, inj_t: int, setup: PointsSetup, tonic: TonicBaseline
) -> SingleTraceModel:
    """A SingleTraceModel with every module parameter frozen -- only local q(s), q(a) may move.

    The scoring model for the held-out probe. It always installs the plain free
    per-bin block regardless of which arm produced the betas: the probe is meant to
    measure the *globals* handed over, so it must not vary with the arm's guide.
    """
    phasic = PhasicEventProcess(
        setup.kernel(), beta0=beta0, beta_inj=0.0,
        mu_a=setup.mu_a, sigma_a=setup.sigma_a, dt=setup.dt, history=None,
    )
    m = SingleTraceModel(
        phasic, tonic, AdditiveEmission(sigma_y=setup.sigma_y, ar_coeffs=setup.ar_coeffs),
        beta0_prior_loc=beta0, beta_inj_prior_loc=0.0,
    )
    for p in m.parameters():
        p.requires_grad_(False)
    m.train(False)
    return m
