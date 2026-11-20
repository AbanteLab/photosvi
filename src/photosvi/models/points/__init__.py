"""Points model family — sparse marked point process for phasic dopamine.

Third model family alongside the DMM and HMM. Where those carry the signal in
a continuous latent or discrete regimes, the points family models the *phasic*
component as an explicit sparse event train convolved with a fixed sensor
kernel (`events.py`), plus a smooth *tonic* baseline x_hat(t) (`tonic.py`) it
rides on top of. Together with the additive-mean emission (`emission.py`) they
assemble the generative trace y ~ Normal/StudentT(x_hat + c, sigma_y), fit by
relaxed-discrete SVI (`SingleTraceModel`).
"""

from photosvi.models.points.emission import (
    AdditiveEmission,
    estimate_ar_coeffs,
    estimate_log_noise_scale,
)
from photosvi.models.points.event_guides import (
    FreeEventGuide,
    MatchedFilterEventGuide,
    WarmFreeEventGuide,
    WindowedCategoricalEventGuide,
)
from photosvi.models.points.events import (
    PhasicEventProcess,
    bi_exponential_kernel,
    bigamma_kernel,
    biphasic_kernel,
    build_kernel,
    estimate_log_baseline_rate,
    gamma_lobe,
)
from photosvi.models.points.hierarchical import HierarchicalModel
from photosvi.models.points.points import SingleTraceModel
from photosvi.models.points.tonic import (
    BatchedTonicBaseline,
    TonicBaseline,
    fit_tonic_baseline,
    stack_tonic_baselines,
)

__all__ = [
    "AdditiveEmission",
    "BatchedTonicBaseline",
    "FreeEventGuide",
    "HierarchicalModel",
    "MatchedFilterEventGuide",
    "WindowedCategoricalEventGuide",
    "PhasicEventProcess",
    "SingleTraceModel",
    "TonicBaseline",
    "WarmFreeEventGuide",
    "bi_exponential_kernel",
    "biphasic_kernel",
    "bigamma_kernel",
    "gamma_lobe",
    "build_kernel",
    "estimate_log_baseline_rate",
    "estimate_ar_coeffs",
    "estimate_log_noise_scale",
    "fit_tonic_baseline",
    "stack_tonic_baselines",
]
