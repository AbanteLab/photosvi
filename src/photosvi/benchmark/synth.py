"""Synthetic traces with known onsets -- the only way to score detection quantitatively.

Real traces carry no labelled events, so recall/precision are undefined on them. We
generate from the points model's own forward process, keeping the onset indices as the
answer key. The generating kernel is passed in: use a *faster* kernel than the detector's
to keep a deliberate shape mismatch (a model-matched kernel flatters whichever detector
shares it). Feed real per-treatment fitted params (beta0, beta_inj, sigma_y, marks) to
make the synthetic regime real-grounded rather than illustrative.
"""

from __future__ import annotations

import numpy as np
import torch

from photosvi.data.loader import DEFAULT_FS
from photosvi.models.points import PhasicEventProcess


def make_synth(
    kernel: torch.Tensor,
    *,
    T: int = 30_000,
    inj_t: int = 15_000,
    beta0: float = -2.0,
    beta_inj: float = 0.7,
    mu_a: float = 0.0,
    sigma_a: float = 0.25,
    sigma_y: float = 0.4,
    dt: float = 1.0 / DEFAULT_FS,
    seed: int = 0,
) -> tuple[torch.Tensor, np.ndarray]:
    """Known-onset synthetic trace from the phasic forward process + additive Gaussian noise.

    Returns (y, onsets): y is a torch tensor [1, T]; onsets is a numpy array of event bins.
    """
    g = torch.Generator().manual_seed(int(seed))
    phasic = PhasicEventProcess(
        kernel, beta0=beta0, beta_inj=beta_inj, mu_a=mu_a, sigma_a=sigma_a, dt=dt
    )
    ex = phasic.sample(T, inj_t, batch=1, generator=g)
    y = ex["c"] + sigma_y * torch.randn(1, T, generator=g)
    onsets = ex["s"][0].nonzero().reshape(-1).numpy()
    return y, onsets
