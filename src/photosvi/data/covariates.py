"""Drug one-hot covariate `u` for the conditional DMM."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

DRUG_NAMES = ("none", "saline", "morphine", "fentanyl")
DRUG_TO_IDX = {d: i for i, d in enumerate(DRUG_NAMES)}


def build_vector_u(
    labels: pd.DataFrame,
    B: int,
    T: int,
    inj_t: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the per-animal, per-time drug one-hot `u` of shape [B, T, 4].

    Before `inj_t` every animal is labelled 'none'; from `inj_t` onward each
    animal carries its own DRUG label.
    """
    u = torch.zeros(B, T, len(DRUG_NAMES), dtype=torch.float32, device=device)
    u[:, :inj_t, DRUG_TO_IDX["none"]] = 1.0
    for i in range(B):
        d = str(labels.loc[i, "DRUG"]).lower()
        u[i, inj_t:, DRUG_TO_IDX[d]] = 1.0
    return u


def build_u_auto_from_labels(
    labels: pd.DataFrame,
    n_samples: int,
    T: int,
    inj_t: int,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Sample drug identities from the empirical distribution and build `u`.

    Used by `photosvi-generate --cond-from-labels` to synthesize drug-conditioned
    trajectories whose label mix matches the training set.

    Returns (u, sampled_drugs, probs) with `u` of shape [n_samples, T, 4],
    `sampled_drugs` an array of drug-name strings of length n_samples,
    `probs` the (saline, morphine, fentanyl) empirical probabilities.
    """
    rng = np.random.default_rng(seed)
    counts = labels["DRUG"].astype(str).str.lower().value_counts()
    drugs = ["saline", "morphine", "fentanyl"]
    probs = np.array([counts.get(d, 0) for d in drugs], dtype=float)
    probs = probs / probs.sum()
    sampled = rng.choice(drugs, size=n_samples, p=probs)

    u = torch.zeros(n_samples, T, 4, dtype=torch.float32, device=device)
    u[:, :inj_t, DRUG_TO_IDX["none"]] = 1.0
    for i, d in enumerate(sampled):
        u[i, inj_t:, DRUG_TO_IDX[d]] = 1.0
    return u, sampled, probs
