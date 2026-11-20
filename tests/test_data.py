"""Tests for the data helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from photosvi.data.covariates import build_u_auto_from_labels, build_vector_u
from photosvi.data.loader import TRANSIENT_BAND_HZ, filter_signal
from photosvi.data.windows import build_windows


def test_filter_signal_smoke():
    x = np.random.RandomState(0).randn(2000)
    with pytest.warns(RuntimeWarning):          # 1 Hz is inside the transient band
        y = filter_signal(x, cutoff_hz=1.0, order=8, fs=25.0)
    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_filtering_into_the_transient_band_warns():
    """The failure it warns about is silent: the fit succeeds and finds nothing.

    This module used to low-pass at 1 Hz by DEFAULT, inherited from the state-space
    era. Under the point process that deletes the latent being estimated.
    """
    x = np.random.RandomState(0).randn(1000)
    with pytest.warns(RuntimeWarning, match="removes the transient band"):
        filter_signal(x, cutoff_hz=TRANSIENT_BAND_HZ[0], fs=25.0)


def test_filtering_above_the_transient_band_is_quiet():
    x = np.random.RandomState(0).randn(1000)
    warnings.simplefilter("error")               # any warning fails this test
    try:
        filter_signal(x, cutoff_hz=11.0, fs=25.0)
    finally:
        warnings.resetwarnings()


def test_build_windows_shape():
    B, T, D = 2, 30, 1
    x = torch.arange(B * T * D, dtype=torch.float32).reshape(B, T, D)
    out = build_windows(x, d=4, stride=1)
    # windows end at t = 4, 5, ..., T-1  -> T - 4 windows
    assert out.shape == (B, T - 4, 5, D)
    # last window ends at the last frame
    assert torch.equal(out[:, -1, -1, :], x[:, -1, :])


def test_build_vector_u():
    labels = pd.DataFrame({"DRUG": ["saline", "morphine", "fentanyl"]})
    u = build_vector_u(labels=labels, B=3, T=10, inj_t=4)
    assert u.shape == (3, 10, 4)
    # before injection: all "none" (index 0)
    assert u[:, :4, 0].sum().item() == 3 * 4
    # after injection: identity drug
    assert u[0, 5, 1].item() == 1.0  # saline
    assert u[1, 5, 2].item() == 1.0  # morphine
    assert u[2, 5, 3].item() == 1.0  # fentanyl


def test_build_u_auto_from_labels():
    labels = pd.DataFrame({"DRUG": ["saline"] * 5 + ["morphine"] * 3 + ["fentanyl"] * 2})
    u, sampled, probs = build_u_auto_from_labels(
        labels=labels, n_samples=100, T=8, inj_t=3, seed=0
    )
    assert u.shape == (100, 8, 4)
    assert sampled.shape == (100,)
    np.testing.assert_allclose(probs, [0.5, 0.3, 0.2], atol=1e-6)
    # pre-injection slot is always 'none'
    assert torch.all(u[:, :3, 0] == 1.0)
