"""Device selection: the rule, and the refusal to fall back silently."""

from __future__ import annotations

import pytest
import torch

from photosvi.inference.points import prepare, resolve_device
from photosvi.models.points import (
    AdditiveEmission,
    PhasicEventProcess,
    SingleTraceModel,
    TonicBaseline,
    biphasic_kernel,
)

CUDA = torch.cuda.is_available()


def test_explicit_cpu_is_honoured_regardless_of_batch():
    assert resolve_device("cpu", batch=64).type == "cpu"


def test_auto_keeps_a_single_trace_on_cpu():
    """B = 1 is the fan-out case: 24 CPU fits at once beat one on the card."""
    assert resolve_device("auto", batch=1).type == "cpu"


@pytest.mark.skipif(not CUDA, reason="no CUDA device")
def test_auto_sends_a_cohort_to_the_gpu():
    """B > 1 is the batch case, where the GPU's flat per-step cost pays."""
    assert resolve_device("auto", batch=16).type == "cuda"


def test_auto_falls_back_to_cpu_when_there_is_no_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto", batch=32).type == "cpu"


def test_requesting_cuda_without_cuda_is_an_error_not_a_fallback(monkeypatch):
    """A GPU-sized run that quietly took 60x longer on CPU is worse than one that stops."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda", batch=8)


def test_an_out_of_range_cuda_index_is_an_error():
    if not CUDA:
        pytest.skip("no CUDA device")
    with pytest.raises(RuntimeError, match="only .* CUDA device"):
        resolve_device(f"cuda:{torch.cuda.device_count()}", batch=8)


def test_an_unknown_device_string_is_rejected():
    with pytest.raises(ValueError, match="unrecognized device"):
        resolve_device("tpu", batch=1)


def test_a_torch_device_passes_through():
    d = torch.device("cpu")
    assert resolve_device(d, batch=99) is d


def _model():
    kern = biphasic_kernel(length=40, dt=1 / 25, k_rise_fast=20.0, k_decay_fast=9.0,
                           k_rise_slow=8.0, k_decay_slow=3.5)
    return SingleTraceModel(
        PhasicEventProcess(kern, beta0=-2.8, beta_inj=0.0, mu_a=1.7, sigma_a=0.2,
                           dt=1 / 25, history=None),
        TonicBaseline(dt=1 / 25), AdditiveEmission(sigma_y=0.1))


def test_prepare_moves_the_model_and_the_traces_together():
    m, y = prepare(_model(), torch.zeros(2, 100), torch.device("cpu"))
    assert y.device.type == "cpu"
    assert next(m.parameters()).device.type == "cpu"


@pytest.mark.skipif(not CUDA, reason="no CUDA device")
def test_prepare_moves_buffers_not_just_parameters():
    """The kernel is a buffer; a fit whose kernel stayed behind would fail mid-step."""
    m, y = prepare(_model(), torch.zeros(2, 100), torch.device("cuda"))
    assert m.phasic.kernel.device.type == "cuda"
    assert y.device.type == "cuda"
