"""The paper's final model (2026-10-21): one single-trace architecture, built in one place."""
import math

import numpy as np
import torch
import yaml

from photosvi.experiments.marks import noise_unit_generation
from photosvi.experiments.single_trace import build_single_trace, is_final
from photosvi.models.points import build_kernel

CFG = yaml.safe_load(open("configs/final_detection_48.yaml"))


def test_the_builder_fits_the_configured_model():
    assert is_final(CFG)
    k = CFG["kernel"]
    dt = 1 / CFG["data"]["fs"]
    kern = build_kernel(dict(k["fit"]), k["length"], dt)
    torch.manual_seed(0)
    y = torch.randn(1, 6000) * 3.0
    model, info = build_single_trace(y, 3000, kernel=kern, dt=dt, fit=CFG["fit"],
                                     warm=CFG["warm"], detection=CFG["detection"],
                                     ar_coeffs=CFG["emission"]["ar_coeffs"])
    nu = CFG["fit"]["mark_noise_units"]
    # Marks fixed at sigma_y e^{mu_r}, width sigma_r, nothing about them trainable.
    assert math.isclose(float(model.phasic.mu_a), math.log(info["sigma_y"]) + nu["mu"], rel_tol=1e-5)
    assert math.isclose(float(model.phasic.sigma_a), nu["sigma"], rel_tol=1e-5)
    assert not model.phasic.mu_a.requires_grad and not model.phasic.log_sigma_a.requires_grad
    # One gate floor, the configured priors, tonic refined (not frozen).
    assert math.isclose(model.mark_gate_floor, CFG["fit"]["mark_gate_floor"])
    assert any(p.requires_grad for p in model.tonic.parameters())


def test_the_simulator_keeps_the_absolute_snr():
    """Generation at the config's mu_a with sigma 0.75, so SNR = e^{mu_a}/sigma_y."""
    f = CFG["fit"]
    mu, sig = noise_unit_generation(f, 2.0, CFG["gt"]["mu_a"], CFG["gt"]["sigma_a"])
    assert mu == CFG["gt"]["mu_a"] and sig == 0.75
    assert np.isclose(math.exp(mu) / 2.0, 2.93, atol=0.01)
    # Without the generation keys, the simulator draws from the fit's own distribution.
    f2 = dict(f, mark_noise_units={"mu": 0.668, "sigma": 0.75})
    mu2, sig2 = noise_unit_generation(f2, 2.0, CFG["gt"]["mu_a"], 0.2)
    assert math.isclose(mu2, math.log(2.0) + 0.668) and sig2 == 0.75
