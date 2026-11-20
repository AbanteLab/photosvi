"""The tonic-response cohort benchmark (photosvi.experiments.tonic_benchmark)."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
import yaml

from photosvi.experiments.cohort import spec_from_config
from photosvi.experiments.tonic_benchmark import (
    RSCRIPT,
    bfp_arms,
    binned_traces,
    flmm_arm,
    hier_arm,
    tonic_config,
    twostage_arm,
    write_cell_config,
)

CFG = "configs/cohort_tonic_48.yaml"


def test_the_cell_config_changes_only_the_second_conditions_peak(tmp_path):
    base = yaml.safe_load(open(CFG))
    c = tonic_config(base, 0.5)
    p0 = base["cohort"]["tonic_peak"][0]
    assert c["cohort"]["tonic_peak"] == pytest.approx([p0, 1.5 * p0])
    assert base["cohort"]["tonic_peak"][1] == p0          # the base is not mutated
    p = write_cell_config(CFG, 1.0, tmp_path, {"tonic_block": True})
    w = yaml.safe_load(open(p))
    assert w["fit"]["tonic_block"] is True and w["fit"]["mark_mode"] == "noise_units"
    assert spec_from_config(w, 0.0).mu_drug[0] == spec_from_config(w, 0.0).mu_drug[1]


def test_bins_are_aligned_to_each_injection_and_share_one_grid():
    fs, T = 4.0, 4 * 60 * 30
    inj = [4 * 60 * 10 - 30, 4 * 60 * 10 + 50]
    y = np.zeros((2, T))
    for s, t in enumerate(inj):
        y[s, t:] = 1.0                                    # a step at each injection
    b, c = binned_traces(y, inj, fs)
    assert b.shape[0] == 2 and b.shape[1] == c.size
    assert np.allclose(b[:, c < 0], 0) and np.allclose(b[:, c > 0], 1)


def test_the_lab_auc_arms_see_a_slow_difference_and_ignore_none():
    fs, T, inj = 4.0, 4 * 60 * 40, 4 * 600
    rng = np.random.default_rng(1)
    t = np.arange(T)
    bump = np.where(t >= inj, np.exp(-(t - inj) / (fs * 600)) - np.exp(-(t - inj) / (fs * 60)), 0)
    animal, drug = [a for a in range(5) for _ in (0, 1)], [0, 1] * 5
    y = np.array([rng.normal(0, 1, T) + (1 + d) * 3 * bump for d in drug])
    r = bfp_arms(y, [inj] * 10, fs, animal, drug)
    for v in ("zlp", "zals"):
        assert r[v]["t"] > 3 and len(r[v]["auc_post_windows"][0]) == 6
    y0 = np.array([rng.normal(0, 1, T) + 3 * bump for _ in drug])
    r0 = bfp_arms(y0, [inj] * 10, fs, animal, drug)
    assert all(r0[v]["stat"] < r[v]["stat"] for v in ("zlp", "zals"))


def test_the_als_baseline_sits_under_peaks():
    from photosvi.experiments.tonic_benchmark import als_baseline
    x = np.linspace(0, 1, 2000)
    y = x.copy()
    y[::200] += 5.0
    z = als_baseline(y, lam=1e4)
    assert np.abs(z - x).max() < 0.2


def test_the_two_photosvi_arms_estimate_log_one_plus_dp():
    rows = [dict(animal=a, drug=d, tonic_peak=1.0 * (2.0 if d else 1.0) * (1 + 0.01 * a))
            for a in range(5) for d in (0, 1)]
    rows[1]["tonic_peak"] *= 1.02                         # some spread for the t-test
    t = twostage_arm(rows)
    assert t["est"] == pytest.approx(math.log(2), abs=0.01) and t["n_pairs"] == 5
    q = {"vehicle": {"loc": 0.0, "scale": 0.1}, "treated": {"loc": math.log(2), "scale": 0.1}}
    h = hier_arm({"tonic_alpha_q": q})
    assert h["est"] == pytest.approx(math.log(2)) and h["scale"] == pytest.approx(0.1 * math.sqrt(2))
    # A model that reports the contrast itself (a joint tonic block) is read first.
    h2 = hier_arm({"tonic_alpha_q": q, "tonic_alpha_contrast": {"loc": 0.7, "scale": 0.05}})
    assert h2["stat"] == pytest.approx(14.0)
    h3 = hier_arm({"tonic_alpha_q": q, "tonic_contrast_alpha": [0.6, 0.2]})
    assert h3["est"] == 0.6 and h3["stat"] == pytest.approx(3.0)


def test_the_simulated_cohort_carries_the_tonic_difference(tmp_path):
    p = write_cell_config(CFG, 1.0, tmp_path)
    spec = spec_from_config(yaml.safe_load(open(p)), 0.0)
    assert spec.tonic_peak[1] == pytest.approx(2 * spec.tonic_peak[0])
    assert spec.mu_drug[0] == spec.mu_drug[1]             # no change in the transient rate


@pytest.mark.skipif(not os.path.exists(RSCRIPT), reason="R with fastFMM not available")
def test_flmm_finds_a_post_injection_difference_and_not_a_null(tmp_path):
    import subprocess
    ok = subprocess.run([RSCRIPT, "-e", "library(fastFMM)"], capture_output=True).returncode == 0
    if not ok:
        pytest.skip("fastFMM not installed")
    rng = np.random.default_rng(0)
    c = np.arange(-9, 29) + 0.5
    animal, drug = [a for a in range(5) for _ in (0, 1)], [0, 1] * 5
    eff = np.array([rng.normal(0, 0.01, c.size) + 0.02 * a
                    + (0.03 * (c > 0) * np.exp(-np.maximum(c, 0) / 15) if d else 0)
                    for a, d in zip(animal, drug)])
    null = np.array([rng.normal(0, 0.01, c.size) + 0.02 * a for a in animal])
    r1 = flmm_arm(eff, c, animal, drug, tmp_path / "e")
    r0 = flmm_arm(null, c, animal, drug, tmp_path / "n")
    assert r1["joint_excludes_zero"] and r1["stat"] > r0["stat"]
