"""Every config parses and describes something buildable.

A bad config is expensive here in a specific way: a run is launched, twenty-four
workers spend two hours, and the failure appears at the end or -- worse -- does not
appear at all and produces numbers. These tests are cheap and catch the classes of
mistake that have actually happened: a kernel spec whose lobes are the wrong way
round, an injection bin past the end of the recording, a per-drug array that does not
match the number of drugs.

They walk `configs/` rather than a list, so a new config is covered the moment it is
added and cannot be forgotten.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from photosvi.models.points import TonicBaseline, build_kernel

CONFIGS = sorted(Path("configs").glob("*.yaml"))


def _load(p):
    return yaml.safe_load(p.read_text())


def test_there_are_configs_to_check():
    assert CONFIGS, "no configs found; the tests below would be vacuous"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_parses_and_names_itself(path):
    cfg = _load(path)
    assert isinstance(cfg, dict), path
    assert cfg.get("name"), f"{path} has no name"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_the_kernel_spec_builds(path):
    cfg = _load(path)
    k = cfg.get("kernel")
    if not k:
        return
    kern = build_kernel(dict(k["fit"]), k["length"], 1 / cfg.get("data", {}).get("fs", 25))
    assert kern.numel() == k["length"]
    assert bool(kern.isfinite().all())
    if k["fit"].get("family") == "biphasic":
        # the property the whole identifiability argument rests on
        assert abs(float(kern.sum())) < 1e-4, f"{path}: kernel is not mean-free"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_the_injection_falls_inside_the_recording(path):
    cfg = _load(path)
    for block in (cfg.get("data"), cfg.get("cohort")):
        if not block or "T" not in block or "inj_t" not in block:
            continue
        T, inj = block["T"], block["inj_t"]
        assert 0 < inj < T, f"{path}: injection at {inj} of {T}"
        jitter = block.get("inj_jitter", 0)
        assert 0 <= inj - jitter and inj + jitter < T, f"{path}: jitter runs off the trace"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_the_tonic_parameters_are_a_valid_bateman(path):
    """k_rise > k_decay > 0 and A > 0, or TonicBaseline refuses to build."""
    cfg = _load(path)
    ton = cfg.get("tonic")
    if ton:
        TonicBaseline(dt=1 / 25, **ton)
    c = cfg.get("cohort")
    if c and "tonic_peak" in c:
        n = len(c["tonic_peak"])
        for f in ("tonic_k_rise", "tonic_k_decay"):
            assert len(c[f]) == n, f"{path}: {f} has {len(c[f])} entries, tonic_peak has {n}"
        for d in range(n):
            TonicBaseline(dt=1 / 25, peak=c["tonic_peak"][d], k_rise=c["tonic_k_rise"][d],
                          k_decay=c["tonic_k_decay"][d], drift=c.get("tonic_drift", 0.0))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_the_fit_block_is_self_consistent(path):
    cfg = _load(path)
    f = cfg.get("fit")
    if not f:
        return
    assert f.get("lr", 1e-2) > 0, path
    assert f.get("tau", 0.5) > 0, path
    if "max_steps" in f and "stop_min_steps" in f:
        assert f["stop_min_steps"] <= f["max_steps"], f"{path}: floor above the ceiling"
    if "stop_probe" in f and "stop_window" in f and "max_steps" in f:
        # the criterion cannot fire before probe*window steps have elapsed
        need = f["stop_probe"] * f["stop_window"]
        assert need <= f["max_steps"], (
            f"{path}: the stopping window spans {need} steps but the ceiling is "
            f"{f['max_steps']}, so it can never fire")
    if f.get("beta_prior") == "normal":
        assert f.get("beta0_prior_scale", 1) > 0 and f.get("beta_inj_prior_scale", 1) > 0


def test_the_cohort_config_is_internally_consistent():
    cfg = _load(Path("configs/cohort_comparison.yaml"))
    c, g = cfg["cohort"], cfg["grid"]
    assert c["n_animals"] >= 2, "a cohort needs more than one animal"
    assert len(c["tonic_peak"]) == 3, "three drugs: vehicle and two opioids"
    assert c["mu_saline"] > 0, (
        "vehicle is not a null condition -- handling and the injection raise the rate")
    assert 0.0 in g["delta"], "the null cell is what calibrates every arm's threshold"
    # the two opioids must differ pharmacokinetically or the tonic question is untestable
    peak = [math.log(r / d) / (r - d)
            for r, d in zip(c["tonic_k_rise"], c["tonic_k_decay"], strict=True)]
    assert max(peak[1], peak[2]) / min(peak[1], peak[2]) > 2.0


def test_the_three_guide_sweeps_differ_only_in_their_kernel():
    """A bases comparison is only a bases comparison if nothing else varies.

    The three `points_guide_sweep_*` configs feed one figure and one section of S1
    Text, whose whole claim is that each difference between them is the kernel. They
    drifted apart anyway: when `k5` was moved to the frozen production settings
    (tau 0.2, Normal prior, wider prior scales, the joint beta guide) the other two
    were left on the pre-freeze settings, so the non-negative basis was compared to
    the mean-free one across five differences and reported as one. This asserts the
    property the comparison rests on, which nothing else did.
    """
    names = ["points_guide_sweep_k5", "points_guide_sweep_nonneg",
             "points_guide_sweep_k5_misspec"]
    cfgs = {n: yaml.safe_load(Path(f"configs/{n}.yaml").read_text()) for n in names}
    ref = cfgs["points_guide_sweep_k5"]
    for n, c in cfgs.items():
        if n == "points_guide_sweep_k5":
            continue
        # `name` and `kernel` are what a basis arm is allowed to change; everything
        # else -- gt, data, tonic, fit, grid, detection -- has to be the reference's.
        for key in set(ref) | set(c):
            if key in ("name", "kernel"):
                continue
            assert c.get(key) == ref.get(key), f"{n}: block {key!r} differs from k5"
        # Inside the kernel block only the bases themselves may move; the support
        # length is shared, since a different length would change the comparison too.
        assert c["kernel"]["length"] == ref["kernel"]["length"], n
