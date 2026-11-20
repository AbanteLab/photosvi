"""A config's slow response has to be in proportion to its transients.

`tonic_peak` is an amplitude in the trace's own units, and so are `mu_a` and `sigma_y`.
When a config is re-derived for a different recording scale, all three have to move
together. On 2026-10-05 `cohort_power_48.yaml` was found carrying
`tonic_peak = 3.939445759311477` -- the opioid entry of `cohort_comparison.yaml`'s
per-drug vector -- while `sigma_y` had come down from 0.1062 to 0.01299 and `mu_a` from
1.7666 to -3.421. The transient and the noise had been recalibrated to the recordings by a
factor of about 180 and the slow response had not, which left it at 121 times the
transient amplitude.

That is not a cosmetic error. The slow response and the change in transient rate both
start at the injection and both raise the signal afterwards, so they are nearly
confounded; at 121 times the transient, a fraction of a percent of misfit in the slow
component is the whole of the effect being estimated. The cell simulated that way reported
the estimator recovering 0.23 of a true 0.40 with none of its twelve cohorts covering it,
which reads as a verdict on the method and was a verdict on the simulation.

For reference, measured and as-simulated:

    the recordings, fitted     9.5   (0.1962 over exp(-3.8797), sigma_y 0.01224)
    cohort_comparison, opioid  0.67
    benchmark_guppy_48         0.67
    cohort_power_48, was       121
    cohort_power_48, now       6.0
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

# The recordings put the slow response at 9.5 times the transient amplitude. The bound is
# deliberately loose -- it is there to catch a value that survived a rescaling, which is
# an error of two orders of magnitude, not to pin the calibration.
MAX_TONIC_OVER_MARK = 25.0

CONFIGS = sorted(Path("configs").glob("*.yaml"))


def _scales(cfg: dict) -> tuple[float, float] | None:
    """(tonic amplitude, E[a]) if the config simulates both, else None."""
    for block in (cfg.get("cohort"), cfg.get("gt"), cfg):
        if not isinstance(block, dict):
            continue
        peak = block.get("tonic_peak")
        if peak is None and isinstance(cfg.get("tonic"), dict):
            peak = cfg["tonic"].get("peak")
        if peak is None or "mu_a" not in block:
            continue
        peaks = peak if isinstance(peak, list) else [peak]
        return max(peaks), math.exp(block["mu_a"])
    return None


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_the_tonic_is_in_proportion_to_the_transients(path):
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        pytest.skip("not a mapping")
    s = _scales(cfg)
    if s is None:
        pytest.skip("simulates no tonic, or no mark scale to compare it with")
    tonic, mark = s
    ratio = tonic / mark
    assert ratio <= MAX_TONIC_OVER_MARK, (
        f"{path.name}: slow response is {ratio:.0f}x the transient amplitude "
        f"(tonic_peak {tonic:g}, E[a] {mark:g}). The recordings carry 9.5. A value this "
        f"large is the signature of an amplitude carried across a rescaling of mu_a and "
        f"sigma_y -- see this test's docstring.")
