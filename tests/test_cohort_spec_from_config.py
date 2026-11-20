"""Every CohortSpec field a config can name, the config actually reaches.

The failure this guards against has no symptom. A config that says 48 Hz and gives a
kernel length in 48 Hz bins, while `spec_from_config` leaves the spec at its 25 Hz default,
simulates recordings at one rate with a basis built for another and reports nothing. Five
errors of this exact shape have been found in this project, which is four more
than it takes to justify a test.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
import yaml

from photosvi.experiments.cohort import CohortSpec, spec_from_config

CONFIGS = sorted(Path("configs").glob("cohort_*.yaml"))


def test_there_are_cohort_configs_to_check():
    assert CONFIGS, "no cohort configs found; this test would pass vacuously"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_key_the_config_names_reaches_the_spec(path):
    """A key in the config's `cohort` block that the spec has must arrive in the spec.

    Not that the config names everything -- defaults are deliberate -- but that nothing it
    does name is silently dropped.
    """
    cfg = yaml.safe_load(path.read_text())
    spec = spec_from_config(cfg, 0.0)
    fields = {f.name for f in dataclasses.fields(CohortSpec)}
    from photosvi.experiments.marks import noise_unit_mode
    # Under noise-unit marks the mark distribution is derived from sigma_y by design, so a
    # named mu_a/sigma_a is overridden rather than dropped (experiments.marks).
    derived = {"mu_a", "sigma_a"} if noise_unit_mode(cfg.get("fit")) else set()
    for key, want in cfg["cohort"].items():
        if key not in fields or key in derived:
            continue                      # the config may carry keys the spec derives
        got = getattr(spec, key)
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            assert got == pytest.approx(want), f"{path.name}: {key} named {want}, spec has {got}"
        elif isinstance(want, (list, tuple)):
            assert tuple(got) == tuple(want), f"{path.name}: {key} did not arrive"
        else:
            assert got == want, f"{path.name}: {key} named {want!r}, spec has {got!r}"


def test_the_constructor_reads_every_field_it_could():
    """A field no branch of `spec_from_config` assigns cannot be set from any config.

    `tonic_b` and `fs` were both in this state until 2026-10-02. Listing the exceptions
    here rather than allowing any means a new field that nothing reads has to be argued for.
    """
    src = Path("src/photosvi/experiments/cohort.py").read_text()
    body = src[src.index("def spec_from_config"):src.index("def simulate_cohort")]
    assigned = set(re.findall(r"(\w+)=", body))
    fields = {f.name for f in dataclasses.fields(CohortSpec)}
    unreachable = sorted(f for f in fields if f not in assigned)
    assert not unreachable, f"these fields cannot be set from a config: {unreachable}"
