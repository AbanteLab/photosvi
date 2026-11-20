"""The three arms of the comparison must simulate the same cohort, overrides included.

Their comparability rests entirely on regenerating the same recordings from `(spec, seed)`.
An override applied by one arm and not the others hands them different data, and nothing
downstream could detect it: the numbers would be wrong in a way that looks like a result.
Until 2026-10-02 the hierarchical arm applied `DESIGN` and `N_ANIMALS` and the other two
did not.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

from photosvi.experiments.cohort import _COHORT_ENV, apply_cohort_env, spec_from_config

ARMS = ["scripts/cohort_hierarchical.py", "scripts/cohort_twostage.py",
        "scripts/cohort_twostage_guppy.py"]


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_applies_the_shared_overrides(arm):
    src = Path(arm).read_text()
    assert "apply_cohort_env" in src, f"{arm} does not apply the environment overrides"


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_can_resolve_the_name_it_calls(arm):
    """Calling it is not enough: the name has to be in scope at module level.

    The substring test above passes on a script that calls `apply_cohort_env` without
    importing it, which is exactly what both two-stage arms did until 2026-10-05 --- they
    raised NameError on the first line that touches the config, so every cell of the grid
    would have died on launch. The names are resolved from the AST rather than by importing
    the script, which would run the whole fit.
    """
    tree = ast.parse(Path(arm).read_text())
    bound = set()
    for node in tree.body:                       # module level only: that is where it is called
        if isinstance(node, ast.ImportFrom):
            bound |= {a.asname or a.name for a in node.names}
        elif isinstance(node, ast.Import):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            bound |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "apply_cohort_env" in called, f"{arm} never calls apply_cohort_env"
    assert "apply_cohort_env" in bound, (
        f"{arm} calls apply_cohort_env but never imports or defines it")


@pytest.mark.parametrize("arm", ARMS)
def test_no_arm_applies_an_override_of_its_own(arm):
    """An arm reaching into os.environ for a cohort key is the bug coming back."""
    src = Path(arm).read_text()
    for env in _COHORT_ENV:
        hits = [ln for ln in src.splitlines()
                if env in ln and "environ" in ln and "apply_cohort_env" not in ln]
        assert not hits, f"{arm} reads {env} itself: {hits}"


def test_the_overrides_reach_the_spec(monkeypatch):
    cfg = yaml.safe_load(Path("configs/cohort_power_48.yaml").read_text())
    before = spec_from_config(dict(cfg, cohort=dict(cfg["cohort"])), 0.0)
    monkeypatch.setenv("N_ANIMALS", "7")
    monkeypatch.setenv("DESIGN", "uncrossed")
    after = spec_from_config(apply_cohort_env(cfg), 0.0)
    assert before.n_animals != 7
    assert after.n_animals == 7
    assert after.design == "uncrossed"


def test_the_env_table_covers_what_the_runner_sets():
    """Whatever the grid script exports has to be in the table, or it is silently ignored."""
    runner = Path("scripts/run_cohort_power_grid.sh").read_text()
    used = set(re.findall(r"\b(DESIGN|N_ANIMALS|SIM_[A-Z0-9_]+)=", runner))
    assert used, "the runner sets no cohort overrides; this test would pass vacuously"
    assert used <= set(_COHORT_ENV), f"the runner sets {used - set(_COHORT_ENV)}, which nothing reads"
