"""The command line, which is what a user touches first and nothing covered.

These are contract tests: that the entry point exists and dispatches, that the flags
a user is told about in the docs actually parse, and that a wrong invocation fails
loudly instead of doing something surprising. Nothing here runs a fit -- `--print-grid`
enumerates the grid without fitting, which is exactly the cheap surface worth testing.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

# tomllib is stdlib only from 3.11, and the package supports 3.10. The dev extra
# pins tomli there, which has the same loads(); without this the whole module fails
# to import on 3.10 and takes the rest of the run down with it.
try:
    import tomllib
except ModuleNotFoundError:                              # Python 3.10
    import tomli as tomllib

from photosvi.cli.main import _COMMANDS, main


def _run(args, **kw):
    return subprocess.run([sys.executable, "-m", "photosvi.cli.main", *args],
                          capture_output=True, text=True, **kw)


def test_the_console_script_is_declared_and_points_somewhere_real():
    """`photosvi = photosvi.cli.main:main` in pyproject must resolve."""
    pt = tomllib.loads(Path("pyproject.toml").read_text())
    target = pt["project"]["scripts"]["photosvi"]
    mod_path, _, fn = target.partition(":")
    assert callable(getattr(importlib.import_module(mod_path), fn))


def test_every_registered_command_imports_and_exposes_main():
    """A typo in the dispatch table is otherwise only found by a user."""
    for (family, command), dotted in _COMMANDS.items():
        mod = importlib.import_module(dotted)
        assert hasattr(mod, "main"), f"{family} {command} -> {dotted} has no main()"


def test_help_exits_zero():
    r = _run(["--help"])
    assert r.returncode == 0
    assert "photosvi" in r.stdout


def test_missing_family_is_an_error_not_a_traceback():
    r = _run([])
    assert r.returncode != 0
    assert "Traceback" not in r.stderr


def test_unknown_family_is_rejected():
    r = _run(["nope", "sweep"])
    assert r.returncode != 0


def test_print_grid_enumerates_the_cells_without_fitting():
    """The flag the runner scripts pipe into xargs; if it changes shape they break."""
    r = _run(["points", "sweep", "--config", "configs/points_guide_sweep_smoke.yaml",
              "--print-grid"])
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert lines, "no cells printed"
    for ln in lines:
        # each line must be consumable as `--arm X --seed N`
        assert "--arm" in ln and "--seed" in ln, ln


def test_sweep_requires_a_target():
    """Neither --arm/--seed nor --all nor --print-grid is a user error, not a crash."""
    r = _run(["points", "sweep", "--config", "configs/points_guide_sweep_smoke.yaml"])
    assert r.returncode != 0
    assert "Traceback" not in r.stderr


def test_sweep_rejects_an_unknown_arm():
    r = _run(["points", "sweep", "--config", "configs/points_guide_sweep_smoke.yaml",
              "--arm", "not-an-arm", "--seed", "0"])
    assert r.returncode != 0


def test_main_is_callable_in_process():
    """The console script calls main(argv); it must work without a subprocess."""
    with pytest.raises(SystemExit):
        main(["--help"])


def test_kernel_calibrate_writes_a_config_block(tmp_path):
    """The kernel block the CLI writes is the one configs and FitSettings read."""
    import numpy as np
    import yaml

    from photosvi.models.points.events import build_kernel

    fs, n = 48.0, int(20 * 60 * 48)
    rng = np.random.default_rng(0)
    s = np.zeros(n)
    ev = rng.random(n) < 6 / 60 / fs
    s[ev] = 1.0
    h = build_kernel(dict(family="bigamma", shape_fast=5.68, rate_fast=9.98,
                          shape_slow=11.5, rate_slow=3.59), 384, 1 / fs).numpy()
    y = np.convolve(s, h)[:n] + rng.normal(0, 0.05, n)
    np.save(tmp_path / "rec.npy", y)
    r = _run(["kernel", "calibrate", str(tmp_path / "rec.npy"), "--fs", "48",
              "--out", str(tmp_path / "k.yaml")])
    assert r.returncode == 0, r.stderr
    block = yaml.safe_load((tmp_path / "k.yaml").read_text())["kernel"]
    assert block["length"] > 0 and block["fit"]["family"] in {"biphasic", "bigamma", "biexp"}
