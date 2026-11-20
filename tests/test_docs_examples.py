"""Execute the documentation examples.

The examples are included into the docs by file reference, so there is exactly one
copy of each and it is this one. Running them here is what stops the documentation
from drifting away from the API: a rename that breaks an example fails the build
rather than leaving a snippet that no longer works.

They run on `configs/toy.yaml` and finish in seconds, which is the point -- an
example nobody can afford to run in CI is an example that rots.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "docs" / "examples").glob("*.py"))


def test_there_are_examples_to_run():
    """A glob that silently matches nothing would make every test below vacuous."""
    assert EXAMPLES, "no examples found under docs/examples/"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(path, monkeypatch, tmp_path):
    # The examples open configs by relative path, as a reader would from the repo root.
    monkeypatch.chdir(ROOT)
    # The toy examples write their files here rather than into the source tree.
    monkeypatch.setenv("PHOTOSVI_TOY", str(tmp_path / "photosvi_toy"))
    runpy.run_path(str(path), run_name="__main__")


def test_every_example_is_referenced_by_a_docs_page():
    """An example nobody includes is one that will be deleted by accident."""
    pages = "\n".join(p.read_text() for p in (ROOT / "docs").rglob("*.md"))
    missing = [p.name for p in EXAMPLES if p.name not in pages]
    assert not missing, f"examples not included in any page: {missing}"


@pytest.mark.skipif(
    subprocess.run([sys.executable, "-c", "import mkdocs"],
                   capture_output=True).returncode != 0,
    reason="mkdocs not installed",
)
def test_the_docs_build_without_warnings():
    """--strict turns a broken link or a missing include into a failure."""
    r = subprocess.run([sys.executable, "-m", "mkdocs", "build", "--strict",
                        "--site-dir", "/tmp/photosvi_docs_test"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
