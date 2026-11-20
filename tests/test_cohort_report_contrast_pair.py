"""The report has to score a two-condition cell, not only the three-drug study.

`scripts/cohort_report.py` fixed the contrast at drug indices (1, 2) -- the two opioids of
`configs/cohort_comparison.yaml`. The study the recordings have is two conditions, vehicle
and treated, indexed (0, 1), and on such a cell `by_drug[2]` is empty and the report died
with "mean requires at least one data point". Every cell of the power grid would have hit
it, after the fits had already been paid for.

The pair is now the last two drug indices present, which keeps the three-drug case
comparing the opioids to each other rather than either of them to the vehicle.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "cohort_report", Path(__file__).resolve().parents[1] / "scripts" / "cohort_report.py")
cohort_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cohort_report)


def test_two_conditions_contrast_the_pair_that_is_there():
    assert cohort_report._contrast_pair([0, 1, 0, 1]) == (0, 1)


def test_three_drugs_still_contrast_the_two_opioids():
    """Not (0, 1): the vehicle is not one side of this estimand."""
    assert cohort_report._contrast_pair([0, 1, 2, 0, 1, 2]) == (1, 2)


def test_one_condition_is_not_a_contrast():
    with pytest.raises(ValueError):
        cohort_report._contrast_pair([0, 0, 0])


def _sessions(drugs, betas):
    return [{"animal": i // 2, "drug": d, "beta_inj": b}
            for i, (d, b) in enumerate(zip(drugs, betas))]


@pytest.mark.parametrize("drugs,expect_pairs", [
    ([0, 1, 0, 1], 2),          # two conditions, two animals, crossed
    ([0, 1, 2, 0, 1, 2], 3),    # three drugs, three animals
])
def test_the_report_scores_both_designs(drugs, expect_pairs):
    """End to end through statistics_per_cohort: a row, and a finite estimate."""
    betas = [0.1 * (i + 1) for i in range(len(drugs))]
    data = {"hier": {}, "sess": {(0.4, 0): _sessions(drugs, betas)}, "guppy": {}}
    rows = cohort_report.statistics_per_cohort(data, n_sessions=len(drugs))
    assert len(rows) == 1
    row = rows[0]
    assert row["ts_session_est"] == row["ts_session_est"]      # not NaN
    # One animal per two sessions in `_sessions`, so the three-drug case pairs across
    # animals that share the two opioids rather than within one.
    assert row["n_pairs"] >= 1


def test_a_cell_with_only_guppy_results_is_still_scored():
    """The cohort key set has to include the GuPPy arm.

    It was built from the hierarchical and two-stage arms alone, so a cell holding only
    GuPPy results reported "cohorts with any result: 0" with 240 of them on disk. The
    GuPPy arm costs seconds against the others' hours, so it is the arm most likely to be
    the only one in a cell.
    """
    rows = cohort_report.statistics_per_cohort(
        {"hier": {}, "sess": {},
         "guppy": {(0.4, 0): _sessions([0, 1, 0, 1], [0.1, 0.5, 0.2, 0.6])}},
        n_sessions=4)
    assert len(rows) == 1
    assert "gp_session_t" in rows[0] and "gp_animal_t" in rows[0]
