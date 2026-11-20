"""The mark posterior's scale is bounded, and the bound cannot bind on a healthy fit."""
from __future__ import annotations

import math

import pyro
import torch

from photosvi.models.points.points import _MARK_SCALE_INIT, _MARK_SCALE_MAX


def test_the_bound_is_far_above_where_a_working_fit_sits():
    """It is a guard, not a tuning knob.

    The initialisation is 0.5 and a prior spread of order 1 is already generous for a
    log-amplitude, so a bound of 5 sits an order of magnitude clear of any fit that is
    working. What it rules out is the regime that killed the opioid-cohort run: the mark KL
    is gated by the event probability, so in the bins holding no event nothing constrains
    q(a)'s width, and the scale ran to 30 before exp(loc + 4 scale) passed the largest
    float32 and one infinite mark gave 0 * inf = NaN in the mean.
    """
    assert _MARK_SCALE_MAX >= 10 * _MARK_SCALE_INIT


def test_the_bound_keeps_the_mark_finite_in_single_precision():
    """At the bound, a four-sigma draw still exponentiates to a finite float32."""
    limit = math.log(torch.finfo(torch.float32).max)
    # a_loc of 2 is already an enormous location for these units; the point is the margin.
    assert 2.0 + 4.0 * _MARK_SCALE_MAX < limit


def test_the_guide_cannot_push_the_scale_past_the_bound():
    """Whatever the optimiser does to the unconstrained value, the constrained one holds."""
    pyro.clear_param_store()
    p = pyro.param("q_a_scale", torch.full((2, 4), _MARK_SCALE_INIT),
                   constraint=pyro.distributions.constraints.interval(0.0, _MARK_SCALE_MAX))
    assert torch.isclose(p, torch.tensor(_MARK_SCALE_INIT)).all()
    with torch.no_grad():                       # as a runaway gradient would
        pyro.get_param_store()._params["q_a_scale"].fill_(60.0)
    out = pyro.param("q_a_scale")
    assert float(out.max()) <= _MARK_SCALE_MAX
    assert torch.isfinite(out.exp()).all()
    pyro.clear_param_store()
