"""The mark posterior's scale is bounded, and the bound cannot bind on a healthy fit."""
from __future__ import annotations

import math

import pyro
import pytest
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
    pyro.param("q_a_scale", torch.full((2, 4), _MARK_SCALE_INIT),
               constraint=pyro.distributions.constraints.interval(0.0, _MARK_SCALE_MAX))
    with torch.no_grad():                       # as a runaway gradient would
        pyro.get_param_store()._params["q_a_scale"].fill_(60.0)
    out = pyro.param("q_a_scale")
    assert float(out.max()) <= _MARK_SCALE_MAX
    assert torch.isfinite(out.exp()).all()
    pyro.clear_param_store()


def test_the_gate_floor_defaults_to_the_old_behaviour():
    """Zero by default, so nothing already fitted changes meaning.

    The floor exists because gating the mark divergence by q(s_t) leaves q(a)'s width
    unconstrained wherever the gate is near zero -- which is almost everywhere, since most
    bins hold no event. Turning it on changes the objective, so it is opt-in and the runs
    that predate it keep their semantics.
    """
    from photosvi.models.points.points import _MARK_GATE_FLOOR

    assert _MARK_GATE_FLOOR == 0.0


def test_a_floor_outside_zero_to_one_is_refused():
    """A floor of 1 charges every bin in full, which is the ungated objective under another
    name; above 1 it is not a weight at all. Both are mistakes worth catching at the call."""
    import pytest

    from photosvi.models.points import (
        AdditiveEmission,
        PhasicEventProcess,
        TonicBaseline,
        build_kernel,
    )
    from photosvi.models.points.points import SingleTraceModel

    def build(floor):
        kern = build_kernel(dict(family="biexp", k_rise=2.0, k_decay=1.3), 50, 1 / 25)
        return SingleTraceModel(
            PhasicEventProcess(kern, beta0=-2.8, beta_inj=0.0, mu_a=1.7, sigma_a=0.2,
                               dt=1 / 25, history=None),
            TonicBaseline(dt=1 / 25), AdditiveEmission(sigma_y=0.1),
            mark_gate_floor=floor)

    build(0.0), build(0.5)                      # both legal
    for bad in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="mark_gate_floor"):
            build(bad)


def test_the_floor_lifts_a_vanishing_gate_and_leaves_a_full_one_alone():
    """What the clamp does, stated on the quantity itself.

    At a bin with no event the event guide's clamp puts the gate at 1e-6, so the divergence
    there is charged at a millionth and the width is free; the floor charges it at the floor
    instead. At a bin that does hold an event the gate is already near one and the floor is
    below it, so nothing changes -- which is why the floor cannot distort the marks that
    matter.
    """
    floor = 0.01
    empty = torch.sigmoid(torch.tensor(-13.8))     # the guide's own clamp, logit(1e-6)
    held = torch.sigmoid(torch.tensor(13.8))
    assert float(empty) < floor < float(held)
    assert float(empty.clamp_min(floor)) == pytest.approx(floor)
    assert float(held.clamp_min(floor)) == float(held)
