"""The fit-quality statistic, and the distinction it exists to draw.

A plateau on the estimand is not a statement that the model explained the recording.
These tests pin the statistic that says whether it did, on data where the answer is
known, and pin what it cannot say as firmly as what it can -- the phasic component
turns out to absorb a great deal that the generative story never claimed.
"""

from __future__ import annotations

import math

import pyro
import torch

from photosvi.inference.points import FitQuality, fitted_mean, residual_ratio
from photosvi.models.points import (
    AdditiveEmission,
    PhasicEventProcess,
    SingleTraceModel,
    TonicBaseline,
    bi_exponential_kernel,
)

DT, FS = 0.04, 25


def _model(sigma_y=0.1, mu_a=1.0, T=1200):
    return SingleTraceModel(
        PhasicEventProcess(bi_exponential_kernel(40, k_rise=2.0, k_decay=1.0),
                           beta0=-3.0, beta_inj=0.0, mu_a=mu_a, sigma_a=0.21,
                           dt=DT, history=None),
        TonicBaseline(), AdditiveEmission(sigma_y=sigma_y), mark_window=5)


def test_fitted_mean_is_the_models_own_emission_mean():
    """Read from the model, not rebuilt beside it.

    The point of going through a replayed trace is that it cannot disagree with the
    model: whatever `y` was scored under is what comes back. A second implementation
    of tonic-plus-phasic would be correct until someone changed one of them.
    """
    T, INJ = 800, 400
    m = _model(T=T)
    m.train(False)
    pyro.clear_param_store()
    pyro.set_rng_seed(0)
    y = torch.randn(1, T) * 0.1

    mean = fitted_mean(m, y, INJ, draws=2)
    assert mean.shape == y.shape
    assert torch.isfinite(mean).all()

    # Against a single replay done by hand, which is what `draws=1` reduces to.
    import pyro.poutine as poutine
    pyro.set_rng_seed(7)
    with torch.no_grad():
        g = poutine.trace(m.guide).get_trace(y, INJ)
        mt = poutine.trace(poutine.replay(m.model, trace=g)).get_trace(y, INJ)
        by_hand = mt.nodes["y"]["fn"].mean
    pyro.set_rng_seed(7)
    assert torch.allclose(fitted_mean(m, y, INJ, draws=1), by_hand, atol=1e-6)


def test_the_ratio_does_not_depend_on_the_scale_of_the_recording():
    """The normalisation is the whole point: "eleven" must mean the same everywhere.

    A recording in different units, with the emission told about it, is the same
    problem and has to give the same number. Without that the statistic cannot be
    compared across sessions, which is exactly what it is for -- one cohort fit being
    sound on twenty-three recordings and lost on the twenty-fourth.

    Deliberately NOT asserted: that the ratio falls as a fit proceeds. It need not.
    Measured on self-generated data, 600 steps at lr 0.02 took it from 9.2 to 34.2,
    because the fitted mean starts nearly flat -- so the residual starts as the
    recording's own spread -- and the phasic overshoots before it settles. Anyone
    reading this number mid-fit should read `worst` beside it, not a trend.
    """
    T, INJ, SIGMA, MU_A = 900, 400, 0.05, 1.0
    torch.manual_seed(13)
    y = torch.randn(1, T) * SIGMA

    ratios = []
    for factor in (1.0, 37.0):
        # EVERYTHING in the units of the trace moves together: the noise, the mark
        # prior (log units, so it shifts) and the tonic's peak. Scaling only the
        # recording would pose a different problem rather than the same one in other
        # units -- a mark prior of 1.0 is twenty sigma_y at one scale and half a
        # sigma_y at the other, and the fits are then not comparable at all.
        m = SingleTraceModel(
            PhasicEventProcess(bi_exponential_kernel(40, k_rise=2.0, k_decay=1.0),
                               beta0=-3.0, beta_inj=0.0, mu_a=MU_A + math.log(factor),
                               sigma_a=0.21, dt=DT, history=None),
            TonicBaseline(peak=1.0 * factor, dt=DT),
            AdditiveEmission(sigma_y=SIGMA * factor), mark_window=5)
        m.train(False)
        pyro.clear_param_store()
        pyro.set_rng_seed(6)
        ratios.append(residual_ratio(m, y * factor, INJ, draws=2)[0])

    assert abs(ratios[0] - ratios[1]) < 1e-3, ratios


def test_a_fit_whose_errors_exceed_its_claimed_noise_reads_high():
    """The units are the model's own sigma_y, so understating it shows up here.

    This is the failure the statistic is for, reproduced deliberately: the emission
    claims errors of 0.02 and the recording carries ten times that, so the ratio must
    report roughly ten however well the rest of the fit goes. It is the same shape as
    the real morphine recording, whose tonic left the data and whose residual sat at
    eleven times a sigma_y of 0.0122 while the estimand looked settled.
    """
    T, INJ = 1000, 500
    CLAIMED, ACTUAL = 0.02, 0.2
    torch.manual_seed(11)
    y = torch.randn(1, T) * ACTUAL

    m = _model(sigma_y=CLAIMED, T=T)
    m.train(False)
    pyro.clear_param_store()
    pyro.set_rng_seed(4)
    overall, per = residual_ratio(m, y, INJ, draws=4)

    assert len(per) == 1 and abs(per[0] - overall) < 1e-6
    assert overall > 5, f"understated noise must show as a large ratio, got {overall}"
    q = FitQuality()
    q.update(1000, overall)
    assert not q.sound and "should not be read" in q.verdict()


def test_the_tracker_separates_where_it_ended_from_how_bad_it_got():
    """`ratio` is the end state, `worst` is the excursion; a mean would lose both."""
    q = FitQuality(tol=1.5, window=4)
    assert not q.sound and q.verdict().endswith("not probed")

    for step, r in ((1000, 1.02), (2000, 1.05), (3000, 11.2), (4000, 1.03),
                    (5000, 1.01), (6000, 1.04)):
        q.update(step, r)
    assert q.sound, "a fit that spiked once and recovered ended sound"
    assert abs(q.ratio - 1.035) < 1e-9
    assert q.worst == 11.2 and q.worst_step == 3000, "the excursion is still recorded"
    assert q.summary()["n_probes"] == 6

    bad = FitQuality(tol=1.5, window=4)
    for step, r in ((1000, 9.0), (2000, 10.5), (3000, 11.2), (4000, 10.9)):
        bad.update(step, r)
    assert not bad.sound
    # The message has to say what it means, not just print a number: this is the
    # line that would have stopped the real morphine estimate being written out.
    assert "should not be read" in bad.verdict()
    assert bad.verdict().startswith("FIT QUALITY")


def test_the_tracker_guards_its_arguments():
    for kw in (dict(tol=0.0), dict(tol=-1.0), dict(window=0)):
        try:
            FitQuality(**kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kw}")
    try:
        fitted_mean(_model(), torch.zeros(1, 100), 50, draws=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for draws=0")


def test_block_spread_sees_a_fan_out_that_the_centre_hides():
    """The failure mode is a tail, so the summary has to be built on tails.

    Every divergence of the week of 2026-09-21 was a per-window block moving somewhere
    nobody was looking, and in the one that was traced step by step the marks at empty
    windows fanned out while their median barely moved. A mean-and-sd summary reads
    that as calm. Percentiles read it as what it is.
    """
    import pyro
    import torch

    from photosvi.inference.points.fitquality import PER_BIN_BLOCKS, block_spread
    from photosvi.models.points import WindowedCategoricalEventGuide

    T, INJ, W = 4000, 2000, 5
    m = SingleTraceModel(
        PhasicEventProcess(bi_exponential_kernel(125, k_rise=2.0, k_decay=1.0),
                           beta0=-2.85, beta_inj=0.0, mu_a=1.77, sigma_a=0.21,
                           dt=DT, history=None),
        TonicBaseline(), AdditiveEmission(sigma_y=0.1),
        event_guide=WindowedCategoricalEventGuide(torch.zeros(1, T), window=W),
        mark_window=W)
    m.train(False)
    pyro.clear_param_store()
    pyro.set_rng_seed(0)
    m.guide(torch.randn(1, T) * 0.1, INJ)

    before = block_spread()
    assert "q_a_loc" in before and "q_s_window_scores" in before
    assert all(before[k]["finite"] for k in before)
    assert before["q_a_loc"]["spread"] == 0.0, "an untouched block starts uniform"

    with torch.no_grad():
        torch.manual_seed(1)
        pyro.param("q_a_loc").add_(torch.randn_like(pyro.param("q_a_loc")) * 2.0)
    after = block_spread()

    moved_centre = abs(after["q_a_loc"]["median"] - before["q_a_loc"]["median"])
    assert moved_centre < 0.2, "the centre is not what moved"
    assert after["q_a_loc"]["spread"] > 5.0, "the spread is"
    # and the block that was not touched is unchanged, so this localises
    assert after["q_s_window_scores"]["spread"] == before["q_s_window_scores"]["spread"]

    # A block the installed guide does not create is absent, not an error.
    assert set(after) <= set(PER_BIN_BLOCKS)


def test_the_residual_diagnostic_survives_a_coloured_emission():
    """`fitted_mean` must return the fitted SIGNAL, not a one-step-ahead prediction.

    Under a correlated emission the mean at the `y` site is the conditional mean: it
    carries the previous residual. Reading that as "the signal the model fitted" turns
    every residual diagnostic into a prediction error, which on a smooth trace is near
    zero and would report a flawless fit to anything. Regression test, because adding
    the AR likelihood broke exactly this and nothing else noticed: the numbers stayed
    plausible, they just stopped meaning what the paper says they mean.
    """
    import pyro
    import torch

    from photosvi.inference.points.fitquality import fitted_mean
    from photosvi.models.points import PhasicEventProcess, TonicBaseline, build_kernel
    from photosvi.models.points.emission import AdditiveEmission
    from photosvi.models.points.points import SingleTraceModel

    torch.manual_seed(0)
    T = 400
    kernel = build_kernel(dict(family="biphasic", k_rise_fast=20.0, k_decay_fast=9.0,
                               k_rise_slow=8.0, k_decay_slow=3.5), 50, 1 / 25)
    y = torch.randn(1, T) * 0.3

    for phi in (None, [0.5], [1.4576, -0.4677]):
        pyro.clear_param_store()
        em = (AdditiveEmission(sigma_y=0.3) if phi is None
              else AdditiveEmission(sigma_y=0.3, ar_coeffs=phi))
        model = SingleTraceModel(PhasicEventProcess(kernel, dt=1 / 25),
                                 TonicBaseline(dt=1 / 25), em)
        got = fitted_mean(model, y, T // 2, draws=1)
        want = model.reconstruct(y, T // 2)["mean"]
        # one guide draw against posterior means, so not identical -- but the same
        # quantity, which the conditional mean is not (it sat at 0.44 rms and
        # correlation 0.06 before the fix)
        rms = float((got - want).pow(2).mean().sqrt())
        assert rms < 0.1, f"phi={phi}: fitted_mean is not the structural mean ({rms})"
