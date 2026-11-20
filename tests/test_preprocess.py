"""The preprocessing port, tested on signals whose answer is known analytically.

There is no `.doric` file in this project, so none of this is checked against the
MATLAB it was ported from. What it checks instead is that each step does the thing its
name claims on inputs constructed so the right answer is derivable -- which catches a
transcription error but not a misreading of the original. The comparison against the
vendor script is owed and is not these tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from photosvi.data.preprocess import (
    als_baseline,
    correct_isosbestic,
    downsample,
    moving_average,
    savgol_seconds,
)

FS = 100.0


def test_moving_average_of_a_constant_is_that_constant_away_from_the_ends():
    y = np.full(500, 3.0)
    out = moving_average(y, 11)
    assert np.allclose(out[10:-10], 3.0)


def test_moving_average_ends_are_biased_low_which_is_why_the_pipeline_trims():
    y = np.full(500, 3.0)
    out = moving_average(y, 51)
    assert out[0] < 2.0, "the ends convolve against zeros, as in MATLAB 'same'"


def test_savgol_preserves_a_quadratic_exactly():
    """An order-2 filter reproduces an order-2 signal; that is the definition."""
    t = np.arange(1000) / FS
    y = 1.0 + 2.0 * t + 3.0 * t**2
    out = savgol_seconds(y, FS, 0.5, order=2)
    assert np.allclose(out, y, atol=1e-6)


def test_savgol_refuses_a_window_too_short_for_its_polynomial():
    y = np.zeros(100)
    with pytest.raises(ValueError, match="cannot carry an order"):
        savgol_seconds(y, FS, 0.01)


def test_als_recovers_a_smooth_trend_under_one_sided_peaks():
    """The case it exists for: a drifting baseline with positive events on top."""
    t = np.arange(4000) / FS
    trend = 10.0 + 0.5 * np.sin(2 * np.pi * t / 40.0)
    y = trend.copy()
    for i in range(200, 4000, 200):
        y[i:i + 5] += 3.0
    base = als_baseline(y, lam=FS * 1e6, p=0.01)
    # it should track the trend, not the peaks
    assert np.abs(base - trend).max() < 0.35
    assert base.mean() < y.mean(), "a one-sided signal must sit above its baseline"


def test_als_of_a_straight_line_is_that_line():
    y = np.linspace(0.0, 10.0, 2000)
    base = als_baseline(y, lam=FS * 1e6, p=0.5)
    assert np.abs(base - y).max() < 1e-3


def test_downsample_block_mean_matches_a_hand_computation():
    y = np.arange(12, dtype=float)
    out = downsample(y, 6.0, 2.0)
    assert np.allclose(out, [1.0, 4.0, 7.0, 10.0])


def test_downsample_refuses_a_non_integer_ratio():
    with pytest.raises(ValueError, match="not an integer"):
        downsample(np.zeros(100), 30.0, 25.0)


def test_downsample_refuses_to_upsample():
    with pytest.raises(ValueError, match="cannot upsample"):
        downsample(np.zeros(100), 25.0, 50.0)


def _two_channel(n=120_000, fs=FS, drift=True):
    """A synthetic pair where the true dF/F is known by construction."""
    t = np.arange(n) / fs
    # the nuisance every fiber recording has: bleaching plus slow movement
    nuisance = 1000.0 * np.exp(-t / 3000.0) + (20.0 * np.sin(2 * np.pi * t / 120.0)
                                               if drift else 0.0)
    iso = nuisance / 2.0 + 50.0              # the control, on its own scale
    true_dff = np.zeros(n)
    for i in range(1000, n, 1000):
        true_dff[i:i + 20] = 0.05
    green = nuisance * (1.0 + true_dff)
    return green, iso, true_dff


def test_the_pipeline_divides_out_a_nuisance_the_control_carries():
    """The whole point: what both channels share must not survive into dF/F."""
    green, iso, true_dff = _two_channel()
    out = correct_isosbestic(green, iso, FS, inj_t=80_000, smooth_green=False)
    got = out["dff"]
    trim = int(FS)
    want = true_dff[trim:green.size - trim]
    # the events survive
    assert got[want > 0].mean() > 0.03
    # and the nuisance does not: away from events the residual is small against the
    # 0.05 the events carry
    assert np.abs(got[want == 0]).mean() < 0.01


def test_the_control_is_fitted_on_the_pre_injection_window_almost_only():
    """A post-injection response barely moves the scaling that defines F0.

    Barely, not at all, and the gap is a property of the original pipeline rather
    than of this port. The regression is fitted on the pre-injection slice of the ALS
    BASELINE, and that baseline is estimated over the whole recording with a
    second-difference penalty -- which couples every sample to every other. So a drug
    response leaks backwards through the smoothness penalty into the window that is
    supposed to be drug-free.

    Measured here at a relative 1e-8 for a response that multiplies the signal by
    1.6, which is nothing. It is pinned because the leak exists rather than because
    it matters: if a future change made the ALS window match the fit window, this
    would become exact, and if one made it looser the number would grow.
    """
    green, iso, _ = _two_channel()
    inj = 80_000
    ramped = green.copy()
    ramped[inj:] *= np.linspace(1.0, 1.6, green.size - inj)   # a big drug response
    a_plain = correct_isosbestic(green, iso, FS, inj_t=inj,
                                 smooth_green=False)["slope"]
    a_ramped = correct_isosbestic(ramped, iso, FS, inj_t=inj,
                                  smooth_green=False)["slope"]
    assert a_plain == pytest.approx(a_ramped, rel=1e-6), (
        "the isosbestic scaling moved more than the ALS coupling can explain")


def test_inj_t_comes_back_shifted_by_the_trim():
    green, iso, _ = _two_channel()
    out = correct_isosbestic(green, iso, FS, inj_t=80_000, smooth_green=False)
    assert out["inj_t"] == 80_000 - int(FS)


def test_channels_of_different_length_are_refused():
    with pytest.raises(ValueError, match="differ in length"):
        correct_isosbestic(np.zeros(100), np.zeros(90), FS, inj_t=50)


def test_a_recording_shorter_than_the_flat_zone_is_refused():
    green, iso, _ = _two_channel(n=20_000)
    with pytest.raises(ValueError, match="pre-injection window"):
        correct_isosbestic(green, iso, FS, inj_t=10_000, smooth_green=False)


def test_smoothing_the_biosensor_is_optional_and_changes_the_result():
    """It is offered because the original does it, and defaulted on for that reason.

    It also removes evidence this package's model would otherwise use, so the two
    must be distinguishable rather than silently the same.
    """
    green, iso, _ = _two_channel()
    on = correct_isosbestic(green, iso, FS, inj_t=80_000, smooth_green=True)["dff"]
    off = correct_isosbestic(green, iso, FS, inj_t=80_000, smooth_green=False)["dff"]
    assert not np.allclose(on, off)
    assert on.std() < off.std(), "smoothing should reduce, not increase, variance"


def test_the_f0_window_stops_at_the_injection():
    """The isosbestic map must be fitted before the drug, not merely for `flatzone_s`.

    The window used to be the first `flatzone_s` seconds unconditionally, with `inj_t`
    only bounds-checked -- correct for a recording injected after ten minutes and silently
    wrong for one injected before, which is the case the app will meet. On a real trace an
    injection at 5 min moves the fitted slope from 0.92 to 1.14 and the dF/F by a fifth of
    its own standard deviation.
    """
    import numpy as np

    from photosvi.data.preprocess import correct_isosbestic

    fs = 48.0
    n = int(40 * 60 * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    # a control channel with slow structure, and a green channel that follows it until an
    # injection at 5 min and then departs
    con = 100.0 + 5.0 * np.sin(2 * np.pi * t / 900.0) + rng.normal(0, 0.2, n)
    green = 1.0 * con.copy()
    inj = int(5 * 60 * fs)
    green[inj:] += 20.0 * (1 - np.exp(-(t[inj:] - t[inj]) / 300.0))

    early = correct_isosbestic(green, con, fs, inj_t=inj, smooth_green=False)
    late = correct_isosbestic(green, con, fs, inj_t=int(35 * 60 * fs),
                              smooth_green=False)   # inside the trim
    # told where the injection is, the map is fitted before it and differs from the
    # version that ran past it
    assert abs(early["slope"] - late["slope"]) > 1e-3

    # and a window with nothing in it is refused rather than fitted on two points
    from photosvi.data.preprocess import TRIM_S
    just_inside = int(round(TRIM_S * fs)) + 2
    with pytest.raises(ValueError, match="pre-injection window"):
        correct_isosbestic(green, con, fs, inj_t=just_inside, smooth_green=False)
