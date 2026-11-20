"""The tonic summaries recorded by the detection benchmark (scripts/detection_tonic.py)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from detection_tonic import downsample, lowpass, summarize, tonic_record  # noqa: E402

from photosvi.models.points.tonic import TonicBaseline

FS, T, INJ = 48.0, 48 * 3600, 48 * 600


def _true():
    tb = TonicBaseline(dt=1 / FS, b=-0.34, peak=3.94, k_rise=0.005, k_decay=0.0009)
    return tb, tb.baseline(T, INJ).detach().double().numpy()


def test_summary_of_a_bateman_curve_is_its_peak_and_peak_time():
    tb, x = _true()
    s = summarize(x, INJ, FS)
    assert s["pre_level"] == pytest.approx(-0.34)
    assert s["peak_height"] == pytest.approx(3.94, rel=1e-4)
    assert s["peak_time_s"] == pytest.approx(float(tb.peak_time), abs=1.0)  # float32 curve, flat at its top


def test_lowpass_is_causal_and_passes_the_slow_curve():
    _, x = _true()
    lp = lowpass(x, FS)
    s = summarize(lp, INJ, FS)
    assert s["peak_height"] == pytest.approx(3.94, rel=0.01)
    # causal: nothing moves before the injection once the start-up transient is over
    assert np.allclose(lp[int(300 * FS):INJ], -0.34, atol=1e-6)


def test_record_keeps_four_curves_at_one_second():
    _, x = _true()
    rng = np.random.default_rng(0)
    y = x + rng.normal(0, 1, x.size)
    r = tonic_record(y, x, x, {"P": 3.94}, {"P": 3.94}, INJ, FS)
    assert set(r["curves"]) == {"true", "photosvi", "lowpass", "als", "data"}
    assert all(len(c) == T // 48 for c in r["curves"].values())
    assert r["summary"]["photosvi"] == r["summary"]["true"]
    assert downsample(np.arange(96.0), FS) == [0.0, 48.0]
