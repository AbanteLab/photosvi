"""The z-peaks detector (Bonaventura lab prominence rule)."""
import numpy as np

from photosvi.benchmark import zpeaks_detect


def _trace(n=4000, inj=2000, peaks=(500, 1200, 2600, 3300), height=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 0.1, n)
    t = np.arange(n)
    for p in peaks:
        y += height * np.exp(-0.5 * ((t - p) / 4.0) ** 2)
    return y


def test_known_peaks_are_found_and_scored_by_prominence():
    y = _trace()
    times, prom = zpeaks_detect(y, 2000, fs=48.0)
    big = times[prom > 5]
    for p in (500, 1200, 2600, 3300):
        assert np.min(np.abs(big - p)) <= 2
    assert (prom >= 1.6).all()


def test_the_prominence_threshold_is_respected():
    y = _trace(height=0.0)
    lo = zpeaks_detect(y, 2000, fs=48.0, min_prominence=1.0)[0].size
    hi = zpeaks_detect(y, 2000, fs=48.0, min_prominence=3.0)[0].size
    assert hi < lo


def test_the_z_score_uses_the_pre_injection_segment_only():
    """Scaling the post-injection noise must not change the pre-injection calls."""
    y = _trace(height=0.0, seed=3)
    y2 = y.copy()
    y2[2000:] *= 5.0
    a = zpeaks_detect(y, 2000, fs=48.0)[0]
    b = zpeaks_detect(y2, 2000, fs=48.0)[0]
    # calls before the injection (away from its edge) are identical
    assert np.array_equal(a[a < 1900], b[b < 1900])
    # and the louder post-injection segment produces more calls
    assert (b >= 2000).sum() > (a >= 2000).sum()
