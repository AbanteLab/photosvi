"""Data loading and low-pass filtering for fiber photometry signals.

**Filtering is off by default, deliberately.** This module used to low-pass at 1 Hz
with an 8th-order Butterworth, a default inherited from the state-space era where the
latent of interest was slow and smoothing helped. Under the point-process model the
latent of interest is a transient with a FWHM near 0.12 s -- roughly 8 Hz of content --
so a 1 Hz low-pass deletes exactly what the model estimates.

That failure is silent. The fit does not error; it explains a smooth curve with a
point process and returns a rate ratio, and the only symptom is that suspiciously few
events are found. Anything that quietly removes the signal a model is about should not
be a default, so it is now opt-in and it complains when it would do harm.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

DEFAULT_DATA_PATH = Path(os.path.join(os.environ.get("PHOTOSVI_DATA", "data"), "bonaventura_lab/photometry/FPdata.csv.gz"))
DEFAULT_LABELS_PATH = Path(os.path.join(os.environ.get("PHOTOSVI_DATA", "data"), "bonaventura_lab/photometry/FPdata_labels.csv"))
DEFAULT_FS = 25.0

# Where the phasic component lives. The shipped K5 kernel has a FWHM near 0.12 s, so
# its energy sits in a shoulder from about 1 Hz upward; the pharmacokinetic carrier is
# below 0.01 Hz. A cutoff inside the shoulder removes transients, not noise.
TRANSIENT_BAND_HZ = (1.0, 10.0)


def load_data(
    path: str | Path = DEFAULT_DATA_PATH,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    cutoff_hz: float | None = None,
    order: int = 8,
    fs: float = DEFAULT_FS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load photometry signals and their per-session labels.

    Returns `(df_filtered, df_raw, labels)`. With `cutoff_hz=None` -- the default --
    no filter is applied and `df_filtered` is the raw frame, so the two are the same
    object and nothing has been removed behind the caller's back.

    Each column of `df_raw` is one session's full recording; `labels` has one row per
    session with at least `DRUG` and `ANIMAL ID`.
    """
    df_raw = pd.read_csv(path, header=None)
    labels = pd.read_csv(labels_path)
    if cutoff_hz is None:
        return df_raw, df_raw, labels
    df_filtered = df_raw.apply(
        lambda col: filter_signal(col.values, cutoff_hz=cutoff_hz, order=order, fs=fs),
        axis=0,
    )
    return df_filtered, df_raw, labels


def filter_signal(
    y: np.ndarray,
    cutoff_hz: float = 1.0,
    order: int = 8,
    fs: float = DEFAULT_FS,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass.

    Warns when the cutoff falls at or below the transient band, because that is the
    case where the filter removes the signal rather than the noise and the result
    still looks like a perfectly good trace.
    """
    if cutoff_hz <= TRANSIENT_BAND_HZ[0]:
        warnings.warn(
            f"low-pass at {cutoff_hz} Hz removes the transient band "
            f"({TRANSIENT_BAND_HZ[0]}-{TRANSIENT_BAND_HZ[1]} Hz): the phasic component "
            "this model estimates lives there, and a fit on the result will find "
            "almost no events without reporting an error",
            RuntimeWarning, stacklevel=2,
        )
    nyquist = fs / 2.0
    b, a = scipy_signal.butter(order, cutoff_hz / nyquist, btype="low", analog=False)
    return scipy_signal.filtfilt(b, a, y)
