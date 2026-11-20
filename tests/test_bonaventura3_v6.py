"""The Studio-6 reader of the nociceptin cohort's second batch (photosvi.data.bonaventura3)."""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from photosvi.data import bonaventura3 as b


def _bump(t, t0, fwhm=0.5):
    s = fwhm / 2.355
    return np.exp(-0.5 * ((t - t0) / s) ** 2)


def test_resampling_to_48_hz_keeps_transient_heights():
    fs_raw = 12166 / 250
    t = np.arange(0, 120, 1 / fs_raw)
    x = sum(_bump(t, c) for c in np.arange(5, 115, 7.3))
    y = b.to_48(t, x)
    grid = np.arange(len(y)) / b.FS
    truth = sum(_bump(grid, c) for c in np.arange(5, 115, 7.3))
    peaks = [y[np.argmin(abs(grid - c))] for c in np.arange(5, 115, 7.3)]
    assert max(abs(p - 1) for p in peaks) < 0.01
    assert np.max(np.abs(y - truth)) < 0.01


def _v6(path, named=True, events=True):
    base = b.BASE
    t = np.arange(0, 60, 1 / 48.19)
    with h5py.File(path, "w") as h:
        for k, val in (("LockInAOUT01", 1.0), ("LockInAOUT02", 0.5)):
            h[f"{base}/{k}/Time"] = t
            ds = h.create_dataset(f"{base}/{k}/AIN01", data=np.full(t.size, val))
            if named:
                ds.attrs["Username"] = "Signal (AOut1)" if k.endswith("1") else "Isosbestic (AOut2)"
        if events:
            ev = "DataAcquisition/FPConsole/Events/Series0001"
            h[f"{ev}/Event05"] = np.array([[10.0, 12.0]])
            h[f"{ev}/Event05"].attrs["Username"] = "inj 1"
            h[f"{ev}/Event06"] = np.array([[30.0, 41.5]])
            h[f"{ev}/Event06"].attrs["Username"] = "inj 2"


def test_channels_and_events_are_read_by_name(tmp_path):
    p = tmp_path / "x.doric"
    _v6(p)
    with h5py.File(p, "r") as h:
        assert b.doric_v6(h)
        _, g, c = b.v6_trace(h)
        assert g[0] == 1.0 and c[0] == 0.5
        assert b.doric_events(h) == {"inj 1": (10.0, 12.0), "inj 2": (30.0, 41.5)}


def test_unnamed_channels_fall_back_to_the_led_outputs(tmp_path):
    p = tmp_path / "y.doric"
    _v6(p, named=False, events=False)
    with h5py.File(p, "r") as h:
        _, g, c = b.v6_trace(h)
        assert g[0] == 1.0 and c[0] == 0.5
        assert b.doric_events(h) is None


def test_a_session_without_events_or_sidecar_is_an_error(tmp_path, monkeypatch):
    p = tmp_path / "20261099VehMorphDL9_0001.doric"
    _v6(p, events=False)
    monkeypatch.setattr(b, "ROOT2", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no injection events"):
        b.load_sessions(batches=("batch2",), verbose=False)
