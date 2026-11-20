"""`photosvi fit` and api.load_doric_session: where the injection comes from."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from photosvi.api import load_doric_session
from photosvi.cli.main import main

BASE = "DataAcquisition/FPConsole/Signals/Series0001"
EV = "DataAcquisition/FPConsole/Events/Series0001"


def _doric(path, events=True, minutes=12, fs=48.19, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, minutes * 60, 1 / fs)
    with h5py.File(path, "w") as h:
        for k, lvl, name in (("LockInAOUT01", 1.0, "Signal (AOut1)"),
                             ("LockInAOUT02", 0.5, "Isosbestic (AOut2)")):
            h[f"{BASE}/{k}/Time"] = t
            d = h.create_dataset(f"{BASE}/{k}/AIN01", data=lvl + 0.01 * rng.standard_normal(t.size))
            d.attrs["Username"] = name
        if events:
            h[f"{EV}/Event05"] = np.array([[100.0, 110.0]])
            h[f"{EV}/Event05"].attrs["Username"] = "inj 1"
            h[f"{EV}/Event06"] = np.array([[300.0, 320.0]])
            h[f"{EV}/Event06"].attrs["Username"] = "inj 2"


def test_the_last_event_is_the_injection_when_none_is_given(tmp_path):
    p = tmp_path / "a.doric"
    _doric(p)
    s, info = load_doric_session(p)
    assert info["source"] == "doric event 'inj 2'"
    assert abs((s.inj_t + info["trim_s"] * s.fs) / s.fs - 300.0) < 0.05
    assert abs(info["mask_s"] - 20.0) < 0.1 and int((~s.mask).sum()) > 0


def test_user_times_override_the_file(tmp_path):
    p = tmp_path / "a.doric"
    _doric(p)
    s, info = load_doric_session(p, inj_start_s=250.0, inj_end_s=255.0, mask_min_s=60)
    assert info["source"] == "user" and info["inj_start_s"] == 250.0
    assert abs(info["mask_s"] - 60.0) < 0.1


def test_a_file_without_events_needs_times(tmp_path):
    p = tmp_path / "b.doric"
    _doric(p, events=False)
    with pytest.raises(ValueError, match="no injection events"):
        load_doric_session(p)
    s, info = load_doric_session(p, inj_start_s=300.0)
    assert info["source"] == "user"


def test_the_cli_fits_a_doric_file_and_records_the_injection(tmp_path):
    p = tmp_path / "c.doric"
    _doric(p)
    out = tmp_path / "run"
    main(["fit", str(p), "--out", str(out), "--steps", "3", "--max-steps", "3",
          "--threads", "1", "--device", "cpu"])
    info = json.loads((out / "injection.json").read_text())
    assert info["source"].startswith("doric event")
    assert (out / "result.json").exists()


def test_the_cli_needs_fs_and_injection_for_a_trace(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("\n".join(str(v) for v in np.random.default_rng(1).normal(0, 1, 2000)))
    with pytest.raises(SystemExit):
        main(["fit", str(p), "--out", str(tmp_path / "r")])
