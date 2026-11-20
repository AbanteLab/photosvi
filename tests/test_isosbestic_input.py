"""Raw sensor and isosbestic channels from any file, corrected as a Doric file is.

The correction itself is tested in tests/test_preprocess.py. These check that every way of
handing over the two channels -- two arrays, two columns of a text file, two arrays of an
.npz, two columns of an .npy, a row of the table of sessions -- reaches the same correction
and keeps the injection and the handling mask aligned with the trimmed trace.
"""

from __future__ import annotations

import numpy as np
import pytest

from photosvi.api import Session, load_session, read_manifest
from photosvi.cli.manifest import load_recording
from photosvi.data.preprocess import correct_isosbestic

FS = 20.0
N = int(FS * 900)                     # 15 min: the correction fits on up to 10 before
INJ_S = 720.0


def _channels(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    iso = 2.0 + 0.05 * np.exp(-t / 300) + 0.002 * rng.normal(size=N)
    sig = 1.5 * iso + 0.3 + 0.01 * rng.normal(size=N)
    sig[int(INJ_S * FS):] += 0.05
    return sig, iso


def test_from_channels_is_the_doric_correction_with_the_injection_shifted():
    sig, iso = _channels()
    s = Session.from_channels(sig, iso, FS, inj_s=INJ_S, animal="m1")
    pp = correct_isosbestic(sig, iso, FS, inj_t=int(INJ_S * FS), smooth_green=False)
    np.testing.assert_array_equal(s.trace, np.asarray(pp["dff"], np.float32))
    assert s.inj_t == pp["inj_t"] == int(INJ_S * FS) - int(FS)      # 1 s trimmed
    assert s.metadata == {"animal": "m1"} and s.mask is None


def test_the_handling_mask_follows_the_trim():
    sig, iso = _channels()
    s = Session.from_channels(sig, iso, FS, inj_s=INJ_S, inj_end_s=INJ_S + 30,
                              mask_min_s=120)
    masked = np.flatnonzero(~s.mask)
    assert masked[0] == s.inj_t and masked.size == int(120 * FS)


@pytest.mark.parametrize("kind", ["csv", "tsv", "npz", "npy"])
def test_every_file_type_reaches_the_same_correction(tmp_path, kind):
    sig, iso = _channels()
    want = Session.from_channels(sig, iso, FS, inj_s=INJ_S)
    if kind in ("csv", "tsv"):
        sep = "," if kind == "csv" else "\t"
        p = tmp_path / f"raw.{kind}"
        np.savetxt(p, np.c_[np.arange(N) / FS, sig, iso], delimiter=sep,
                   header=sep.join(["time", "signal", "iso"]), comments="", fmt="%.10g")
        cols = dict(column="signal", isosbestic_column="iso")
    elif kind == "npz":
        p = tmp_path / "raw.npz"
        np.savez(p, signal=sig, iso=iso)
        cols = dict(column="signal", isosbestic_column="iso")
    else:
        p = tmp_path / "raw.npy"
        np.save(p, np.c_[np.arange(N) / FS, sig, iso])
        cols = dict(column=1, isosbestic_column=2)
    got = load_session(p, fs=FS, inj_s=INJ_S, **cols)
    assert got.inj_t == want.inj_t
    np.testing.assert_allclose(got.trace, want.trace, atol=1e-6)


def test_an_isosbestic_column_needs_the_sensor_column(tmp_path):
    p = tmp_path / "raw.npz"
    np.savez(p, signal=np.zeros(10), iso=np.zeros(10))
    with pytest.raises(ValueError, match="sensor channel"):
        load_session(p, fs=FS, inj_s=1, isosbestic_column="iso")


def test_the_table_and_the_command_line_take_the_columns_by_name_or_index(tmp_path):
    sig, iso = _channels()
    np.savetxt(tmp_path / "raw.csv", np.c_[np.arange(N) / FS, sig, iso], delimiter=",",
               header="time,signal,iso", comments="", fmt="%.10g")
    (tmp_path / "sessions.csv").write_text(
        "path,animal,condition,fs,inj_start,inj_end,column,isosbestic_column\n"
        f"raw.csv,m1,veh,{FS},{INJ_S},{INJ_S + 30},signal,iso\n"
        f"raw.csv,m1,drug,{FS},{INJ_S},{INJ_S + 30},1,2\n")
    (a, b), infos = read_manifest(tmp_path / "sessions.csv", mask_min_s=60)
    np.testing.assert_array_equal(a.trace, b.trace)
    np.testing.assert_array_equal(a.mask, b.mask)
    assert infos[0]["isosbestic"] and infos[0]["mask_s"] == 60
    c, _ = load_recording(tmp_path / "raw.csv", fs=FS, column="1", isosbestic_column="2",
                          inj_start_s=INJ_S, inj_end_s=INJ_S + 30, mask_min_s=60)
    np.testing.assert_array_equal(c.trace, a.trace)
