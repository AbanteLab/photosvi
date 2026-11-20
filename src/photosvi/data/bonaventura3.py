"""Loader for the third batch: five animals, nociceptin or saline, then morphine.

The recordings live outside the repository and their layout is not something the rest of
the package should know about -- two group folders, a `.doric` per session with its
channel names varying between sessions, and a `.mat` sidecar carrying the injection times.
Both the cohort fit and the paper's calibration numbers need exactly the same ten windows,
so the loading lives here rather than in either script: a second copy would drift, and the
two would then be describing different data while reporting the same dataset.

The second batch (2026-10, animals DL4 and DL5) was acquired with Doric Neuroscience Studio
6.7, whose files lay the lock-in channels out differently (`LockInAOUT01/AIN01` with its own
`Time`), sample at 12166/250 = 48.19 Hz, and record the injections as events ('inj 1',
'inj 2'). Both layouts are read here; the second batch is resampled to 48 Hz, the rate the
kernel and the autoregressive coefficients were calibrated at.

Injection times come from the Doric events when the file has them, otherwise from the
sidecars in seconds; a session with neither is an error. The two vehicle sessions of the
second batch have no recorded events and their sidecars carry the protocol times
(620/640/1541/1550 s, identical in both): they are tagged `mat-protocol` so the Methods can
say so. `INJ2` is the morphine injection in
every session -- the one the analysis window is anchored to -- and is the second of two
in the nociceptin group and the only one in the saline group, which is why it is read by
name rather than by position.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import h5py
import numpy as np
import scipy.io as sio
from scipy.signal import detrend

from photosvi.data.preprocess import correct_isosbestic

FS = 48.0
PRE_S = 600.0
# Smoothness of the ALS baseline, per unit of sampling rate: lambda = ALS_LAMBDA_PER_FS x fs.
# 1e6 is the Bonaventura lab's current pipeline (since 2026-10-21; it was 1e4). The simulator's
# calibration was measured at 1e4, so the calibration generators pass that explicitly.
ALS_LAMBDA_PER_FS = 1e6
BASE = "DataAcquisition/FPConsole/Signals/Series0001"
ROOT = os.path.join(os.environ.get("PHOTOSVI_DATA", "data"),
                    "bonaventura_lab/photometry/batch3_nociceptin/3 - Third batch - Nociceptine")
# The folder name is the condition: the group is not recoverable from the signal and
# guessing it from the number of tonic steps got it wrong for four of five animals.
GROUPS = (("noci", "Noci + Morphine Files/.doric", "Noci + Morphine Files/.mat"),
          ("sali", "Morphine Only Files/.doric Files", "Morphine Only Files/.mat Files"))
# The second batch: one flat folder, .doric and .mat side by side, condition in the name.
ROOT2 = os.path.join(os.environ.get("PHOTOSVI_DATA", "data"), "bonaventura_lab/photometry/batch3_nociceptin/batch2_2026-10")
COND_BY_NAME = (("ro64", "noci"), ("noci", "noci"), ("veh", "sali"), ("sali", "sali"))
# Sessions whose sidecar times are the protocol schedule, not a measurement (2026-10-21:
# no events were recorded in these two).
PROTOCOL_TIMES = {"20261015VehMorphDL4_0002", "20261015VehMorphDL5_0001"}
BATCHES = ("batch1", "batch2")


@dataclass
class Session:
    """One recording, already corrected, windowed and detrended."""

    animal: str
    cond: str
    y: np.ndarray
    iso: str
    inj_bin: int
    mask_s: list = field(default_factory=list)
    mask_min_s: float = 0.0
    batch: str = "batch1"
    source: str = ""            # the .doric file
    inj_source: str = ""        # doric-event | mat | mat-protocol
    fs_raw: float = FS          # the file's own sampling rate before any resampling

    @property
    def mask_bins(self) -> list:
        """The handling window, in bins of `y`.

        Delivering the injection moves the fibre and the animal, so these bins carry an
        artefact shaped like a transient at exactly the moment the rate is meant to
        change. They are excluded from the likelihood rather than modelled.
        """
        i2, e2 = self.mask_s
        return [self.inj_bin, self.inj_bin + int(max(e2 - i2, self.mask_min_s) * FS)]


def channels(h: h5py.File) -> tuple:
    """The signal and isosbestic channel names of one file.

    The isosbestic is whichever other `AIN01x` channel is present -- AOUT03 in seven
    sessions and AOUT02 in three -- so it is found by elimination, not by name. `AIN02x`
    is a disconnected red channel and is never a candidate.
    """
    ks = [k for k in h[BASE] if "LockIn" in k]
    sig = "AIN01xAOUT01-LockIn"
    iso = [k for k in ks if k.startswith("AIN01x") and k != sig]
    if len(iso) != 1:
        raise ValueError(f"cannot tell which channel is the isosbestic among {ks}")
    return sig, iso[0]


def injections(matdir: str, animal: str) -> list:
    """`[INJ1start, INJ1end, INJ2start, INJ2end]` in seconds, from the `.mat` sidecar."""
    for f in glob.glob(matdir + "/*.mat"):
        if animal.lower() in os.path.basename(f).lower():
            d = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
            return [int(d[k]) for k in ("INJ1start", "INJ1end", "INJ2start", "INJ2end")]
    raise FileNotFoundError(f"no .mat for {animal} in {matdir}")


def doric_v6(h: h5py.File) -> bool:
    """Neuroscience Studio 6 layout: `LockInAOUT0n/AIN01` groups with their own `Time`."""
    return f"{BASE}/LockInAOUT01" in h


def v6_trace(h: h5py.File) -> tuple:
    """`(time, signal, isosbestic)` of a Studio-6 file, read by the channel usernames."""
    pairs = {}
    for k in h[BASE]:
        if not k.startswith("LockIn"):
            continue
        ds = h[f"{BASE}/{k}/AIN01"]
        pairs[str(ds.attrs.get("Username", k))] = (h[f"{BASE}/{k}/Time"][:], ds[:])
    sig = [v for n, v in pairs.items() if n.lower().startswith("signal")]
    iso = [v for n, v in pairs.items() if n.lower().startswith("isosbestic")]
    if not sig and not iso and f"{BASE}/LockInAOUT02" in h:
        # Unnamed channels (the 2026-10-15 sessions): AOUT01 drives the 470-nm LED and AOUT02
        # the 405-nm one, as in the named sessions of the same rig and batch, whose lock-in
        # reference frequencies (208.6 and 572.2 Hz) these files share.
        sig = [(h[f"{BASE}/LockInAOUT01/Time"][:], h[f"{BASE}/LockInAOUT01/AIN01"][:])]
        iso = [(h[f"{BASE}/LockInAOUT02/Time"][:], h[f"{BASE}/LockInAOUT02/AIN01"][:])]
    if len(sig) != 1 or len(iso) != 1:
        raise ValueError(f"cannot tell signal from isosbestic among {sorted(pairs)}")
    (t, g), (_, c) = sig[0], iso[0]
    return t, g, c


def doric_events(h: h5py.File) -> dict | None:
    """`{'inj 1': (start, end), 'inj 2': (start, end)}` from the file, or None if absent."""
    base = "DataAcquisition/FPConsole/Events/Series0001"
    if base not in h:
        return None
    out = {}
    for k in h[base]:
        ds = h[f"{base}/{k}"]
        name = str(ds.attrs.get("Username", k)).strip().lower()
        v = np.asarray(ds[()], float).reshape(-1)
        if v.size >= 2:
            out[name] = (float(v[0]), float(v[1]))
    return out or None


def to_48(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation from the file's time stamps onto k / FS, k = 0, 1, ...

    The file samples at 48.19 Hz with occasional longer gaps; the grid is the one every other
    session uses (sample k at k / 48 s), so seconds map to bins identically in both batches.
    At a 0.5-s transient and a 0.02-s step the interpolation changes peak heights by well
    under a percent (checked in tests/test_bonaventura3_v6.py).
    """
    grid = np.arange(0.0, float(t[-1]), 1.0 / FS)
    return np.interp(grid, np.asarray(t, float), np.asarray(x, float))


def _session_files(batches) -> list:
    """`(batch, cond, doric, matdir_or_matfile)` for every session of the given batches."""
    out = []
    if "batch1" in batches:
        for cond, dsub, msub in GROUPS:
            for f in sorted(glob.glob(f"{ROOT}/{dsub}/*.doric")):
                out.append(("batch1", cond, f, f"{ROOT}/{msub}"))
    if "batch2" in batches:
        for f in sorted(glob.glob(f"{ROOT2}/*.doric")):
            low = os.path.basename(f).lower()
            cond = next((c for k, c in COND_BY_NAME if k in low), None)
            if cond is None:
                raise ValueError(f"cannot tell the condition of {f} from its name")
            out.append(("batch2", cond, f, f[:-len(".doric")] + ".mat"))
    return out


def _mat_injections(mat: str) -> list:
    d = sio.loadmat(mat, squeeze_me=True, struct_as_record=False)
    return [float(d[k]) for k in ("INJ1start", "INJ1end", "INJ2start", "INJ2end")]


def load_sessions(pre_s: float = PRE_S, root: str = ROOT, verbose: bool = True, *,
                  detrend_window: bool = True, mask_min_s: float = 0.0,
                  batches=("batch1",), als_lambda_per_fs: float = ALS_LAMBDA_PER_FS) -> list:
    """Every session, windowed from `pre_s` before the morphine injection to the end.

    The window is the span over which a pretreatment can change the morphine response.
    `inj_bin` is where the injection falls inside the returned `y`, which is `pre_s * FS`
    by construction; the trace is linearly detrended over the window so the slow drift the
    tonic component is not meant to absorb does not reach it.

    `detrend_window=False` skips that detrend. Over a window that is flat before the
    injection and rises after it, the fitted line tilts the pre-injection segment down and
    digs a trough at the injection -- measured on this cohort, a group-mean dip of 0.02
    dF/F where the undetrended trace dips 0.004 -- exactly where the contrast is read, so
    the cohort fit leaves the slow trend to the model's per-session drift instead
    (2026-10-08). The default keeps the calibration generators on what they were run with.

    `mask_min_s` extends each handling window to at least that many seconds after the
    morphine injection: the sensor and the isosbestic channels both dip for one to two
    minutes after it, longer than the 15-80 s recorded as handling.

    `batches` selects the first batch (default, the five animals the simulator was calibrated
    on) and/or the second; `als_lambda_per_fs` sets the ALS smoothness (lambda = value x fs).
    """
    if root != ROOT:
        raise ValueError("root is fixed per batch; see ROOT and ROOT2")
    out = []
    for batch, cond, f, matref in _session_files(batches):
        stem = os.path.basename(f)[:-len(".doric")]
        animal = re.search(r"DL(\d+)", stem, re.I).group(0).upper()
        with h5py.File(f, "r") as h:
            if doric_v6(h):
                t, g, c = v6_trace(h)
                fs_raw = 1.0 / float(np.median(np.diff(t)))
                g, c = to_48(t, g), to_48(t, c)
                iso = "LockInAOUT02"
                ev = doric_events(h)
            else:
                sig, iso = channels(h)
                g, c = h[f"{BASE}/{sig}/Values"][:], h[f"{BASE}/{iso}/Values"][:]
                fs_raw, ev = FS, None
        if ev and "inj 2" in ev:
            i2, e2 = ev["inj 2"]
            inj_source = "doric-event"
        else:
            try:
                times = (injections(matref, animal) if os.path.isdir(matref)
                         else _mat_injections(matref))
            except (FileNotFoundError, OSError) as err:
                raise FileNotFoundError(
                    f"{f}: no injection events in the file and no sidecar ({err})") from None
            i2, e2 = times[2], times[3]
            inj_source = "mat-protocol" if stem in PROTOCOL_TIMES else "mat"
        pp = correct_isosbestic(g, c, FS, inj_t=int(i2 * FS), smooth_green=False,
                                als_lambda=als_lambda_per_fs * FS)
        dff = np.asarray(pp["dff"], float)
        # `dff` starts after the trim, so the injection sits at pp["inj_t"], not at i2 * FS.
        # Until 2026-10-21 the window was cut at (i2 - pre_s) * FS, which put the injection
        # (and the masked handling window) one trim -- 1 s -- later than inj_bin said.
        y = dff[int(pp["inj_t"]) - int(pre_s * FS):]
        if detrend_window:
            y = detrend(y, type="linear")
        out.append(Session(animal=animal, cond=cond, y=y, iso=iso,
                           inj_bin=int(pre_s * FS), mask_s=[i2, e2],
                           mask_min_s=float(mask_min_s), batch=batch, source=f,
                           inj_source=inj_source, fs_raw=float(fs_raw)))
        if verbose:
            print(f"  {animal} {cond} [{batch}, {inj_source}, {fs_raw:.2f} Hz]: "
                  f"{len(y)/FS/60:.1f} min, iso {iso.replace('-LockIn','')}, "
                  f"mask {e2-i2:.0f} s", flush=True)
    return out
