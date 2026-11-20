"""Read a Doric fiber-photometry file: sensor and isosbestic channels, and injection events.

Two layouts are read. Neuroscience Studio 6 keeps each lock-in channel as
`.../Signals/Series0001/LockInAOUT0n/AIN01` with its own `Time`, names it in the dataset's
`Username` ('Signal (AOut1)', 'Isosbestic (AOut2)'), and records keyboard events such as
'inj 1' as `(1, 2)` arrays of [start, end] seconds under `.../Events/Series0001`. Older
files keep `AIN01xAOUT0n-LockIn/Values` and `.../Time` and carry no events. When the
Studio-6 channels are unnamed, AOUT01 is taken as the sensor and AOUT02 as the
isosbestic, the assignment of the named files from the same rig.

Studio 6 samples at the console rate over the decimation (for example 12166/250 = 48.19 Hz)
with occasional longer gaps; the traces are put on a uniform grid at the median rate by
linear interpolation, so seconds and bins map the same way throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

BASE = "DataAcquisition/FPConsole/Signals/Series0001"
EVENTS = "DataAcquisition/FPConsole/Events/Series0001"


@dataclass
class DoricRecording:
    signal: np.ndarray
    isosbestic: np.ndarray
    fs: float
    events: dict = field(default_factory=dict)   # name -> (start_s, end_s)
    layout: str = ""                             # "studio6" or "legacy"


def _uniform(t: np.ndarray, *xs: np.ndarray) -> tuple:
    t = np.asarray(t, float)
    fs = 1.0 / float(np.median(np.diff(t)))
    grid = np.arange(0.0, float(t[-1]), 1.0 / fs)
    return (fs,) + tuple(np.interp(grid, t, np.asarray(x, float)) for x in xs)


def read_doric(path: str | Path) -> DoricRecording:
    with h5py.File(path, "r") as h:
        if f"{BASE}/LockInAOUT01" in h:
            named = {}
            for k in h[BASE]:
                if k.startswith("LockIn") and f"{BASE}/{k}/AIN01" in h:
                    ds = h[f"{BASE}/{k}/AIN01"]
                    named[str(ds.attrs.get("Username", "")).lower()] = k
            sig = next((k for n, k in named.items() if n.startswith("signal")), "LockInAOUT01")
            iso = next((k for n, k in named.items() if n.startswith("isosbestic")),
                       "LockInAOUT02")
            t = h[f"{BASE}/{sig}/Time"][:]
            fs, g, c = _uniform(t, h[f"{BASE}/{sig}/AIN01"][:], h[f"{BASE}/{iso}/AIN01"][:])
            layout = "studio6"
        else:
            ks = [k for k in h[BASE] if k.endswith("-LockIn")]
            sig = "AIN01xAOUT01-LockIn"
            iso = [k for k in ks if k.startswith("AIN01x") and k != sig]
            if sig not in ks or len(iso) != 1:
                raise ValueError(f"{path}: cannot tell sensor from isosbestic among {ks}")
            t = h[f"{BASE}/{sig}/Time"][:]
            fs, g, c = _uniform(t, h[f"{BASE}/{sig}/Values"][:], h[f"{BASE}/{iso[0]}/Values"][:])
            layout = "legacy"
        events = {}
        if EVENTS in h:
            for k in h[EVENTS]:
                ds = h[f"{EVENTS}/{k}"]
                v = np.asarray(ds[()], float).reshape(-1)
                if v.size >= 2:
                    events[str(ds.attrs.get("Username", k)).strip()] = (float(v[0]), float(v[1]))
    return DoricRecording(signal=g, isosbestic=c, fs=fs, events=events, layout=layout)
