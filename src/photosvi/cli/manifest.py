"""A table of sessions, one row per recording, for the commands that take several.

    path,animal,condition,inj_start,inj_end
    m1_vehicle.doric,m1,vehicle,1502.6,1520.1
    m1_drug.doric,m1,drug,,
    m2_vehicle.csv,m2,vehicle,600,640

`path` is required and is read relative to the table. The other columns are optional:
`animal` and `condition` (needed by `photosvi fit-cohort`), `inj_start` and `inj_end`
(seconds; an empty cell on a Doric file means the file's injection event), `event` (the
Doric event to use), and, for files other than Doric, `fs` and `column`. Any other column
is kept as session metadata.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

_KNOWN = {"path", "inj_start", "inj_end", "event", "fs", "column"}


def _num(v: str | None) -> float | None:
    v = (v or "").strip()
    return float(v) if v else None


def load_recording(path: Path | str, *, fs: float | None = None, column: str | None = None,
                   inj_start_s: float | None = None, inj_end_s: float | None = None,
                   event: str | None = None, mask_min_s: float = 0.0, name: str | None = None,
                   **metadata) -> tuple[Any, dict[str, Any]]:
    """One recording as a Session with its handling window masked, from any supported file.

    A Doric file is corrected to dF/F with its isosbestic channel and may take its injection
    from its events. Any other file must hold dF/F and needs `fs` and `inj_start_s`.
    """
    # Imported here: photosvi.api re-exports read_manifest, so a module-level import of
    # it would be circular.
    from photosvi.api import load_doric_session, load_session
    path = Path(path)
    if path.suffix.lower() == ".doric":
        return load_doric_session(path, inj_start_s=inj_start_s, inj_end_s=inj_end_s,
                                  event=event, mask_min_s=mask_min_s, name=name, **metadata)
    if fs is None or inj_start_s is None:
        raise ValueError(f"{path}: a trace file needs a sampling rate and an injection "
                         "start (fs, inj_start)")
    session = load_session(path, fs=fs, inj_s=inj_start_s, column=column, name=name,
                           **metadata)
    end = inj_end_s if inj_end_s is not None else inj_start_s
    lo = session.inj_t
    hi = int(round(max(end, inj_start_s + mask_min_s) * fs))
    if hi > lo:
        keep = np.ones(session.n_samples, dtype=bool)
        keep[lo:min(hi, session.n_samples)] = False
        session.mask = keep
    info = dict(source="user", inj_start_s=inj_start_s, inj_end_s=end,
                mask_s=max(0, hi - lo) / fs, fs=fs)
    return session, info


def read_manifest(table: Path | str, *, mask_min_s: float = 0.0,
                  fs: float | None = None) -> tuple[list, list[dict[str, Any]]]:
    """The sessions a table lists, and for each where its injection times came from."""
    table = Path(table)
    with table.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{table} lists no sessions")
    if "path" not in rows[0]:
        raise ValueError(f"{table} has no 'path' column; columns are {list(rows[0])}")
    sessions, infos = [], []
    for i, row in enumerate(rows, start=2):
        row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
        p = Path(row["path"])
        p = p if p.is_absolute() else table.parent / p
        meta = {k: v for k, v in row.items() if k not in _KNOWN and v}
        try:
            s, info = load_recording(p, fs=_num(row.get("fs")) or fs,
                                     column=row.get("column") or None,
                                     inj_start_s=_num(row.get("inj_start")),
                                     inj_end_s=_num(row.get("inj_end")),
                                     event=row.get("event") or None, mask_min_s=mask_min_s,
                                     **meta)
        except (ValueError, OSError) as exc:
            raise ValueError(f"{table}, line {i}: {exc}") from None
        sessions.append(s)
        infos.append(dict(path=str(p), name=s.name, **info))
    return sessions, infos


def read_kernel(path: Path | str) -> tuple[dict[str, Any], float | None]:
    """The `kernel:` block a `photosvi kernel calibrate` run wrote, and its sampling rate.

    The block gives the kernel's support in bins at the rate it was calibrated at
    (`data.fs`), so a fit at another rate rescales it.
    """
    import yaml
    doc = yaml.safe_load(Path(path).read_text()) or {}
    if "kernel" not in doc or "fit" not in doc["kernel"] or "length" not in doc["kernel"]:
        raise ValueError(f"{path} holds no kernel block (kernel: length, fit)")
    fs = (doc.get("data") or {}).get("fs")
    return doc["kernel"], (float(fs) if fs is not None else None)


def kernel_settings(path: Path | str, fs: float) -> dict[str, Any]:
    """`FitSettings` fields for a kernel file, its support rescaled to `fs`."""
    block, kfs = read_kernel(path)
    length = int(round(block["length"] / kfs * fs)) if kfs else int(block["length"])
    return dict(kernel=dict(block["fit"]), kernel_length=length)
