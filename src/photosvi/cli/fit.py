"""`photosvi fit` -- fit one recording with the paper's single-trace model.

    photosvi fit session.doric --out run/                      # injection from the file
    photosvi fit session.doric --inj-start 1502.6 --inj-end 1520.1 --out run/
    photosvi fit trace.csv --fs 48 --inj-start 600 --inj-end 640 --mask-min-s 120 --out run/

A Doric file is corrected with its isosbestic channel to dF/F; any other file must already
hold dF/F. The model has one injection: the times given (or, for a Doric file without them,
the file's last injection event, or `--event`) are the one the rate contrast is read at.
`--inj-end` closes the handling window excluded from the likelihood; `--mask-min-s`
extends it to at least that many seconds after the injection. Times given on the command
line override the file's events. Writes result.json, events.csv, curves.npz and
injection.json (where the times came from) to `--out`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from photosvi.api import FitSettings, fit_session, load_doric_session, load_session


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="photosvi fit")
    p.add_argument("recording", help=".doric, or a dF/F trace (.csv, .txt, .tsv, .npy, .npz)")
    p.add_argument("--fs", type=float, default=None,
                   help="sampling rate in Hz (required unless the file is .doric)")
    p.add_argument("--column", default=None, help="which column holds the trace")
    p.add_argument("--inj-start", type=float, default=None, help="injection start (s)")
    p.add_argument("--inj-end", type=float, default=None,
                   help="end of the handling window (s); masked from the likelihood")
    p.add_argument("--mask-min-s", type=float, default=0.0,
                   help="mask at least this many seconds after the injection start")
    p.add_argument("--event", default=None,
                   help="Doric event to use as the injection when no times are given "
                        "(default: the last one recorded)")
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--steps", type=int, default=None, help="minimum steps (default: shipped)")
    p.add_argument("--max-steps", type=int, default=None)
    a = p.parse_args(argv)

    path = Path(a.recording)
    if path.suffix.lower() == ".doric":
        session, info = load_doric_session(path, inj_start_s=a.inj_start,
                                           inj_end_s=a.inj_end, event=a.event,
                                           mask_min_s=a.mask_min_s)
    else:
        if a.fs is None or a.inj_start is None:
            p.error("a trace file needs --fs and --inj-start")
        session = load_session(path, fs=a.fs, inj_s=a.inj_start, column=a.column)
        end = a.inj_end if a.inj_end is not None else a.inj_start
        lo = session.inj_t
        hi = int(round(max(end, a.inj_start + a.mask_min_s) * a.fs))
        if hi > lo:
            keep = np.ones(session.n_samples, dtype=bool)
            keep[lo:min(hi, session.n_samples)] = False
            session.mask = keep
        info = dict(source="user", inj_start_s=a.inj_start, inj_end_s=end,
                    mask_s=max(0, hi - lo) / a.fs, fs=a.fs)
    print(f"injection at {info['inj_start_s']:.2f} s ({info['source']}), "
          f"{info['mask_s']:.1f} s masked, {session.fs:.2f} Hz, "
          f"{session.n_samples / session.fs / 60:.1f} min")
    settings = FitSettings(device=a.device, threads=a.threads, steps=a.steps,
                           max_steps=a.max_steps)
    res = fit_session(session, settings)
    out = res.save(a.out)
    (out / "injection.json").write_text(json.dumps(info, indent=1, default=float))
    rr = res.rate_ratio
    print(f"rate ratio {rr['estimate']:.3f} [{rr['lo']:.3f}, {rr['hi']:.3f}] (90%); "
          f"wrote {out}")


if __name__ == "__main__":
    main()
