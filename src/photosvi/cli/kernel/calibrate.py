"""`photosvi kernel calibrate` -- fit the transient kernel to a set of recordings.

Writes the chosen kernel as the `kernel:` block of a config, which is what the configs
and `api.FitSettings` read, and a figure of every candidate over the average isolated
transient so the choice can be checked by eye:

    photosvi kernel calibrate rec1.doric rec2.doric --mask-min-s 120 \\
        --out kernel.yaml --plot kernel.pdf
    photosvi kernel calibrate rec1.csv rec2.csv --fs 48 --sensor GCaMP6f --out kernel.yaml
    photosvi kernel calibrate --sessions sessions.csv --mask-min-s 120 --out kernel.yaml

A Doric file is corrected to dF/F with its isosbestic channel, and the handling window
after its injection event is left out, so artefacts there are not taken for transients.
Trace files must hold dF/F. `--sessions` takes the table of `photosvi fit-cohort`, which
gives each file its own injection times.

The suggested kernel is the one with the lowest held-out error; `--choose` overrides it
by key (the keys are printed). See `photosvi.kernel_calibration` for the procedure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from photosvi.api import Session, calibrate_kernel, load_doric_session, load_session
from photosvi.cli.manifest import read_manifest
from photosvi.kernel_calibration import SENSORS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="photosvi kernel calibrate")
    parser.add_argument("recordings", nargs="*",
                        help=".doric files, or dF/F traces (.csv, .txt, .tsv, .npy, .npz)")
    parser.add_argument("--sessions", default=None,
                        help="a table of sessions (.csv), as for `photosvi fit-cohort`")
    parser.add_argument("--fs", type=float, default=None,
                        help="sampling rate in Hz (required for trace files)")
    parser.add_argument("--mask-min-s", type=float, default=0.0,
                        help="leave out at least this many seconds after each injection")
    parser.add_argument("--column", default=None, help="which column holds the trace")
    parser.add_argument("--sensor", default="dLight1.3b", choices=sorted(SENSORS),
                        help="starting values for the fit (default dLight1.3b)")
    parser.add_argument("--choose", default=None,
                        help="candidate key to write instead of the suggested one")
    parser.add_argument("--out", default="kernel.yaml", help="the kernel config block")
    parser.add_argument("--plot", default=None, help="figure of the candidates (.pdf/.png)")
    parser.add_argument("--json", default=None, help="the full calibration as JSON")
    parser.add_argument("--min-transients", type=int, default=20)
    parser.add_argument("--isolation-s", type=float, default=4.0,
                        help="no larger peak within this many seconds (default 4)")
    args = parser.parse_args(argv)

    if bool(args.recordings) == bool(args.sessions):
        parser.error("give the recordings, or --sessions, but not both")
    try:
        if args.sessions:
            sessions, _ = read_manifest(args.sessions, mask_min_s=args.mask_min_s,
                                        fs=args.fs)
        else:
            sessions = [_one(Path(p), args) for p in args.recordings]
    except ValueError as exc:
        parser.error(str(exc))
    cal = calibrate_kernel(sessions, sensor=args.sensor,
                           min_transients=args.min_transients,
                           isolation_s=args.isolation_s)
    try:
        chosen = cal.candidate(args.choose)
    except KeyError as exc:
        sys.exit(str(exc))

    print(f"{cal.transients.n} isolated transients from {len(sessions)} recording(s); "
          f"{cal.folds} folds")
    print(f"{'key':18} {'held-out':>9} {'peak':>7} {'FWHM':>7}  label")
    for c in cal.candidates:
        mark = "*" if c.key == cal.recommended else " "
        print(f"{mark}{c.key:17} {c.heldout_rms:9.4f} {c.shape['peak_s']:6.3f}s "
              f"{c.shape['fwhm_s']:6.3f}s  {c.label}")
    for w in cal.warnings:
        print(f"note: {w}")

    block = dict(cal.config(chosen.key), data=dict(fs=float(sessions[0].fs)))
    header = (f"# Kernel calibrated by `photosvi kernel calibrate` ({chosen.label}, "
              f"{'suggested' if chosen.key == cal.recommended else 'chosen by hand'}), "
              f"{cal.transients.n} isolated transients at {sessions[0].fs:g} Hz.\n")
    Path(args.out).write_text(header + yaml.safe_dump(block, sort_keys=False))
    print(f"wrote {args.out} ({chosen.key})")
    if args.plot:
        cal.plot(args.plot)
        print(f"wrote {args.plot}")
    if args.json:
        Path(args.json).write_text(json.dumps(cal.to_dict(), indent=1))
        print(f"wrote {args.json}")


def _one(path: Path, args) -> Session:
    """A recording for calibration. Its injection matters only for the handling mask."""
    if path.suffix.lower() == ".doric":
        try:
            return load_doric_session(path, mask_min_s=args.mask_min_s)[0]
        except ValueError as exc:
            raise ValueError(f"{exc}; or list the file in a --sessions table with its "
                             "injection times") from None
    if args.fs is None:
        raise ValueError(f"{path}: a trace file needs --fs")
    return load_session(path, fs=args.fs, inj_t=0, column=args.column)


if __name__ == "__main__":
    main()
