"""`photosvi kernel calibrate` -- fit the transient kernel to a set of recordings.

Writes the chosen kernel as the `kernel:` block of a config, which is what the configs
and `api.FitSettings` read, and a figure of every candidate over the average isolated
transient so the choice can be checked by eye:

    photosvi kernel calibrate rec1.csv rec2.csv --fs 48 --sensor GCaMP6f \\
        --out kernel.yaml --plot kernel.pdf

The suggested kernel is the one with the lowest held-out error; `--choose` overrides it
by key (the keys are printed). See `photosvi.kernel_calibration` for the procedure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from photosvi.api import calibrate_kernel, load_session
from photosvi.kernel_calibration import SENSORS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="photosvi kernel calibrate")
    parser.add_argument("recordings", nargs="+",
                        help="corrected dF/F traces (.csv, .txt, .tsv, .npy, .npz)")
    parser.add_argument("--fs", type=float, required=True, help="sampling rate (Hz)")
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

    # The injection is irrelevant to the kernel, but a Session needs one.
    sessions = [load_session(p, fs=args.fs, inj_t=0, column=args.column)
                for p in args.recordings]
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

    block = cal.config(chosen.key)
    header = (f"# Kernel calibrated by `photosvi kernel calibrate` ({chosen.label}, "
              f"{'suggested' if chosen.key == cal.recommended else 'chosen by hand'}), "
              f"{cal.transients.n} isolated transients at {args.fs:g} Hz.\n")
    Path(args.out).write_text(header + yaml.safe_dump(block, sort_keys=False))
    print(f"wrote {args.out} ({chosen.key})")
    if args.plot:
        cal.plot(args.plot)
        print(f"wrote {args.plot}")
    if args.json:
        Path(args.json).write_text(json.dumps(cal.to_dict(), indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
