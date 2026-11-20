"""`photosvi points sweep` -- run one (arm, seed) cell of the guide-family grid.

One process, one cell, one JSON file. That granularity is what makes the sweep
fan out on a machine with no scheduler and resume after an interruption:

    # the whole grid, 8 workers, one thread each
    photosvi points sweep --config configs/points_guide_sweep.yaml --print-grid \\
      | xargs -P 8 -I{} sh -c 'photosvi points sweep \\
          --config configs/points_guide_sweep.yaml {} --out runs/'

    # or serially, in this process
    photosvi points sweep --config configs/points_guide_sweep.yaml --all --out runs/

Threads are pinned to 1 per worker by default (`--threads`): without it each worker
inherits torch's machine-wide default and a wide fan-out thrashes.
"""

from __future__ import annotations

import argparse
import sys
import time

from photosvi.experiments import ARMS, cell_path, load_config, pin_threads, run_cell, write_cell


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="photosvi points sweep")
    parser.add_argument("--config", required=True, help="YAML grid config")
    parser.add_argument("--arm", default=None, choices=sorted(ARMS),
                        help="which arm to run (omit with --all or --print-grid)")
    parser.add_argument("--seed", type=int, default=None, help="which seed to run")
    parser.add_argument("--out", default="runs", help="output directory for the JSON cells")
    parser.add_argument("--all", action="store_true",
                        help="run every (arm, seed) in this process, serially")
    parser.add_argument("--print-grid", action="store_true",
                        help="print one '--arm X --seed N' line per cell and exit")
    parser.add_argument("--threads", type=int, default=1,
                        help="math-backend threads per process (default 1; see module docs)")
    parser.add_argument("--force", action="store_true",
                        help="recompute cells whose output file already exists")
    parser.add_argument("--quiet", action="store_true", help="suppress per-checkpoint logging")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    arms = cfg["grid"]["arms"]
    seeds = cfg["grid"]["seeds"]

    if args.print_grid:
        for arm in arms:
            for seed in seeds:
                print(f"--arm {arm} --seed {seed}")
        return

    pin_threads(args.threads)

    if args.all:
        cells = [(a, s) for a in arms for s in seeds]
    else:
        if args.arm is None or args.seed is None:
            parser.error("give --arm and --seed, or --all, or --print-grid")
        cells = [(args.arm, args.seed)]

    for arm, seed in cells:
        out = cell_path(args.out, arm, seed)
        if out.exists() and not args.force:
            print(f"[skip] {out} exists (--force to recompute)", flush=True)
            continue
        t0 = time.time()
        print(f"[run ] arm={arm} seed={seed} -> {out}", flush=True)
        try:
            rec = run_cell(cfg, arm, seed, verbose=not args.quiet)
        except Exception as exc:                       # one bad cell must not sink the grid
            print(f"[FAIL] arm={arm} seed={seed}: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            raise
        write_cell(rec, args.out)
        print(f"[done] arm={arm} seed={seed} in {(time.time() - t0) / 60:.1f} min "
              f"(fit {rec['fit_secs'] / 60:.1f}m, held-out {rec['heldout_secs'] / 60:.1f}m)",
              flush=True)


if __name__ == "__main__":
    main()
