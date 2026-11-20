"""`photosvi fit-cohort` -- fit a cohort jointly with the hierarchical model of the paper.

    photosvi fit-cohort sessions.csv --reference vehicle --mask-min-s 120 --out run/

`sessions.csv` lists one recording per row, with its animal and its condition; see
`photosvi.cli.manifest` for the columns. The command prints the contrast of each condition
against `--reference`, as a rate ratio with its 90% interval, and writes result.json,
curves.npz and sessions.json (where each injection time came from) to `--out`.
"""
from __future__ import annotations

import argparse
import json

from photosvi.api import CohortSettings, Study, fit_cohort
from photosvi.cli.manifest import read_kernel, read_manifest


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="photosvi fit-cohort")
    p.add_argument("sessions", help="table of sessions (.csv): path, animal, condition, ...")
    p.add_argument("--reference", default=None,
                   help="condition the others are compared against (default: the first "
                        "in alphabetical order)")
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--mask-min-s", type=float, default=0.0,
                   help="mask at least this many seconds after each injection start")
    p.add_argument("--fs", type=float, default=None,
                   help="sampling rate of trace files that do not give one in the table")
    p.add_argument("--pre-s", type=float, default=600.0,
                   help="seconds kept before the injection (default 600)")
    p.add_argument("--kernel", default=None,
                   help="kernel file from `photosvi kernel calibrate` (default: dLight1.3b)")
    p.add_argument("--device", default="auto")
    p.add_argument("--min-steps", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--quiet", action="store_true", help="no progress lines")
    a = p.parse_args(argv)

    try:
        sessions, infos = read_manifest(a.sessions, mask_min_s=a.mask_min_s, fs=a.fs)
    except ValueError as exc:
        p.error(str(exc))
    for s, info in zip(sessions, infos, strict=True):
        print(f"  {s.name}: {s.metadata.get('animal', '?')}, "
              f"{s.metadata.get('condition', s.metadata.get('drug', '?'))}, injection at "
              f"{info['inj_start_s']:.1f} s ({info['source']}), {info['mask_s']:.0f} s masked")
    st = CohortSettings(device=a.device, pre_s=a.pre_s)
    if a.kernel:
        try:
            st.kernel, st.kernel_fs = read_kernel(a.kernel)
        except (ValueError, OSError) as exc:
            p.error(str(exc))
    if a.min_steps is not None:
        st.min_steps = a.min_steps
    if a.max_steps is not None:
        st.max_steps = a.max_steps

    def progress(d):
        if not a.quiet:
            print(f"    {d['step']:>6}/{d['total']}  " + "  ".join(
                f"{c} {v[0]:+.3f} +- {v[1]:.3f}" for c, v in d["contrasts"].items()),
                flush=True)

    try:
        res = fit_cohort(Study(sessions), st, reference=a.reference, on_progress=progress)
    except ValueError as exc:
        p.error(str(exc))
    out = res.save(a.out)
    (out / "sessions.json").write_text(json.dumps(infos, indent=1, default=str))
    for c in res.conditions[1:]:
        r = res.contrast(c)
        print(f"{c} vs {res.reference}: rate ratio {r['ratio']:.3f} "
              f"[{r['ratio_lo']:.3f}, {r['ratio_hi']:.3f}] (90%)")
    print(f"design: {res.design['n_animals']} animals, {res.design['n_sessions']} sessions, "
          f"{'within' if res.design['within_animal'] else 'between'}-animal contrast; "
          f"wrote {out}")


if __name__ == "__main__":
    main()
