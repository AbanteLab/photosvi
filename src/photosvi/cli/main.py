"""`photosvi` — single-entry CLI with per-family subcommands.

Usage:
    photosvi fit session.doric --out run/
    photosvi fit-cohort sessions.csv --reference vehicle --out run/
    photosvi kernel calibrate rec1.doric rec2.doric --out kernel.yaml
    photosvi points sweep --config configs/points_guide_sweep.yaml --all --out runs/

The dispatcher parses only `<family> <command>`; remaining argv is forwarded
to the leaf's own argparse parser, so leaf-specific flags (`--config`,
`--device`, etc.) are owned by the leaf.
"""

from __future__ import annotations

import argparse
import importlib

# (family, command) -> dotted module path. Each target module exposes a
# `main(argv: list[str] | None = None) -> None`.
_COMMANDS: dict[tuple[str, str], str] = {
    ("points", "sweep"): "photosvi.cli.points.sweep",
    ("kernel", "calibrate"): "photosvi.cli.kernel.calibrate",
    ("fit", None): "photosvi.cli.fit",
    ("fit-cohort", None): "photosvi.cli.fit_cohort",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photosvi",
        description="Dynamic generative models of fiber-photometry signals.",
    )
    family = parser.add_subparsers(dest="family", required=True, metavar="FAMILY")


    points = family.add_parser("points", help="Marked point-process family")
    points_sub = points.add_subparsers(dest="command", required=True, metavar="COMMAND")
    points_sub.add_parser("sweep", help="Run one (arm, seed) cell of a guide-family grid",
                          add_help=False)

    family.add_parser("fit", help="Fit one recording (single-trace model)", add_help=False)
    family.add_parser("fit-cohort", help="Fit a cohort (hierarchical model)", add_help=False)

    kernel = family.add_parser("kernel", help="The transient kernel")
    kernel_sub = kernel.add_subparsers(dest="command", required=True, metavar="COMMAND")
    kernel_sub.add_parser("calibrate", help="Fit the kernel to a set of recordings",
                          add_help=False)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args, remaining = parser.parse_known_args(argv)
    target = _COMMANDS[(args.family, getattr(args, "command", None))]
    mod = importlib.import_module(target)
    mod.main(remaining)


if __name__ == "__main__":
    main()
