#!/usr/bin/env python3
"""Write one config per dP cell of the tonic-response benchmark.

    python scripts/cohort_tonic_cells.py --config configs/cohort_tonic_48.yaml --root <dir> \
        [--set tonic_block=true] [--set stop_min_steps=3000 ...]

Each cell dir <root>/dp<dP> gets config_dp<dP>.yaml; `--set key=value` entries (YAML
values) are merged into the fit block of every cell, which is how an opt-in fit setting
such as fit.tonic_block reaches all four arms identically.
"""
from __future__ import annotations

import argparse

import yaml

from photosvi.experiments.tonic_benchmark import write_cell_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()
over = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}
for dp in yaml.safe_load(open(a.config))["grid"]["tonic_dp"]:
    print(write_cell_config(a.config, dp, f"{a.root}/dp{dp:.2f}", over))
