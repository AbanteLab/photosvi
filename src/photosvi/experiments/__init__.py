"""Experiment machinery: run a grid of inference arms and log what each one did.

Scoped apart from `photosvi.benchmark`, which stays a *detection* harness. This package
answers a wider question -- given several variational families over the same generative
points model, what does each recover, and does anything computable from the data tell you
when to stop? -- so it owns the arm registry, the held-out protocol, the checkpointed fit
and the grid driver.

Typical use is the CLI (`photosvi points sweep`), one process per (arm, seed) cell:

    photosvi points sweep --config configs/points_guide_sweep.yaml --arm amort --seed 0 \\
        --out runs/

or in-process:

    from photosvi.experiments import load_config, run_cell, write_cell
    cfg = load_config("configs/points_guide_sweep.yaml")
    write_cell(run_cell(cfg, "amort", 0), "runs/")

Nothing here plots. The sweep writes JSON; reading it back and drawing is a separate job
on a separate machine.
"""

from photosvi.experiments.arms import ARMS, ArmSpec, PointsSetup, build_arm, frozen_local_model
from photosvi.experiments.heldout import heldout_probe, null_recon_ll, recon_ll
from photosvi.experiments.sweep import (
    cell_path,
    generate,
    load_config,
    pin_threads,
    provenance,
    run_cell,
    write_cell,
)
from photosvi.experiments.trajectory import detection_row, ev_per_min, fit_trajectory

__all__ = [
    "ARMS",
    "REFERENCE_ARM",
    "ArmSpec",
    "PointsSetup",
    "build_arm",
    "cell_path",
    "detection_row",
    "ev_per_min",
    "fit_trajectory",
    "frozen_local_model",
    "generate",
    "heldout_probe",
    "load_config",
    "null_recon_ll",
    "pin_threads",
    "provenance",
    "recon_ll",
    "run_cell",
    "write_cell",
]
