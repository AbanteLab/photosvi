"""Fit one simulated recording with `fit_session`, the way you would fit your own.

Writes the toy recording to `photosvi_toy/session.csv` first, so the same file can then
be fitted from the command line. Short, and fewer steps than a real fit.
"""

import os
from pathlib import Path

import numpy as np
import yaml

from photosvi.api import FitSettings, config_path, fit_session, load_session
from photosvi.experiments.cohort import simulate_cohort, spec_from_config

out = Path(os.environ.get("PHOTOSVI_TOY", "photosvi_toy"))
out.mkdir(exist_ok=True)

# --- 1. a toy recording from the paper's simulator: 6 minutes at 48 Hz, the injection at
#        2 minutes doubling the transient rate (0.7 on the log scale). The transients are
#        made larger and more frequent than in the recordings so that 6 minutes suffice.
cfg = yaml.safe_load(config_path("final_cohort_48.yaml").read_text())
cfg["cohort"].update(n_animals=1, T=48 * 360, inj_t=48 * 120, inj_jitter=0, mu_saline=0.7,
                     beta0_mean=-2.2, mu_a=-2.7, sigma_b0=0.0, sigma_b0_session=0.0,
                     sigma_animal=0.0, sigma_cell=0.0)
toy = simulate_cohort(spec_from_config(cfg, delta=0.0), seed=1)
np.savetxt(out / "session.csv", toy["y"][0].numpy(), fmt="%.6f")

# --- 2. read it as you would read your own trace: a dF/F file, its rate, the injection.
session = load_session(out / "session.csv", fs=48, inj_s=120, animal="toy", condition="drug")

# --- 3. fit. FitSettings() alone is the paper's model; the step counts are cut for the toy.
result = fit_session(session, FitSettings(steps=800, max_steps=1600))

rr = result.rate_ratio
print(f"rate ratio {rr['estimate']:.2f} [{rr['lo']:.2f}, {rr['hi']:.2f}] "
      f"(simulated {np.exp(0.7):.2f})")
print(f"{len(result.events()['bins'])} transients above the detection threshold")
result.save(out / "run_session")
