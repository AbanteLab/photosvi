"""Fit a simulated cohort with `fit_cohort`, the way you would fit your own.

Writes one file per recording and the table of sessions to `photosvi_toy/` first, so the
same cohort can then be fitted from the command line. Two animals, each recorded under a
vehicle and under a drug; short recordings and few steps, so the numbers are not the paper's.
"""

import csv
import os
from pathlib import Path

import numpy as np
import yaml

from photosvi.api import CohortSettings, Study, config_path, fit_cohort, read_manifest
from photosvi.experiments.cohort import simulate_cohort, spec_from_config

out = Path(os.environ.get("PHOTOSVI_TOY", "photosvi_toy"))
out.mkdir(exist_ok=True)

# --- 1. a toy cohort from the paper's simulator: 2 animals x {vehicle, drug}, 6 minutes
#        each at 48 Hz with the injection at 2 minutes, the drug raising the transient rate
#        by 0.7 on the log scale over the vehicle. Larger, more frequent transients than in
#        the recordings, so that a few short recordings suffice.
cfg = yaml.safe_load(config_path("final_cohort_48.yaml").read_text())
cfg["cohort"].update(n_animals=2, T=48 * 360, inj_t=48 * 120, inj_jitter=0,
                     beta0_mean=-2.2, mu_a=-2.7)
toy = simulate_cohort(spec_from_config(cfg, delta=0.7), seed=0)
names = ["vehicle", "drug"]
rows = []
for i in range(toy["y"].shape[0]):
    animal, cond = f"m{int(toy['animal'][i]) + 1}", names[int(toy["drug"][i])]
    path = f"{animal}_{cond}.csv"
    np.savetxt(out / path, toy["y"][i].numpy(), fmt="%.6f")
    rows.append(dict(path=path, animal=animal, condition=cond, fs=48, inj_start=120,
                     inj_end=120))
with (out / "sessions.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# --- 2. read the table: one row per recording, with its animal and its condition.
sessions, _ = read_manifest(out / "sessions.csv")

# --- 3. fit jointly. CohortSettings() alone is the paper's fit; steps cut for the toy.
result = fit_cohort(Study(sessions), CohortSettings(min_steps=200, max_steps=300,
                                                     probe_every=100, pre_s=None),
                    reference="vehicle")

# With two animals and a few hundred steps the interval is wide and the estimate is still
# shrunk toward zero; a real cohort is fitted for tens of thousands of steps.
c = result.contrast("drug")
print(f"drug vs vehicle: rate ratio {c['ratio']:.2f} [{c['ratio_lo']:.2f}, "
      f"{c['ratio_hi']:.2f}] (simulated {np.exp(0.7):.2f})")
print("design:", {k: result.design[k] for k in ("n_animals", "n_sessions", "within_animal")})
result.save(out / "run_cohort")
