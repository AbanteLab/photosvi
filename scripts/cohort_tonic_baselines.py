"""The low-pass and ALS slow components of every recording of a cohort benchmark cell.

The cohort benchmark scores PhotoSVI's tonic estimates (hierarchical and two-stage); this
adds the two methods that extract the slow component without modelling the transients, on
the same recordings, for the tonic comparison of the cohort benchmark (2026-10-28). No fit:
each cohort is regenerated from (spec, seed) exactly as the arms drew it -- checked against
the event counts the two-stage files stored -- and each session's low-pass and ALS curves
are summarized as in scripts/detection_tonic.py (peak height above the mean from 300 s to
the injection, and the time of that peak).

Run with the environment of the cell, as the arms were:

    CONFIG=configs/final_cohort_48.yaml DESIGN=crossed N_ANIMALS=5 SIM_SIGMA_Y=0.016757 \
        python scripts/cohort_tonic_baselines.py <cell> [--workers 16]

writes <cell>/tb_<delta>_<seed>.json next to the hier_/ts_ files; existing files are kept.
"""
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from detection_tonic import als, lowpass, summarize  # noqa: E402

from photosvi.experiments.cohort import (  # noqa: E402
    apply_cohort_env,
    simulate_cohort,
    spec_from_config,
)


def one(job):
    cell, delta, seed = job
    out = f"{cell}/tb_{delta:.2f}_{seed:04d}.json"
    if os.path.exists(out):
        return out, "kept"
    cfg = apply_cohort_env(yaml.safe_load(open(os.environ["CONFIG"])))
    spec = spec_from_config(cfg, delta)
    coh = simulate_cohort(spec, seed)
    fs = 1.0 / spec.dt
    sessions = []
    for k in range(spec.n_sessions):
        ts = f"{cell}/ts_{delta:.2f}_{seed:04d}_{k:02d}.json"
        if os.path.exists(ts):  # the regenerated recording must be the one that was fitted
            r = json.load(open(ts))
            assert (r["n_pre"], r["n_post"]) == (int(coh["n_pre"][k]), int(coh["n_post"][k])), ts
        y = np.asarray(coh["y"][k], dtype=float).reshape(-1)
        inj = int(coh["inj_t"][k])
        sessions.append(dict(session=k, animal=int(coh["animal"][k]), drug=int(coh["drug"][k]),
                             lowpass=summarize(lowpass(y, fs), inj, fs),
                             als=summarize(als(y, fs), inj, fs)))
    json.dump(dict(delta=delta, seed=seed, fs=fs, sessions=sessions), open(out + ".tmp", "w"))
    os.replace(out + ".tmp", out)
    return out, "written"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = []
    for f in sorted(glob.glob(f"{a.cell}/hier_*.json")):
        m = re.match(r"hier_([\d.]+)_(\d+)\.json", os.path.basename(f))
        jobs.append((a.cell, float(m.group(1)), int(m.group(2))))
    with ProcessPoolExecutor(a.workers) as ex:
        for out, what in ex.map(one, jobs):
            print(f"{os.path.basename(out)}: {what}", flush=True)
