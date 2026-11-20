"""Recompute the ALS arm of the detection benchmark's tonic record at lambda = 1e6 * fs.

The 30-seed run applied ALS with lambda = 1e6 to the 48-Hz trace, the value of the lab's
script, whose recordings it was set for are smoothed and decimated; on the raw 48-Hz trace
it follows the noise. The paper's preprocessing scales it with the sampling rate (1e6 * fs,
2026-10-26), and this applies the same rule here. The fit is not repeated: the simulated
trace is regenerated from the seed exactly as benchmark_guppy.py draws it (checked against
the stored low-pass and binned data), and only the ALS curve and summary are replaced. The
earlier ALS summary is kept under "als_lam1e6".

usage: detection_tonic_als.py RUNS_DIR   (BM_CONFIG as for benchmark_guppy.py)
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from detection_tonic import ALS_P, bin_mean, downsample, lowpass, summarize  # noqa: E402

from photosvi.experiments.marks import noise_unit_generation  # noqa: E402
from photosvi.experiments.sweep import gen_spec  # noqa: E402
from photosvi.experiments.tonic_benchmark import als_baseline  # noqa: E402
from photosvi.models.points import (  # noqa: E402
    AdditiveEmission,
    PhasicEventProcess,
    TonicBaseline,
    build_kernel,
)

runs = sys.argv[1]
cfg = yaml.safe_load(open(os.environ.get("BM_CONFIG", "configs/final_detection_48.yaml")))
d, k, gt, ton = cfg["data"], cfg["kernel"], cfg["gt"], cfg["tonic"]
dt, T, INJ, fs = 1 / d["fs"], d["T"], d["inj_t"], d["fs"]
PHI = (cfg.get("emission", {}).get("ar_coeffs")) or None
LAM = 1e6 * fs


def simulate(seed, sigma):
    g = torch.Generator().manual_seed(seed)
    x_true = TonicBaseline(dt=dt, **ton).baseline(T, INJ)
    mu, sg = noise_unit_generation(cfg.get("fit"), sigma, gt["mu_a"], gt["sigma_a"])
    ex = PhasicEventProcess(build_kernel(gen_spec(k), k["length"], dt), beta0=gt["beta0"],
                            beta_inj=gt["beta_inj"], mu_a=mu, sigma_a=sg, dt=dt,
                            history=None).sample(T, INJ, 1, g)
    y = AdditiveEmission(sigma_y=sigma, ar_coeffs=PHI).sample(x_true + ex["c"], generator=g)["y"]
    return np.asarray(y, dtype=float).reshape(-1)


for f in sorted(glob.glob(f"{runs}/bm_*.json")):
    r = json.load(open(f))
    tn = r["tonic"]
    if tn["summary"].get("als_lam") == LAM:
        continue
    sigma, seed = float(os.path.basename(f)[3:7]), r["seed"]
    y = simulate(seed, sigma)
    # The regenerated trace must be the one that was fitted.
    assert np.allclose(bin_mean(y, fs, tn["curve_step_s"]), tn["curves"]["data"], atol=1e-5), f
    assert np.allclose(downsample(lowpass(y, fs), fs, tn["curve_step_s"]), tn["curves"]["lowpass"], atol=1e-5), f
    base = als_baseline(y, lam=LAM, p=ALS_P)
    tn["summary"]["als_lam1e6"] = tn["summary"]["als"]
    tn["summary"]["als"] = summarize(base, INJ, fs)
    tn["summary"]["als_lam"] = LAM
    tn["curves"]["als"] = downsample(base, fs, tn["curve_step_s"])
    tmp = f + ".tmp"
    json.dump(r, open(tmp, "w"))
    os.replace(tmp, f)
    print(f"{os.path.basename(f)}: ALS height {tn['summary']['als']['peak_height']:+.2f} "
          f"(was {tn['summary']['als_lam1e6']['peak_height']:+.2f})", flush=True)
