#!/usr/bin/env python3
"""Posterior predictive draws for one experimental recording (Supplementary Fig. 3).

    python scripts/paper_predictive_real.py --session DL55:noci --out <dir>/predictive_real.json

Fits the session with the package's defaults -- the paper's single-trace model -- through
`photosvi.api`, then runs the fitted model forward (`Result.simulate`). Replaces the
simulated recording the figure used until 2026-10-16: the check needs no ground truth, and
showing it on an experimental recording is the point of it. The JSON is what
scripts/paper_fig_predictive.py reads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from photosvi.api import FitSettings, Session, fit_session
from photosvi.data.bonaventura3 import FS, load_sessions
from photosvi.experiments.marks import NOISE_UNITS_MU, NOISE_UNITS_SIGMA

ap = argparse.ArgumentParser()
ap.add_argument("--session", default="DL55:noci", help="animal:condition")
ap.add_argument("--pre-s", type=float, default=600.0)
ap.add_argument("--mask-min-s", type=float, default=120.0)
ap.add_argument("--draws", type=int, default=4)
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--mark-mode", default="default", choices=("default", "noise_units"),
                help="noise_units: marks fixed at a_t = sigma_y r_t (experiments.marks)")
ap.add_argument("--mark-mu", type=float, default=NOISE_UNITS_MU,
                help="noise_units: median mark is exp(mu) sigma_y")
ap.add_argument("--mark-sigma", type=float, default=NOISE_UNITS_SIGMA)
ap.add_argument("--out", required=True)
a = ap.parse_args()

animal, cond = a.session.split(":")
ss = [s for s in load_sessions(a.pre_s, verbose=False, detrend_window=False,
                               mask_min_s=a.mask_min_s)
      if s.animal == animal and s.cond.startswith(cond)]
if len(ss) != 1:
    raise SystemExit(f"{a.session}: {len(ss)} sessions match")
s = ss[0]
keep = np.ones(len(s.y), dtype=bool)
lo, hi = s.mask_bins
keep[lo:hi] = False
res = fit_session(Session(trace=np.asarray(s.y, float), fs=float(FS), inj_t=int(s.inj_bin),
                          name=a.session, mask=keep),
                  FitSettings(device=a.device, threads=a.threads,
                              mark_noise_units=(None if a.mark_mode == "default" else
                                                [a.mark_mu, a.mark_sigma])))
draws = res.simulate(n=a.draws, seed=0)
out = dict(dt=1.0 / FS, y=[float(v) for v in s.y], session=a.session,
           pp=[[float(v) for v in d["y"]] for d in draws], pp_n=[d["n_events"] for d in draws],
           n_obs=int(round(float(res.event_prob.sum()))), obs_label="expected events",
           rate_ratio=res.rate_ratio, fitted=res.to_dict()["fitted"],
           stopped_at=res.stopped_at, mask=[int(lo), int(hi)])
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(out))
print(f"wrote {a.out}: {a.session}, rate ratio {res.rate_ratio['estimate']:.2f}, "
      f"expected events {out['n_obs']}, draws {out['pp_n']}, stopped {res.stopped_at}")
