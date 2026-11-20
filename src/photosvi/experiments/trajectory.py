"""One arm's fit, checkpointed: the ELBO, the betas, and detection over training time.

The two legs of the guide-family comparison are scored here but kept strictly apart:

  * **detection** -- PR-AUC (ranking, threshold-free), best-F1 (the calling ceiling if
    you were allowed to tune the threshold against truth) and F1@thresh (calling as
    shipped). The gap between the last two is the *calibration deficit*: how much of an
    arm's detection quality is unreachable without peeking at the answer.
  * **recovery** -- the betas, reported as events/minute rather than log-rates.

Detection is scored inline at each checkpoint because reading `event_posterior` does not
mutate the parameter store. The held-out probe is *not* run here: it needs its own store,
and clearing the store mid-fit would orphan the live optimizer's Adam state. `sweep.py`
replays it afterwards over the recorded betas, which is exactly equivalent because
everything except the betas is pinned.
"""

from __future__ import annotations

import time

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from photosvi.benchmark import calibrate_lag, event_metrics, match_events, nms, peaks_from_score
from photosvi.experiments.arms import PointsSetup, build_arm

# An ELBO below this is a relaxation blow-up, not a fit: neighbouring steps sit
# ~4 orders of magnitude higher. Used only to *count* excursions, never to gate.
DIVERGENCE_ELBO = -1e4


def ev_per_min(log_rate) -> np.ndarray:
    """Log-rate (Hz) -> events per minute.

    The link is complementary-log-log, so beta is a log-rate and not a logit -- the
    conversion is a plain exponential, with no sigmoid anywhere.
    """
    return np.exp(np.asarray(log_rate, dtype=float)) * 60.0


def detection_row(
    score, onsets: np.ndarray, *, tol: int, refractory: int = 8, thresh: float = 0.5
) -> dict:
    """PR-AUC / best-F1 / F1@thresh for one per-bin score against known onsets.

    The lag is calibrated on the *full* peak set, not the thresholded one, so the
    F1@thresh number is not quietly helped by re-aligning a smaller set of calls.
    """
    pt, ps = nms(*peaks_from_score(np.asarray(score, dtype=float)), refractory=refractory)
    if len(pt) == 0:
        return dict(pr_auc=0.0, best_f1=0.0, f1_at=0.0, n_pred=0, n_call=0, lag=0)
    lag = calibrate_lag(pt, onsets)
    m = event_metrics(pt, ps, onsets, tol, lag)

    keep = ps >= thresh
    ct, cs = pt[keep], ps[keep]
    if len(ct):
        tp = int(match_events(ct, cs, onsets, tol, lag).sum())
        prec, rec = tp / len(ct), tp / max(len(onsets), 1)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    else:
        f1 = 0.0
    return dict(
        pr_auc=float(m["pr_auc"]), best_f1=float(m["best_f1"]), f1_at=float(f1),
        n_pred=int(len(pt)), n_call=int(keep.sum()), lag=int(lag),
    )


def fit_trajectory(
    arm: str,
    y: torch.Tensor,
    onsets: np.ndarray,
    inj_t: int,
    setup: PointsSetup,
    *,
    mf,
    seed: int = 0,
    steps: int = 15_000,
    probe: int = 250,
    lr: float = 1e-3,
    tol: int = 9,
    refractory: int = 8,
    thresh: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Fit one arm, recording a checkpoint every `probe` steps.

    Returns a plain dict of lists (JSON-serializable), plus the final `q` for the
    end-of-run detection report.
    """
    model = build_arm(arm, y, inj_t, setup, seed=seed, mf=mf)
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    svi = SVI(model.model, model.guide, ClippedAdam({"lr": lr}), Trace_ELBO(max_plate_nesting=2))
    n = y.numel()

    # under coupling the block's logits are only part of q(s) -- the guide adds a live
    # prior anchor on top, so the readout needs inj_t to reconstruct the sum.
    ho_inj = inj_t if model.couple_events else None

    rec: dict = dict(
        arm=arm, seed=seed, steps_total=steps, probe=probe,
        elbo=[], steps=[], b0=[], bi=[], pr_auc=[], best_f1=[], f1_at=[], n_call=[], n_pred=[],
    )
    t0 = time.time()
    for step in range(1, steps + 1):
        rec["elbo"].append(-svi.step(y, inj_t) / n)
        if step == 1 or step % probe == 0 or step == steps:
            bp = model.beta_posterior()
            b0, bi = float(bp["beta0"][0]), float(bp["beta_inj"][0])
            q = model.event_posterior(y, ho_inj)[0].numpy()
            d = detection_row(q, onsets, tol=tol, refractory=refractory, thresh=thresh)
            rec["steps"].append(step)
            rec["b0"].append(b0)
            rec["bi"].append(bi)
            for k in ("pr_auc", "best_f1", "f1_at", "n_call", "n_pred"):
                rec[k].append(d[k])
            if verbose:
                print(
                    f"  [{arm:10s} s{seed}] {step:6d}/{steps}  elbo {rec['elbo'][-1]:10.3f}  "
                    f"b0 {b0:+.3f}  b_inj {bi:+.3f} (x{np.exp(bi):.2f})  "
                    f"pr-auc {d['pr_auc']:.3f}  f1@{thresh:g} {d['f1_at']:.3f}",
                    flush=True,
                )
    rec["q_final"] = model.event_posterior(y, ho_inj)[0].numpy().tolist()
    rec["fit_secs"] = time.time() - t0
    # The relaxed ELBO spikes when the Concrete density blows up. Counting the
    # excursions makes optimizer instability a reported number rather than
    # something a reader has to notice in the trace.
    rec["elbo_best"] = float(max(rec["elbo"]))
    rec["n_diverged"] = int(sum(1 for v in rec["elbo"] if v < DIVERGENCE_ELBO))
    rec["frac_diverged"] = rec["n_diverged"] / len(rec["elbo"])
    return rec
