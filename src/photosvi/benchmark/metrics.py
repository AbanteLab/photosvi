"""Scoring a detector's discrete events against known onsets.

Two metric families, because they answer different questions:

- **Event-matched** (`match_events`, `event_metrics`, `pr_curve`) -- greedy one-to-one
  assignment of detections to onsets within a tight tolerance, then PR-AUC (average
  precision), recall at a fixed precision, and best F1. The honest, rare-event-aware
  detection metric; PR not ROC because true negatives (empty bins) dominate and flatter ROC.
- **Per-bin** (`perbin_auc`) -- tolerance-free ROC-AUC over per-bin scores with
  onset-window bins as positives. Measures *localization quality*, and reproduces the
  familiar ~0.99; forgiving by construction.

The matching tolerance is load-bearing: it must be well under the inter-event interval,
or matching is trivial (a random score scores near a real one). See the benchmark note.
"""

from __future__ import annotations

import numpy as np


def match_events(
    pred_t: np.ndarray, pred_s: np.ndarray, onsets: np.ndarray, tol: int, lag: int = 0
) -> np.ndarray:
    """Greedy one-to-one match, highest score first. Returns a TP mask over predictions.

    Prediction i is a true positive if |pred_t[i] - lag - onset| <= tol for some onset not
    yet claimed by a higher-scoring prediction. One-to-one so a burst on one event counts once.
    """
    pred_t = np.asarray(pred_t)
    pred_s = np.asarray(pred_s)
    onsets = np.asarray(onsets)
    used = np.zeros(len(onsets), dtype=bool)
    tp = np.zeros(len(pred_t), dtype=bool)
    if len(onsets) == 0:
        return tp
    for i in np.argsort(-pred_s):
        d = np.abs((pred_t[i] - lag) - onsets).astype(float)
        d[used] = np.inf
        if d.min() <= tol:
            tp[i] = True
            used[int(np.argmin(d))] = True
    return tp


def pr_curve(
    pred_t: np.ndarray, pred_s: np.ndarray, onsets: np.ndarray, tol: int, lag: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Precision & recall arrays swept over the score threshold, started at (recall 0, prec 1)."""
    if len(pred_t) == 0:
        return np.array([0.0, 0.0]), np.array([1.0, 0.0])
    order = np.argsort(-np.asarray(pred_s))
    tp = match_events(pred_t, pred_s, onsets, tol, lag)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    P = max(len(onsets), 1)
    rec, prec = ctp / P, ctp / (ctp + cfp)
    return np.concatenate([[0.0], rec]), np.concatenate([[1.0], prec])


def event_metrics(
    pred_t: np.ndarray,
    pred_s: np.ndarray,
    onsets: np.ndarray,
    tol: int,
    lag: int = 0,
    *,
    prec_at: float = 0.9,
) -> dict:
    """Event-matched PR-AUC (average precision), recall@precision, best F1, and counts."""
    P = len(onsets)
    if len(pred_t) == 0:
        return dict(pr_auc=0.0, recall_at=0.0, best_f1=0.0, n_pred=0, n_true=P)
    order = np.argsort(-np.asarray(pred_s))
    tp = match_events(pred_t, pred_s, onsets, tol, lag)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    prec, rec = ctp / (ctp + cfp), ctp / max(P, 1)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    ok = prec >= prec_at
    return dict(
        pr_auc=float((prec * tp).sum() / max(P, 1)),
        recall_at=float(rec[ok].max()) if ok.any() else 0.0,
        best_f1=float(f1.max()),
        n_pred=int(len(pred_t)),
        n_true=P,
    )


def perbin_auc(score: np.ndarray, onsets: np.ndarray, T: int, *, k: int = 60) -> float:
    """Tolerance-free per-bin ROC-AUC: onset-window (+-k bins) positive vs background."""
    score = np.asarray(score, dtype=float)
    on = np.zeros(T, dtype=bool)
    for o in onsets:
        on[max(0, o - k):min(T, o + k + 1)] = True
    pos, neg = score[on], score[~on]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.concatenate([pos, neg]).argsort().argsort().astype(float)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))
