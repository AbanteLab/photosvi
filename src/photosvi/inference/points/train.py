"""SVI training loop for the points family (relaxed-discrete events).

The points
ELBO needs neither enumeration (the events are relaxed, not summed) nor the
DMM's KL annealing, so this is the plainest of the three loops: `Trace_ELBO`
over the reparameterized relaxed events and marks, `ClippedAdam`, full-batch --
the whole trace is one batch (time subsampling and temperature annealing are
not built).
"""

from __future__ import annotations

import copy
import time

import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam


def train_points_svi(
    model,
    y: torch.Tensor,
    inj_t: int,
    *,
    lr: float = 1e-3,
    steps: int = 1000,
    device: str = "cpu",
    verbose: bool = True,
    log_every: int = 100,
    seed: int | None = None,
    keep_best: bool = True,
) -> dict:
    """Fit a `SingleTraceModel` by relaxed-discrete SVI. Returns the per-element ELBO loss history.

    `y` is [B, T] observed traces sharing the injection bin `inj_t`. Optimizes
    `Trace_ELBO(max_plate_nesting=2)` (sequences x time plates) with ClippedAdam.

    The relaxed ELBO loss is noisy and backslides off its minimum, so by default
    (`keep_best=True`) the param store is snapshotted at every new best and
    restored at the end -- the returned model carries the *best-checkpoint*
    params, not the last step's. Set `keep_best=False` to leave the store at the
    last step (the raw trajectory, for debugging). This is the best-checkpoint
    analogue of standard early stopping: no run is halted early (there is no
    held-out split to stop on), but the fit is read off the minimum, not the tail.

    Returns `{"elbo": [...], "best_loss": float, "best_step": int}` -- the
    per-element negative ELBO history (lower is better), the minimum loss, and
    its 1-indexed step.
    """
    model.to(device)
    y = y.to(device)
    if seed is not None:
        pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    optim = ClippedAdam({"lr": float(lr)})
    elbo = Trace_ELBO(max_plate_nesting=2)
    svi = SVI(model.model, model.guide, optim, elbo)

    n_elem = y.numel()
    hist: list[float] = []
    best_loss, best_step, best_state = float("inf"), 0, None
    start = time.time()
    if verbose:
        print(f"Device: {device}  |  {y.shape[0]} seqs x {y.shape[1]} bins, tau={model.tau}")

    for step in range(1, steps + 1):
        loss = svi.step(y, inj_t) / n_elem
        hist.append(loss)
        if loss < best_loss:
            best_loss, best_step = loss, step
            if keep_best:
                best_state = copy.deepcopy(pyro.get_param_store().get_state())
        if verbose and (step == 1 or step % log_every == 0):
            print(f"[step {step}/{steps}] elbo loss: {loss:.4f}")

    # restore the best checkpoint so the returned model is read off the minimum,
    # not the noisy last step.
    if keep_best and best_state is not None:
        pyro.get_param_store().set_state(best_state)

    if verbose:
        elapsed = time.time() - start
        print(f"Training completed in {elapsed:.2f}s ({elapsed / 60:.2f} min) "
              f"| best elbo {best_loss:.4f} @ step {best_step}")
    return {"elbo": hist, "best_loss": best_loss, "best_step": best_step}
