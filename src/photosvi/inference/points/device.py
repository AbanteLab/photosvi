"""Device selection for the points fits, and the measurement that should drive it.

Both models run on either CPU or GPU; which one is *faster* depends entirely on the
batch shape, and the answer is counter-intuitive enough to be worth stating where the
switch lives.

SVI here is **launch-bound**, not compute-bound: a step is a long chain of small
kernels (the FIR convolution, the elementwise KLs, the emission) and the GPU spends
its time on launch overhead rather than arithmetic. Measured 2026-09-08 on an RTX
6000 Ada at T = 60000, ms per SVI step:

    B         1      8     16     32
    cuda   35.3   56.4   55.4   46.3        <- flat: the batch rides along for free
    cpu    59.2  424.8  756.4 1793.7        <- linear in B

(The CPU column was measured while 24 benchmark workers had the machine saturated,
so it is pessimistic by roughly 2x -- an uncontended single-threaded `B = 1` step is
~29 ms. The GPU column is clean; the cards were idle.)

The GPU curve being *flat* is the whole story, and it splits the two workloads:

  * **`B = 1`, many independent fits** (the arm sweeps, the coverage replicates, the
    GuPPy benchmark -- 50 cells that share nothing) -- **CPU wins on throughput**.
    Per fit the two are comparable (~29 vs ~35 ms/step), but 24 CPU fits run at once
    on this machine against one on the GPU.
  * **`B > 1`, one fit over a cohort** (the hierarchical model) -- **GPU wins
    outright**. A 16-session cohort fit costs the same wall clock as a single-trace
    fit (~19 min to a 21000-step plateau) where the CPU would need ~4.4 hours.

The rule of thumb: put the *batch* on the GPU and the *fan-out* on the CPU. Choosing
"gpu" for a `B = 1` fan-out is not wrong, just slower overall -- so the default here
is "auto", which applies exactly that rule, and an explicit choice is always honoured.

For *many* cohort fits (the two-stage vs hierarchical comparison) the two combine:
each process is launch-bound and leaves the card mostly idle, so several GPU
processes should share one card at close to linear throughput. Not yet measured.
"""

from __future__ import annotations

import torch

_BATCH_THRESHOLD = 2       # B >= this is a "cohort"; below it the fan-out wins


def resolve_device(spec: str | torch.device | None = "auto", *, batch: int = 1) -> torch.device:
    """Pick a torch device.

    Parameters
    ----------
    spec : "auto" (default), "cpu", "cuda", "cuda:1", or a `torch.device`. None is
        treated as "auto". An explicit request for CUDA when none is available is an
        **error**, not a silent fallback: a run that was launched for the GPU and
        quietly took 60x longer on CPU is worse than one that refused to start.
    batch : the number of sequences the fit will carry. Only consulted under "auto",
        where it selects by the rule in the module docstring.
    """
    if isinstance(spec, torch.device):
        return spec
    s = "auto" if spec is None else str(spec).strip().lower()
    if s == "auto":
        return torch.device("cuda" if (batch >= _BATCH_THRESHOLD
                                       and torch.cuda.is_available()) else "cpu")
    if s == "cpu":
        return torch.device("cpu")
    if s.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device {spec!r} was requested but CUDA is not available. Pass "
                "'cpu' or 'auto' to run on CPU -- falling back silently would turn a "
                "GPU-sized run into one that takes far longer without saying so."
            )
        dev = torch.device(s)
        n = torch.cuda.device_count()
        if dev.index is not None and dev.index >= n:
            raise RuntimeError(f"{spec!r} requested but only {n} CUDA device(s) present")
        return dev
    raise ValueError(f"unrecognized device {spec!r}; use 'auto', 'cpu' or 'cuda[:i]'")


def prepare(model, y: torch.Tensor, device) -> tuple[object, torch.Tensor]:
    """Move a model and its traces onto `device` together, and set CPU threading.

    On CPU the fits are run one-per-process under an external fan-out, so a single
    torch thread per process is right: the math backends otherwise build an OpenMP
    pool per worker and oversubscribe the machine several times over. On GPU the
    thread count is irrelevant and left alone.
    """
    dev = torch.device(device)
    if dev.type == "cpu":
        torch.set_num_threads(1)
    return model.to(dev), y.to(dev)
