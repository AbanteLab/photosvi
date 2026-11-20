"""Is the fit any good? A question the stopping rule does not ask.

`EstimandPlateau` stops when the reported quantity has stopped moving, and it does
that well. It is not a statement that the model has explained the recording, and on
2026-09-17 the difference cost a set of published numbers: a real morphine fit
plateaued on `beta_inj` at step 39000 and was written out, while its post-injection
residual sat at 0.132 against an emission noise of 0.0122 -- eleven times the scale
the model claims its own errors have. The tonic had been oscillating between 0.15 and
1.6 for the whole run. Nothing looked, so nothing said.

The statistic here is the one that would have said it. If the model is right, the
data minus its mean is emission noise and nothing else, so

    rho = sd(y - E[y]) / sigma_y

is 1. Above that, the fit's errors are larger than the scale it claims they have;
far above it, the fit is somewhere the emission scale does not describe at all.

**A low value is weak evidence and a high one is strong.** The asymmetry is not a
hedge, it is a property of the model: the phasic component carries a free event at
every window and a free mark on each, so it can follow structure the generative story
never claimed. Measured here -- a sine of amplitude 2.0 on noise of 0.05, four times
faster than the kernel is wide -- the fitted model absorbed it completely and came
back at 1.00, indistinguishable from the same recording without it. So this statistic
does not certify that a model is right, and nothing should be written as if it did.
What it detects is a fit that has LEFT the data: the tonic gone somewhere the phasic
cannot follow it back from, which is what the real morphine recording did at eleven
and what no one was looking for.

**Reported, not enforced.** Nothing in this module stops a fit or changes a result.
A threshold that silently discarded recordings would be a worse failure than the one
it replaced: a fit that is poor for a reason is still evidence, and the reason is
usually the interesting part. What the rule owes the reader is that the number is
computed, recorded beside the estimate, and printed where a person will see it.
"""

from __future__ import annotations

from collections import deque

import pyro
import torch
from pyro import poutine


@torch.no_grad()
def fitted_mean(model, *args, draws: int = 4, **kwargs) -> torch.Tensor:
    """The model's posterior-mean fluorescence, [B, T], averaged over `draws`.

    Replays the model against traces from its own guide and reads the structural mean
    the model recorded, which is exact for whatever the model happens to be: it asks
    Pyro what the model built rather than rebuilding the sum of tonic and phasic here,
    so it cannot drift from the model the way a second implementation would.

    Averaging over draws matters for the hierarchical model, whose tonic is sampled
    rather than held as a point estimate; a single draw is one recording's worth of
    tonic noise on top of the quantity being measured. Four is enough for the
    residual scale to settle to the second decimal and costs four forward passes.

    `args`/`kwargs` are whatever the model's `forward` signature wants -- `(y,
    inj_t)` for the single-trace model, `(y,)` for the cohort -- and are passed to
    both the guide and the model unchanged.
    """
    if draws < 1:
        raise ValueError(f"draws must be >= 1 (got {draws})")
    total = None
    for _ in range(draws):
        g = poutine.trace(model.guide).get_trace(*args, **kwargs)
        m = poutine.trace(poutine.replay(model.model, trace=g)).get_trace(*args, **kwargs)
        # `mean_struct` if the model records it, the emission's mean otherwise. The
        # distinction only appears with a correlated emission, and there it is the whole
        # difference between "the signal the model fitted" and "what it predicts for the
        # next bin given the last one" -- the second is near-perfect on any smooth trace
        # and would report a residual ratio of nothing.
        if "mean_struct" in m.nodes:
            mean = m.nodes["mean_struct"]["value"]
        else:
            mean = m.nodes["y"]["fn"].mean
        total = mean if total is None else total + mean
    return total / draws


@torch.no_grad()
def residual_ratio(model, y: torch.Tensor, *args, draws: int = 4, **kwargs):
    """`sd(y - E[y]) / sigma_y`, overall and per sequence.

    Returns `(overall, per_sequence)`. One is the value a correct model produces on
    data it generated; the per-sequence list is there because a cohort fit can be
    sound on most of its recordings and lost on one, and an average over twenty-four
    hides that completely.
    """
    mean = fitted_mean(model, y, *args, draws=draws, **kwargs)
    resid = (y - mean).reshape(y.shape[0], -1)
    sig = model.emission.sigma_y.detach().reshape(-1).cpu()
    # One scale, or one per sequence (2026-10-20): each residual against its own.
    per = [float(r.std()) / float(sig[i if sig.numel() > 1 else 0])
           for i, r in enumerate(resid)]
    overall = (float(resid.std()) / float(sig[0]) if sig.numel() == 1 else
               float((resid / sig.reshape(-1, 1).to(resid)).std()))
    return overall, per


class FitQuality:
    """Records the residual ratio across a fit and says what it means at the end.

    Holds the last `window` probes so the verdict is not read off a single draw, and
    keeps the worst value seen: a fit that passes through a bad state and returns is
    a different object from one that never left a good one, and the final probe
    cannot tell them apart.

    Parameters
    ----------
    tol : ratio at or below which the fit is called sound. The default of 1.5 is
        deliberately loose. Real recordings carry structure this model does not
        claim to describe -- movement, bleaching that is not first-order, whatever
        the sensor does that a biphasic kernel does not -- and a ratio of 1.3 is
        ordinary on data that is fine. It is set where it separates "explains the
        recording" from "is somewhere else entirely", which is the distinction that
        went unnoticed at eleven.
    window : probes held for the verdict, which is their median.
    """

    def __init__(self, tol: float = 1.5, window: int = 4):
        if tol <= 0:
            raise ValueError(f"tol must be > 0 (got {tol})")
        if window < 1:
            raise ValueError(f"window must be >= 1 (got {window})")
        self.tol = float(tol)
        self.window = int(window)
        self._vals: deque[float] = deque(maxlen=self.window)
        self.worst: float = 0.0
        self.worst_step: int | None = None
        self.n_probes: int = 0

    def update(self, step: int, ratio: float) -> float:
        """Record one probe; returns the ratio, so callers can log it inline."""
        r = float(ratio)
        self._vals.append(r)
        self.n_probes += 1
        if r > self.worst:
            self.worst, self.worst_step = r, int(step)
        return r

    @property
    def ratio(self) -> float:
        """MEDIAN of the held probes -- the state the fit ended in; NaN before any.

        Median and not mean, so that this and `worst` answer different questions. A
        fit that spikes once and recovers has a bad `worst` and a good `ratio`; a
        mean would report it as mediocre throughout and lose both facts. Measured on
        a window of 11.2, 1.03, 1.01, 1.04 the mean says 3.57 and the median 1.03,
        and it is the median that describes where the fit finished.
        """
        if not self._vals:
            return float("nan")
        v = sorted(self._vals)
        h = len(v) // 2
        return v[h] if len(v) % 2 else (v[h - 1] + v[h]) / 2

    @property
    def sound(self) -> bool:
        """Whether the fit ended within tolerance. False before any probe."""
        return bool(self._vals) and self.ratio <= self.tol

    def summary(self) -> dict:
        """What to write beside the estimate. Serialises as-is into a run's JSON."""
        return dict(residual_ratio=self.ratio, worst=self.worst,
                    worst_step=self.worst_step, tol=self.tol,
                    n_probes=self.n_probes, sound=self.sound)

    def verdict(self) -> str:
        """One line for a human reading the log."""
        if not self._vals:
            return "fit quality: not probed"
        if self.sound:
            return (f"fit quality: residual {self.ratio:.2f} x sigma_y "
                    f"(worst {self.worst:.2f} at step {self.worst_step})")
        return (f"FIT QUALITY: residual {self.ratio:.2f} x sigma_y, above {self.tol:.2f} "
                f"-- the model is not explaining this recording, and the estimate beside "
                f"it should not be read as if it were (worst {self.worst:.2f} at step "
                f"{self.worst_step})")


#: Variational blocks with one parameter per window rather than one per recording.
#: In a production single-trace fit these are 96,000 of the 96,010 free parameters --
#: `q_s_window_scores` alone is 75% of them -- and every failure of the week of
#: 2026-09-21 was one of them moving somewhere nobody was looking. The ten that carry
#: the estimand are watched closely; these are not watched at all.
PER_BIN_BLOCKS = ("q_s_window_scores", "q_s_logits", "q_a_loc", "q_a_scale")


@torch.no_grad()
def block_spread(store=None) -> dict[str, dict[str, float]]:
    """Location and spread of each per-bin variational block, from the param store.

    Reports the median and the 1st and 99th percentiles rather than the mean and
    standard deviation, because the failure mode is a tail: on a diverging fit the
    marks at empty windows fanned out while their median barely moved, so a summary
    built on the centre said nothing was happening. The percentiles are what said it.

    Cheap -- three quantiles over ~100k floats -- and returns `{}` for any block the
    installed guide does not create, so it is safe to call whatever is configured.

    This is a measurement and not a rule. There is no threshold here, because what a
    healthy spread looks like depends on the block and nobody has established it:
    that is written up as open work rather than guessed at.
    """
    store = pyro.get_param_store() if store is None else store
    out: dict[str, dict[str, float]] = {}
    for name in PER_BIN_BLOCKS:
        if name not in store:
            continue
        v = store[name].detach().reshape(-1).float()
        if not v.numel():
            continue
        q = torch.quantile(v, torch.tensor([0.01, 0.5, 0.99], device=v.device))
        out[name] = dict(n=int(v.numel()), p01=float(q[0]), median=float(q[1]),
                         p99=float(q[2]), spread=float(q[2] - q[0]),
                         finite=bool(torch.isfinite(v).all()))
    return out
