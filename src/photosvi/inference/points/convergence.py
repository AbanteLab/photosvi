"""Stopping criteria for the points SVI loop.

The obvious criterion -- watch the ELBO flatten -- does not work here, and it is
worth saying why rather than leaving the next person to rediscover it. The bound is
evaluated on Concrete-relaxed samples and its trace swings across orders of
magnitude between neighbouring steps: on a real fit it ranged from -3.1e6 to +1.2e4
with no trend visible at that scale. Any plateau test on it fires on noise.

What does settle is the quantity the model exists to report. On the reference
configuration `beta_inj` climbs for roughly the first 24000 steps and then holds
within about +/-0.02 for the following 90000, so a plateau test on the estimand
stops when the answer stops changing -- which is the thing a step budget was
standing in for.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence


class EstimandPlateau:
    """Stop when the reported estimand has stopped moving.

    Records a value at each probe and reports convergence once the estimand has
    stopped *drifting*, which is not the same as it having stopped moving. At a
    usable learning rate the iterate oscillates around the optimum rather than
    creeping toward it -- measured at lr=0.01, `beta_inj` runs 0.714, 0.748, 0.698,
    0.741, 0.739 while sitting on its answer. A test on the spread of those values
    reads that as unconverged and never fires, while a slow creep at lr=0.001 passes
    it easily and stops the fit 0.09 short. So the criterion compares the mean of
    the first half of the window against the second: no systematic movement between
    them, whatever the oscillation amplitude.

    Report `plateau_mean` rather than the last probe. Once the iterate is orbiting,
    the final value is one draw from the orbit and the average over it is a better
    estimate.

    Parameters
    ----------
    tol : movement of the window's centre, on the log-rate axis, that still counts
        as settled. The default of 0.02 is well inside the estimator's own sampling
        spread across seeds (0.13-0.16), so it is a statement about the optimiser
        having stopped, not a claim about precision.
    window : probes held. Must be even enough to halve; 6 gives three per half.
    min_steps : a floor below which convergence is never declared. Early in a fit
        `beta_inj` passes slowly through the region it starts in, which looks flat
        at any tolerance -- measured, it sits within 0.02 of its initial value for
        the first several hundred steps while going nowhere.
    """

    def __init__(self, tol: float = 0.02, window: int = 6, min_steps: int = 5_000):
        if tol <= 0:
            raise ValueError(f"tol must be > 0 (got {tol})")
        if window < 4:
            raise ValueError(f"window must be >= 4 probes to halve (got {window})")
        self.tol = float(tol)
        self.window = int(window)
        self.min_steps = int(min_steps)
        self._vals: deque[list[float]] = deque(maxlen=self.window)
        self._dim: int | None = None
        self.converged_at: int | None = None

    def update(self, step: int, value: float | Sequence[float]) -> bool:
        """Record a probe; return True once the plateau condition holds.

        `value` may be a scalar or a vector. The cohort estimand is `mu_drug`, one
        coordinate per drug, and the criterion is then the **worst** coordinate's
        drift: a fit is not finished because the drug that settled first has
        settled. A scalar is the one-coordinate case of the same rule.
        """
        v = ([float(value)] if isinstance(value, (int, float))
             else [float(x) for x in value])
        if self._dim is None:
            self._dim = len(v)
        elif len(v) != self._dim:
            raise ValueError(
                f"estimand changed width mid-fit: {self._dim} -> {len(v)}"
            )
        self._vals.append(v)
        if step < self.min_steps or len(self._vals) < self.window:
            return False
        if self.drift <= self.tol:
            if self.converged_at is None:
                self.converged_at = step
            return True
        return False

    @property
    def drift(self) -> float:
        """Largest coordinate drift of the window's centre; NaN until full."""
        per = self.drift_per_coordinate
        return float("nan") if not per else max(per)

    @property
    def drift_per_coordinate(self) -> list[float]:
        """Per-coordinate movement of the window's centre between its halves."""
        if len(self._vals) < self.window:
            return []
        rows = list(self._vals)
        h = len(rows) // 2
        first, second = rows[:h], rows[-h:]
        return [abs(sum(r[i] for r in second) / len(second)
                    - sum(r[i] for r in first) / len(first))
                for i in range(self._dim or 0)]

    @property
    def plateau_mean(self) -> float:
        """Mean over the window -- the value to report once converged.

        Scalar for a scalar estimand; for a vector use `plateau_means`.
        """
        m = self.plateau_means
        if not m:
            return float("nan")
        return m[0] if len(m) == 1 else float("nan")

    @property
    def plateau_means(self) -> list[float]:
        """Per-coordinate mean over the window -- the values to report."""
        if not self._vals:
            return []
        rows = list(self._vals)
        return [sum(r[i] for r in rows) / len(rows) for i in range(self._dim or 0)]
