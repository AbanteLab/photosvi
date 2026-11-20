"""Tonic carrier of the points family: the smooth injection-locked baseline x_hat(t).

The deterministic tonic baseline node. The tonic baseline is the slow carrier
the phasic c_t (events.py) rides on top of:

    x_hat(t) = b + 1[t>=t_inj] * ( P * shape(tau; k_rise, k_decay) + d*tau/3600 ),
    tau = (t - t_inj)*dt   (post-injection clock, in SECONDS, 0 before injection)

where `shape` is the pharmacokinetic Bateman curve NORMALIZED TO PEAK 1, and P is
therefore the height of the tonic bump in the units of the trace. Absorption
k_rise faster than elimination k_decay, so P > 0 and k_rise > k_decay > 0 are
structural, not fitting conveniences. Those constraints are carried in a
reparameterization rather than imposed as inequalities:

    P = exp(alpha),  k_decay = exp(rho),  k_rise = exp(rho) + exp(gamma)

so the unconstrained psi = (b, alpha, rho, gamma, d) lives in a clean R^5 with
positivity and ordering automatic and the difference-of-exponentials' sign/swap
degeneracy removed (the gap exp(gamma) > 0 picks one representative per mirror).

**Why the peak and not the raw amplitude.** Written the textbook way, the bump is
A*(exp(-k_decay*tau) - exp(-k_rise*tau)). When the two rates are close the bracket
is approximately (k_rise - k_decay)*tau, so what a recording constrains is the
PRODUCT of A and the gap between the rates, not A. That is a ridge of equivalent
solutions, and it is not hypothetical: least squares landed on it for every real
recording we have -- three of them fitted to tonics of the right scale, 0.05 to
0.17, with amplitudes of 0.063, 0.993 and 9.9998, the last pinned at its bound.
Gradient ascent then slides along the ridge, and since the parameters are in log
space a small step changes the gap multiplicatively; the fitted curve on a trace of
standard deviation 0.021 oscillated between 0.15 and 1.6 for 39000 steps and
stopped there, because the stopping rule watches the rate estimand and nothing
watches the tonic. Normalizing the shape removes the ridge: P is in the units of
the trace and the data identify it directly.

A consequence worth stating, since it is the reason the hierarchy is worth having:
per-drug `alpha` now means the peak tonic response of that drug, which is a
quantity a reader can use, rather than a coefficient that only means something
alongside the rate gap it multiplies.

**The drift is PER HOUR, for the same reason.** Photobleaching is a rate, so the
parameter should be one -- but expressed per second against a tau that runs to 1800,
its step size at the end of a recording is 1800 times its step size at the start, and
gradient ascent cannot take a step that is reasonable at both ends. That is not a
theoretical concern: on a real trace of standard deviation 0.021, a fitted drift of
0.000568 per second put 1.02 at the end of the recording, 48 times the whole trace's
spread, and it was this term and not the bump that made the tonic swing between 0.15
and 1.6 for 39000 steps. Dividing tau by 3600 leaves it a rate -- so it still pools
correctly across recordings of different lengths, which a total would not -- while
putting a step of 0.01 within a factor of two of the emission noise instead of three
orders of magnitude above it.

This module is the *deterministic* tonic node only -- like events.py it carries
no probability density of its own. It holds a single animal's params as
nn.Parameters; the drug->animal hierarchical prior is part of the SVI
model/guide, so it is not here. The Gaussian/Student-t emission on
x_hat(t) + c_t lives in emission.py.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# The shape function divides by its own peak, so both the gap and that peak need a
# floor -- not to bound the model (the parameter bounds below do that) but to keep the
# division defined at the edge of the box. Set well below any rate the bounds admit,
# so a fit never sits on them.
# The drift's time unit. One hour, so the parameter stays a RATE (it pools across
# recordings of different lengths, which a per-recording total would not) while its
# gradient is on the same scale as everything else in the model. See the module
# docstring for the measurement that forced this.
_DRIFT_SECONDS = 3600.0
_GAP_FLOOR = 1e-9
_PEAK_FLOOR = 1e-12


def bateman_shape(
    tau: torch.Tensor, k_decay: torch.Tensor, k_rise: torch.Tensor
) -> torch.Tensor:
    """Bateman curve normalized to peak 1, evaluated on the post-injection clock.

    `tau` is [..., T] (or broadcastable), `k_decay` and `k_rise` broadcast against
    it; returns the same shape. Zero at tau = 0, exactly 1 at the peak time
    log(k_rise/k_decay)/(k_rise - k_decay), decaying after.

    Both the curve and its peak are evaluated through the same factored form,

        exp(-k_decay*tau) * (-expm1(-gap*tau)),    gap = k_rise - k_decay,

    which is algebraically the difference of exponentials and numerically not the
    same thing. Subtracting two exponentials that agree to seven digits -- which is
    exactly the regime the rates drift into -- leaves nothing but rounding error in
    float32, and the ratio below would then divide one piece of noise by another.
    `expm1` is accurate for small arguments, so the factored form holds its
    precision where the naive difference loses all of it. Evaluating the peak the
    same way is what makes `shape` equal 1 there to machine precision rather than
    approximately.
    """
    gap = (k_rise - k_decay).clamp_min(_GAP_FLOOR)
    t_peak = (k_rise / k_decay).log() / gap
    num = torch.exp(-k_decay * tau) * (-torch.expm1(-gap * tau))
    peak = torch.exp(-k_decay * t_peak) * (-torch.expm1(-gap * t_peak))
    return num / peak.clamp_min(_PEAK_FLOOR)


def bateman_raw_amplitude(
    peak: torch.Tensor, k_decay: torch.Tensor, k_rise: torch.Tensor
) -> torch.Tensor:
    """The textbook amplitude A that reproduces a bump of height `peak`.

    Provided so a fit can be reported in the classical parameterization, and so the
    ridge stays visible rather than merely avoided: A is what is NOT identified, and
    a fit whose A is enormous while its peak is ordinary has landed on the flat
    direction. Nothing in the model uses this.
    """
    gap = (k_rise - k_decay).clamp_min(_GAP_FLOOR)
    t_peak = (k_rise / k_decay).log() / gap
    f_peak = torch.exp(-k_decay * t_peak) * (-torch.expm1(-gap * t_peak))
    return peak / f_peak.clamp_min(_PEAK_FLOOR)


class TonicBaseline(nn.Module):
    """Deterministic hierarchical-Bateman tonic baseline x_hat(t).

    Holds a single animal's tonic parameters in the *unconstrained* space
    psi = (b, alpha, rho, gamma, d): the baseline level b and drift d are free
    reals, while the amplitude A and the rate constants k_decay < k_rise come
    from the reparameterization so positivity and the ordering are
    automatic and the sign/swap degeneracy is gone. The constrained
    (A, k_decay, k_rise) are exposed as read-only properties.

    Parameters are given in the *constrained* space at construction (the natural
    units) and stored as their unconstrained transforms, mirroring how
    PhasicEventProcess stores sigma_a in log space.

    Parameters
    ----------
    b : baseline level (pre-injection ambient tonic; ~0 on a z-scored trace).
    peak : height of the Bateman bump in the units of the trace, must be > 0
        (peak = 0 is the saline-like no-response case; pass a small positive value
        to approximate it). NOT the textbook amplitude A, which is a coordinate the
        data do not identify -- see the module docstring. `A` is available as a
        derived read-only property for anyone who wants it.
    k_rise, k_decay : absorption / elimination rate constants (s^-1), must
        satisfy k_rise > k_decay > 0. Both *slow* (injection timescale) -- this
        is the timescale separation B1 that keeps the tonic identifiable against
        the fast phasic kernel.
    drift : linear drift slope, PER HOUR, a first-order photobleaching / baseline
        correction; typically small and negative. Per hour and not per second so
        that its gradient is on the same scale as the rest of the model -- again,
        see the module docstring.
    dt : bin width in seconds.
    """

    def __init__(
        self,
        *,
        b: float = 0.0,
        peak: float = 1.0,
        k_rise: float = 0.01,
        k_decay: float = 0.002,
        drift: float = 0.0,
        dt: float = 1.0 / 25.0,
    ):
        super().__init__()
        if peak <= 0:
            raise ValueError(
                f"peak must be > 0 (got {peak}); peak=exp(alpha) is positive by construction"
            )
        if not (k_rise > k_decay > 0):
            raise ValueError(
                f"need k_rise > k_decay > 0 (got k_rise={k_rise}, k_decay={k_decay}); "
                "absorption is faster than elimination (Bateman ordering)"
            )
        self.dt = float(dt)

        # Unconstrained psi: peak=exp(alpha), k_decay=exp(rho),
        # k_rise=exp(rho)+exp(gamma) so the gap exp(gamma)>0 enforces the order.
        self.b = nn.Parameter(torch.tensor(float(b)))
        self.alpha = nn.Parameter(torch.tensor(math.log(peak)))
        self.rho = nn.Parameter(torch.tensor(math.log(k_decay)))
        self.gamma = nn.Parameter(torch.tensor(math.log(k_rise - k_decay)))
        self.drift = nn.Parameter(torch.tensor(float(drift)))

    # ------------------------------------------------------- constrained params
    @property
    def peak(self) -> torch.Tensor:
        """Height of the tonic bump, peak = exp(alpha) > 0, in the trace's units."""
        return self.alpha.exp()

    @property
    def A(self) -> torch.Tensor:
        """The textbook Bateman amplitude this fit corresponds to. Derived, not stored.

        Kept because it is diagnostic: A is the quantity the recording does not
        identify, so an ordinary `peak` alongside an enormous `A` says the rates have
        drifted together and the fit is sitting on the flat direction.
        """
        return bateman_raw_amplitude(self.peak, self.k_decay, self.k_rise)

    @property
    def k_decay(self) -> torch.Tensor:
        """Elimination rate k_decay = exp(rho) > 0."""
        return self.rho.exp()

    @property
    def k_rise(self) -> torch.Tensor:
        """Absorption rate k_rise = exp(rho) + exp(gamma) > k_decay."""
        return self.rho.exp() + self.gamma.exp()

    @property
    def peak_time(self) -> torch.Tensor:
        """Bateman peak time t_max = ln(k_rise/k_decay)/(k_rise - k_decay).

        Seconds after injection at which the bump (drift aside) is maximal.
        """
        kr, kd = self.k_rise, self.k_decay
        return (kr / kd).log() / (kr - kd)

    # ----------------------------------------------------------------- baseline
    def baseline(self, T: int, inj_t: int) -> torch.Tensor:
        """x_hat(t) over t = 0..T-1 for an injection at bin `inj_t`.

        Returns a length-T tensor: flat at `b` before injection, then the
        injection-gated Bateman bump plus linear drift. The post-injection clock
        tau is clamped at 0 so the masked-off pre-injection region carries no
        spurious exp(-k*negative) blow-up. Shape [T]; broadcasts against a
        phasic c of shape [B, T] when forming the additive mean x_hat + c.
        """
        t = torch.arange(T, device=self.b.device, dtype=self.b.dtype)
        post = (t >= inj_t).to(self.b.dtype)                 # 1[t >= t_inj]
        tau = (t - inj_t).clamp_min(0).to(self.b.dtype) * self.dt  # post-inj clock, >=0
        bump = self.peak * bateman_shape(tau, self.k_decay, self.k_rise)
        return self.b + post * (bump + self.drift * tau / _DRIFT_SECONDS)

    def forward(self, T: int, inj_t: int) -> torch.Tensor:
        """Alias for `baseline` so the module is callable."""
        return self.baseline(T, inj_t)


def fit_tonic_baseline(
    y: torch.Tensor,
    inj_t: int,
    *,
    dt: float = 1.0 / 25.0,
    smooth_s: float = 10.0,
    peak_max: float = 10.0,
) -> TonicBaseline:
    """Least-squares pre-fit of the Bateman baseline (eq 4) to `y`, as an SVI init.

    A data-driven initializer for the tonic globals psi, the sibling of
    `estimate_log_baseline_rate` for the phasic GLM: it lands the smooth carrier
    near-optimal *before* relaxed-discrete SVI so the globals start
    well-conditioned and the seed-to-seed fit is more reproducible. Ported from
    an earlier double-exponential fit with three upgrades -- the injection bin is known (no t0 search), the linear drift
    `d` is included, and the fit target is a *median-smoothed* trace.

    Two design choices, both empirical:

    - **Fit the smoothed trend, not the raw trace.** The tonic is the *slow*
      carrier; fit to raw `y` it chases the fast phasic spikes it has no events
      to explain. We median-smooth over a `smooth_s`-second window first
      (median, not mean, to shrug off the upward event excursions) and fit the
      Bateman to that slow trend.
    - **Cap the amplitude at `a_max`.** When k_rise ~ k_decay the
      difference-of-exponentials collapses toward a single-rate shape and only
      the product A*gap is identified, so an uncapped fit sends A to ~1e3
      against a vanishing gap. The cap both regularizes that corner and -- per a
      3-seed SVI sweep -- gives the most *stable* downstream fit (a modest tonic
      lets the events carry the fast structure).

    `y` is [B, T] or [T]; a batch is averaged over B first (the tonic is shared
    across an animal's sequences, and the mean cancels phasic events landing at
    different bins, leaving the common slow trend). The Bateman ordering
    k_rise > k_decay is enforced inside `curve_fit`'s box bounds via the same
    gap = exp(gamma) > 0 reparameterization `TonicBaseline` carries internally
    (eq 5): we fit (b, A, k_decay, gap, d) and set k_rise = k_decay + gap.

    Returns a fresh `TonicBaseline` built from the fit (constrained units), ready
    to drop into `SingleTraceModel`; SVI then refines it. A near-flat trace (a saline
    non-responder) yields a tiny A and an essentially flat baseline -- the
    pre-fit does not manufacture a bump.

    Parameters
    ----------
    y : observed trace, [B, T] or [T] (z-scored dF/F in practice).
    inj_t : injection bin index.
    dt : bin width in seconds (sampling period); the smoothing window and the
        Bateman time axis are both in seconds.
    smooth_s : median-smoothing window in seconds (~30s strips phasic events).
    peak_max : cap on the height of the fitted bump, in the units of the trace.
        Was a cap on the raw amplitude A, where it did nothing useful: A is the
        unidentified half of the old parameterization, so capping it bounded a
        coordinate rather than the curve, and one real recording sat on the cap
        while producing an entirely ordinary tonic.
    """
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.signal import medfilt

    Y = y.unsqueeze(0) if y.dim() == 1 else y          # [B, T]
    trace = Y.mean(dim=0).detach().cpu().numpy()       # shared slow target, [T]
    T = trace.shape[0]
    t_s = np.arange(T) * float(dt)                     # seconds
    t_inj_s = float(inj_t) * float(dt)

    # median-smooth (~smooth_s) to strip the fast phasic events: the tonic should
    # see only the slow injection-locked trend. Odd window length for medfilt.
    win = int(round(smooth_s / float(dt))) // 2 * 2 + 1
    y_smooth = medfilt(trace, kernel_size=win) if win >= 3 else trace

    def _bateman(t, b, peak, k_decay, gap, d):
        # x_hat(t), reparameterized k_rise = k_decay + gap so a box bound gap > 0
        # enforces the Bateman ordering, and normalized to peak 1 so `peak` is the
        # height of the bump rather than a coefficient multiplying it. Same factored
        # expm1 form as `bateman_shape`, for the same reason: with gap small the
        # naive difference of exponentials is rounding error.
        k_rise = k_decay + gap
        tau = np.clip(t - t_inj_s, 0.0, None)
        post = (t >= t_inj_s).astype(float)
        t_pk = np.log(k_rise / k_decay) / gap
        f = np.exp(-k_decay * tau) * (-np.expm1(-gap * tau))
        f_pk = np.exp(-k_decay * t_pk) * (-np.expm1(-gap * t_pk))
        return b + post * (peak * f / max(f_pk, 1e-12) + d * tau / 3600.0)

    # Data-driven init: pre-injection median for b, post-injection swing for the
    # peak. That swing was always the right starting value -- it is the height of the
    # bump -- and under the old parameterization it was handed to A, which is the
    # height only when the rates are far apart. The optimizer then had a flat
    # direction to leave along, and left.
    b0 = float(np.median(y_smooth[:inj_t])) if inj_t >= 1 else float(np.median(y_smooth))
    swing = float(y_smooth[inj_t:].max() - b0) if inj_t < T else 0.0
    peak0 = float(np.clip(swing, 1e-4, peak_max * 0.99))
    #       b    peak   k_decay  gap    d       (k_rise = k_decay + gap)
    p0 = [b0,   peak0,  2e-3,    3e-3,  0.0]
    lo = [-np.inf, 1e-6, 1e-6, 1e-6, -np.inf]   # peak, k_decay, gap strictly > 0
    hi = [np.inf,  peak_max, np.inf, np.inf, np.inf]

    # Two passes, because the linear drift and the bump compete. `d * tau / 3600` is a
    # ramp, and over this window a SMALL Bateman is close enough to a ramp that the
    # optimizer can hand the whole post-injection rise to the drift and walk the peak
    # onto its lower bound -- returning exactly zero. Measured: with the bump at 303
    # noise standard deviations the pre-fit recovers it on every session of every seed;
    # at 15, which is what the recordings carry, it returns zero on at least one session
    # of every seed and on half the sessions of one. Those sessions then enter SVI with
    # no slow component and never recover.
    #
    # So fit the bump first with the drift pinned at zero, where it has no competitor,
    # and only then release the drift from that solution. If releasing it collapses the
    # peak onto the bound anyway, keep the pass that found a bump.
    def _no_drift(t, b, peak, k_decay, gap):
        return _bateman(t, b, peak, k_decay, gap, 0.0)

    popt4, _ = curve_fit(_no_drift, t_s, y_smooth, p0=p0[:4],
                         bounds=(lo[:4], hi[:4]), maxfev=40000)
    popt, _ = curve_fit(_bateman, t_s, y_smooth,
                        p0=[*popt4, 0.0], bounds=(lo, hi), maxfev=40000)
    b_f, peak_f, kdec_f, gap_f, d_f = (float(v) for v in popt)
    # "Collapsed" rather than "small": a bump genuinely absent in the data should come
    # back as a bump the model can shrink, not as the bound itself.
    if peak_f <= 10.0 * lo[1]:
        b_f, peak_f, kdec_f, gap_f = (float(v) for v in popt4)
        d_f = 0.0

    return TonicBaseline(
        b=b_f, peak=peak_f, k_rise=kdec_f + gap_f, k_decay=kdec_f, drift=d_f, dt=dt,
    )


# Physical bounds on the constrained tonic parameters, applied before the exponential.
#
# These are a statement about pharmacokinetics, not a numerical patch bolted on after
# a crash -- though a crash is what prompted writing them down. An amplitude outside
# [1e-3, 1e3] on a z-scored trace is not a bump, and a rate constant outside
# [1e-6, 1] s^-1 corresponds to a peak time outside roughly [14 s, 8 days]. Nothing
# real lives outside either interval.
#
# Without them, `A = exp(alpha)` overflows to `inf` once alpha passes ~88 in float32,
# and at the injection bin, where tau = 0, the bump evaluates to inf * (1 - 1) =
# inf * 0 = NaN. Measured 2026-09-09: this killed 2 of 8 concurrent cohort fits.
# Clamping is the identity everywhere a real fit lives, so it changes no result that
# was not already broken -- and a fitted value sitting exactly on a boundary is
# visible in `tonic_posterior`, so the failure becomes diagnosable instead of fatal.
# The peak is the height of the tonic bump in the units of the trace, so this box is
# wide enough for both regimes we fit: simulated cohorts peak near 1-2, real dF/F
# recordings near 0.02-0.2. It replaces a bound on the raw amplitude A, which was
# never the right thing to bound -- A is the unidentified half of the old
# parameterization, and the recording that provoked this change had already pinned it
# at its upper limit while producing a perfectly ordinary tonic.
_PEAK_RANGE = (1e-4, 1e2)
_K_RANGE = (1e-6, 1.0)


def _bounded_exp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """exp(x), with x clamped so the result stays inside [lo, hi]."""
    return x.clamp(math.log(lo), math.log(hi)).exp()


class BatchedTonicBaseline(nn.Module):
    """One Bateman baseline per sequence, with per-sequence injection bins.

    `TonicBaseline` holds a single animal's five scalars and takes one injection bin
    for the whole batch. A cohort is neither: animals differ in ambient tone, drugs
    differ in pharmacokinetics, and sessions are injected at different times. This
    class carries the same unconstrained psi = (b, alpha, rho, gamma, d) with a
    leading sequence dimension, and `baseline` takes `inj_t` as a length-S vector.

    The parameters stay **MAP point estimates**, exactly as in the single-trace model.
    The hierarchy in `HierarchicalModel` is over the *rate* coefficients, which are
    the estimand; pooling the tonic as well would be defensible but adds five more
    variance components to a model whose identifiability rests on the timescale
    separation between tonic and phasic, and none of them is the quantity anyone
    reports. Left as a documented extension rather than a silent choice.

    Parameters are given in the constrained space, each as a length-S sequence (or a
    scalar, broadcast to all S).
    """

    def __init__(
        self,
        *,
        n_seq: int,
        b=0.0,
        peak=1.0,
        k_rise=0.01,
        k_decay=0.002,
        drift=0.0,
        dt: float = 1.0 / 25.0,
    ):
        super().__init__()
        if n_seq < 1:
            raise ValueError(f"n_seq must be >= 1 (got {n_seq})")

        def vec(x, name):
            t = torch.as_tensor(x, dtype=torch.get_default_dtype()).reshape(-1)
            if t.numel() == 1:
                t = t.expand(n_seq).clone()
            if t.numel() != n_seq:
                raise ValueError(f"{name} must be a scalar or length {n_seq} (got {t.numel()})")
            return t

        b_, A_, kr_, kd_, d_ = (vec(x, n) for x, n in
                                ((b, "b"), (peak, "peak"), (k_rise, "k_rise"),
                                 (k_decay, "k_decay"), (drift, "drift")))
        if (A_ <= 0).any():
            raise ValueError(
                "every peak must be > 0; peak = exp(alpha) is positive by construction"
            )
        if not ((kr_ > kd_) & (kd_ > 0)).all():
            raise ValueError(
                "every sequence needs k_rise > k_decay > 0 (Bateman ordering: absorption "
                "faster than elimination)"
            )
        self.dt = float(dt)
        self.n_seq = int(n_seq)
        self.b = nn.Parameter(b_)
        self.alpha = nn.Parameter(A_.log())
        self.rho = nn.Parameter(kd_.log())
        self.gamma = nn.Parameter((kr_ - kd_).log())
        self.drift = nn.Parameter(d_)

    # ------------------------------------------------------- constrained params
    @property
    def peak(self) -> torch.Tensor:
        """Bump peak heights = exp(alpha) > 0, [S], in the traces' units."""
        return self.alpha.exp()

    @property
    def A(self) -> torch.Tensor:
        """The textbook Bateman amplitudes these fits correspond to, [S]. Derived.

        Diagnostic only, as on `TonicBaseline`: an ordinary peak beside an enormous A
        is a session whose rates have drifted together.
        """
        return bateman_raw_amplitude(self.peak, self.k_decay, self.k_rise)

    @property
    def k_decay(self) -> torch.Tensor:
        """Elimination rates k_decay = exp(rho) > 0, [S]."""
        return self.rho.exp()

    @property
    def k_rise(self) -> torch.Tensor:
        """Absorption rates k_rise = exp(rho) + exp(gamma) > k_decay, [S]."""
        return self.rho.exp() + self.gamma.exp()

    # ----------------------------------------------------------------- baseline
    def _clock(self, T: int, inj_t) -> tuple[torch.Tensor, torch.Tensor]:
        """(post-injection gate, post-injection clock in seconds), both [S, T]."""
        dev, dt_ = self.b.device, self.b.dtype
        inj = torch.as_tensor(inj_t, device=dev).reshape(-1).to(dt_)
        if inj.numel() == 1:
            inj = inj.expand(self.n_seq)
        if inj.numel() != self.n_seq:
            raise ValueError(f"inj_t must be a scalar or length {self.n_seq} (got {inj.numel()})")
        t = torch.arange(T, device=dev, dtype=dt_).unsqueeze(0)      # [1, T]
        inj = inj.unsqueeze(-1)                                      # [S, 1]
        return (t >= inj).to(dt_), (t - inj).clamp_min(0) * self.dt

    def baseline_from(self, b, alpha, rho, gamma, drift, T: int, inj_t) -> torch.Tensor:
        """x_hat from *given* unconstrained parameters, [S, T].

        The same curve as `baseline`, but driven by tensors the caller supplies
        instead of this module's own `nn.Parameter`s. That is what lets the
        hierarchical model put the pharmacokinetics under a prior: the tonic block
        stays the place the Bateman curve is defined, and the model passes in values
        sampled from the drug-level hierarchy rather than re-deriving the algebra.

        Each of `b, alpha, rho, gamma, drift` is [S] on the unconstrained axis, with
        the same reparameterization the module uses: peak = exp(alpha),
        k_decay = exp(rho), k_rise = exp(rho) + exp(gamma), so positivity and the
        Bateman ordering hold for any real input and no constraint is needed on the
        prior.
        """
        post, tau = self._clock(T, inj_t)
        peak = _bounded_exp(alpha, *_PEAK_RANGE).unsqueeze(-1)
        kd = _bounded_exp(rho, *_K_RANGE).unsqueeze(-1)
        kr = kd + _bounded_exp(gamma, *_K_RANGE).unsqueeze(-1)
        bump = peak * bateman_shape(tau, kd, kr)
        return b.unsqueeze(-1) + post * (
            bump + drift.unsqueeze(-1) * tau / _DRIFT_SECONDS)

    def baseline(self, T: int, inj_t: torch.Tensor) -> torch.Tensor:
        """x_hat over t = 0..T-1 for per-sequence injection bins. Returns [S, T].

        `inj_t` is a length-S integer tensor (a scalar is broadcast). The
        post-injection clock is clamped at 0 per sequence so the pre-injection region
        carries no spurious exp(-k * negative).
        """
        return self.baseline_from(self.b, self.alpha, self.rho, self.gamma,
                                  self.drift, T, inj_t)

    def forward(self, T: int, inj_t: torch.Tensor) -> torch.Tensor:
        """Alias for `baseline` so the module is callable."""
        return self.baseline(T, inj_t)


def stack_tonic_baselines(blocks: list[TonicBaseline]) -> BatchedTonicBaseline:
    """Assemble per-trace `fit_tonic_baseline` pre-fits into one batched block.

    The pre-fit stays per-trace -- it is a least-squares initializer, and each
    recording has its own bleaching and its own pharmacokinetic bump. This just
    stacks the results so SVI refines all of them together.
    """
    if not blocks:
        raise ValueError("need at least one TonicBaseline to stack")
    dts = {b.dt for b in blocks}
    if len(dts) != 1:
        raise ValueError(f"all blocks must share one dt (got {sorted(dts)})")
    g = lambda f: torch.stack([f(b).detach().reshape(()) for b in blocks])  # noqa: E731
    return BatchedTonicBaseline(
        n_seq=len(blocks),
        b=g(lambda b: b.b),
        # `b.peak`, NOT `b.A` -- since the reparameterization, `A` is a DERIVED
        # textbook amplitude and feeding it back in as the stored parameter would
        # rescale every pre-fit by its own peak factor.
        peak=g(lambda b: b.peak),
        k_rise=g(lambda b: b.k_rise),
        k_decay=g(lambda b: b.k_decay),
        drift=g(lambda b: b.drift),
        dt=blocks[0].dt,
    )
