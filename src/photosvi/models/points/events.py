"""Phasic carrier of the points family: events -> phasic fluorescence c_t.

The forward-generative block of the points model's phasic side:

    s_t ~ Bernoulli(p_t),  p_t = 1 - exp(-lambda_t * dt)
    log lambda_t = beta0 + beta_inj*1[t>=t_inj] + sum_l w_l s_{t-l}
    a_t ~ LogNormal(mu_a, sigma_a^2)
    u_t = s_t * a_t
    c_t = (h * u)_t = sum_k h_k u_{t-k}

Generative side only: a `sample` forward pass and the exact timing
log-likelihood. Posterior inference over the discrete s_{1:T} lives in the
inference layer, so there is no Pyro `model`/`guide` here.

The kernel `h` is injected, not hard-coded -- any causal, shift-invariant
kernel is valid. `bi_exponential_kernel` (rise+decay sensor shape) is the one
used throughout.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def bi_exponential_kernel(
    length: int,
    k_rise: float,
    k_decay: float,
    dt: float = 1.0 / 25.0,
    normalize: str = "peak",
) -> torch.Tensor:
    """Causal difference-of-exponentials transient kernel h.

    h_k = exp(-k_decay * k*dt) - exp(-k_rise * k*dt), the slow minus the fast
    exponential. Nonnegative and zero at lag 0 exactly when k_rise > k_decay
    (the `clamp_min` guards a misconfigured pair). Same bi-exponential shape as
    the tonic baseline (the double-exponential drug-response curve) but at the *transient*
    timescale -- the two uses share no parameters.

    `length` is the kernel support in bins; `normalize` rescales so the peak is
    1 ("peak"), the area is 1 ("sum"), or the area is *zero* ("zerosum" --
    mean-removed, then peak-normalized). Peak-normalization makes the mark a_t
    read directly as the transient's height; under "zerosum" the mark reads as
    the height above the kernel's own mean, so mu_a shifts scale slightly.

    "zerosum" is not a claim about the sensor -- a real transient does not
    undershoot. It is an *identifiability* choice: with Sum(h) != 0 the phasic
    component is non-negative, so its mean grows with the event rate and the
    tonic baseline is pushed to the floor of the trace rather than through its
    middle. Because that absorbed offset steps up at the injection (E[c_t]
    scales with lambda_t) it lands in the same place as the drug's effect on
    the tonic, and beta_inj is biased. Sum(h) = 0 removes the channel.
    """
    k = torch.arange(length, dtype=torch.float32)
    tau = k * dt
    h = torch.exp(-k_decay * tau) - torch.exp(-k_rise * tau)
    h = h.clamp_min(0.0)
    if normalize == "peak":
        peak = h.max()
        if peak > 0:
            h = h / peak
    elif normalize == "sum":
        total = h.sum()
        if total > 0:
            h = h / total
    elif normalize == "zerosum":
        # Mean-remove, then peak-normalize the positive lobe. Sum(h) == 0 makes the
        # phasic carrier mean-free: E[c_t] = 0 for any intensity, so the event rate
        # no longer contributes to the signal mean and cannot compete with the tonic
        # for it. See the "mean-free phasic basis" note in the README.
        h = h - h.mean()
        peak = h.max()
        if peak > 0:
            h = h / peak
    return h


def biphasic_kernel(
    length: int,
    k_rise_fast: float,
    k_decay_fast: float,
    k_rise_slow: float,
    k_decay_slow: float,
    dt: float = 1.0 / 25.0,
) -> torch.Tensor:
    """Fast positive lobe minus a scaled slower lobe, the scale set so the areas cancel.

    h = f - (sum f / sum s) * s, then peak-normalized. Both lobes are ordinary
    difference-of-exponential transients, so both decay: unlike the "zerosum"
    normalization of `bi_exponential_kernel`, which buys sum(h) = 0 with a flat DC
    offset that runs to the end of the support, this returns to zero and an event's
    influence genuinely ends.

    Why sum(h) = 0 at all: with a non-negative kernel the phasic carrier c_t >= 0, so
    E[c_t] grows with the event rate and the tonic baseline is identified against the
    floor of the trace rather than its middle. Worse, that absorbed offset steps up at
    the injection (E[c_t] scales with lambda_t), landing in the same place as the
    drug's effect on the tonic and biasing beta_inj. A mean-free basis removes the
    channel: E[c_t] = 0 for any intensity.

    Parameters
    ----------
    k_rise_fast, k_decay_fast : the positive lobe; must satisfy k_rise > k_decay.
    k_rise_slow, k_decay_slow : the subtracted lobe. Must be genuinely slower
        (`k_decay_slow < k_decay_fast`) or the difference has no positive lobe left.
    """
    if k_rise_fast <= k_decay_fast:
        raise ValueError(f"fast lobe needs k_rise > k_decay (got {k_rise_fast}, {k_decay_fast})")
    if k_rise_slow <= k_decay_slow:
        raise ValueError(f"slow lobe needs k_rise > k_decay (got {k_rise_slow}, {k_decay_slow})")
    if k_decay_slow >= k_decay_fast:
        raise ValueError(
            f"the subtracted lobe must decay slower than the fast one "
            f"(got k_decay_slow={k_decay_slow} >= k_decay_fast={k_decay_fast})"
        )
    fast = bi_exponential_kernel(length, k_rise_fast, k_decay_fast, dt=dt)
    slow = bi_exponential_kernel(length, k_rise_slow, k_decay_slow, dt=dt)
    h = fast - (fast.sum() / slow.sum()) * slow
    peak = h.max()
    if peak <= 0:
        raise ValueError("no positive lobe survives the subtraction; lobes are too close")
    return h / peak


def gamma_lobe(
    length: int,
    shape: float,
    rate: float,
    dt: float = 1.0 / 25.0,
) -> torch.Tensor:
    """A gamma density on t >= 0, peak-normalized: t^(shape-1) exp(-rate t).

    The alternative to a difference of exponentials as a single lobe. Its onset is
    smooth rather than kinked, which is what the recordings show: measured against the
    peak-triggered average of 476 isolated transients, the sharp onset of a
    difference-of-exponentials is the largest single term in the residual.

    `shape` must exceed 1 or the density is unbounded at the origin and there is no
    peak to normalize by; shape = 1 is the pure exponential, which starts at its
    maximum and has no rise at all.
    """
    if shape <= 1.0:
        raise ValueError(f"gamma lobe needs shape > 1 for a finite peak (got {shape})")
    if rate <= 0.0:
        raise ValueError(f"gamma lobe needs rate > 0 (got {rate})")
    t = torch.arange(length, dtype=torch.float32) * dt
    # in logs, so a large shape does not overflow t**(shape-1) before the exponential
    # pulls it back down
    log_h = (shape - 1.0) * torch.log(t.clamp_min(1e-12)) - rate * t
    log_h[0] = -float("inf")                      # t = 0 is exactly zero for shape > 1
    return torch.exp(log_h - log_h.max())


def bigamma_kernel(
    length: int,
    shape_fast: float,
    rate_fast: float,
    shape_slow: float,
    rate_slow: float,
    dt: float = 1.0 / 25.0,
    undershoot_tol: float = 0.05,
) -> torch.Tensor:
    """Fast gamma lobe minus a scaled slower one, the scale set so the areas cancel.

    Same construction and the same reason as `biphasic_kernel` -- see there for why
    sum(h) = 0 is not optional -- with gamma lobes in place of differences of
    exponentials. Fitted to the peak-triggered average of the real transients it halves
    the residual (RMS 0.062 / 0.071 against 0.116 / 0.137) and, unlike the biphasic
    family, its parameters are identified rather than lying on a flat ridge.

    **This family needs its support**, which the biphasic one does not. There the slow
    lobe is rescaled over whatever length it is given and the shape barely moves
    (`test_the_biphasic_basis_is_mean_free_at_any_length`); here, truncating before the
    slow lobe has decayed forces a large subtraction and the undershoot grows to swallow
    the transient -- at 3.1 s for the fitted parameters it reaches 1.1x the peak, with
    sum(h) = 0 holding the whole way, so nothing about the mean-free property warns you.
    Hence `undershoot_tol`, which checks that failure directly rather than through a
    proxy: the kernel is built again on a support long enough to have converged, and if
    truncation deepens the undershoot by more than this fraction, this raises instead of
    returning it.

    Parameters
    ----------
    shape_fast, rate_fast : the positive lobe.
    shape_slow, rate_slow : the subtracted lobe, which must have the slower tail
        (`rate_slow < rate_fast`).
    undershoot_tol : how much deeper than its converged value the undershoot may get.
        Pass 0 to skip the check, which is only sensible when you are deliberately
        studying the truncated shape.
    """
    if rate_slow >= rate_fast:
        raise ValueError(
            f"the subtracted lobe must have the slower tail "
            f"(got rate_slow={rate_slow} >= rate_fast={rate_fast})"
        )
    h = _bigamma_raw(length, shape_fast, rate_fast, shape_slow, rate_slow, dt)
    if undershoot_tol > 0:
        need, converged = _bigamma_support(shape_fast, rate_fast, shape_slow, rate_slow,
                                           dt, undershoot_tol)
        if need > length:
            raise ValueError(
                f"{length} taps ({length * dt:.2f} s) truncates this kernel: the "
                f"undershoot reaches {-h.min():.3f} of the peak against "
                f"{-converged:.3f} converged. Use at least {need} taps "
                f"({need * dt:.2f} s), or pass undershoot_tol=0 to study the "
                f"truncated shape deliberately."
            )
    return h


def _bigamma_raw(length, shape_fast, rate_fast, shape_slow, rate_slow, dt):
    """The mean-free difference, peak-normalized, with no support check."""
    fast = gamma_lobe(length, shape_fast, rate_fast, dt=dt)
    slow = gamma_lobe(length, shape_slow, rate_slow, dt=dt)
    h = fast - (fast.sum() / slow.sum()) * slow
    peak = h.max()
    if peak <= 0:
        raise ValueError("no positive lobe survives the subtraction; lobes are too close")
    return h / peak


def _bigamma_support(shape_fast, rate_fast, shape_slow, rate_slow, dt, tol):
    """`(taps, converged_undershoot)`: the shortest support within `tol` of converged.

    Converged is measured on a support eight times the slow lobe's mean, which is far
    past where its tail matters; the answer is then found by bisection so the number in
    the error message is advice worth following rather than the next power of two.
    """
    far = max(64, int(8.0 * shape_slow / rate_slow / dt))
    target = _bigamma_raw(far, shape_fast, rate_fast, shape_slow, rate_slow, dt).min()
    ok = lambda n: _bigamma_raw(n, shape_fast, rate_fast, shape_slow, rate_slow,  # noqa: E731
                                dt).min() >= target * (1.0 + tol)
    if ok(8):
        return 8, float(target)
    lo, hi = 8, far
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi, float(target)


def build_kernel(spec: dict, length: int, dt: float) -> torch.Tensor:
    """Build a kernel from a config dict, dispatching on `family`.

    `{"family": "biexp", "k_rise":, "k_decay":, "normalize":}` -> `bi_exponential_kernel`
    `{"family": "biphasic", "k_rise_fast":, "k_decay_fast":, "k_rise_slow":,
      "k_decay_slow":}` -> `biphasic_kernel`
    `{"family": "bigamma", "shape_fast":, "rate_fast":, "shape_slow":, "rate_slow":,
      "undershoot_tol":}` -> `bigamma_kernel`

    One factory so the generating and fitted kernels are always built the same way and
    a config cannot silently disagree with itself about which family it means.
    """
    fam = spec.get("family", "biexp")
    if fam == "biexp":
        return bi_exponential_kernel(
            length, k_rise=spec["k_rise"], k_decay=spec["k_decay"], dt=dt,
            normalize=spec.get("normalize", "peak"),
        )
    if fam == "biphasic":
        return biphasic_kernel(
            length, k_rise_fast=spec["k_rise_fast"], k_decay_fast=spec["k_decay_fast"],
            k_rise_slow=spec["k_rise_slow"], k_decay_slow=spec["k_decay_slow"], dt=dt,
        )
    if fam == "bigamma":
        return bigamma_kernel(
            length, shape_fast=spec["shape_fast"], rate_fast=spec["rate_fast"],
            shape_slow=spec["shape_slow"], rate_slow=spec["rate_slow"], dt=dt,
            undershoot_tol=spec.get("undershoot_tol", 0.05),
        )
    raise ValueError(
        f"unknown kernel family {fam!r}; known: 'biexp', 'biphasic', 'bigamma'")


class PhasicEventProcess(nn.Module):
    """Discrete-time marked point process driving phasic fluorescence c_t.

    Holds the conditional-intensity GLM parameters (beta0, beta_inj, history
    FIR w), the log-normal mark parameters (mu_a, sigma_a), and a fixed causal
    kernel `h` registered as a buffer (no grad, follows the module's device).

    `beta0` / `beta_inj` are **buffers, not nn.Parameters**: under SingleTraceModel
    SVI they are latent globals (Normal prior + mean-field posterior), so they
    are not point-optimized here. The buffer carries the value the generative
    `.sample()` / `log_intensity` path uses, and serves as the default
    prior-mean anchor SingleTraceModel reads. The marks (mu_a, sigma_a) and FIR w
    stay MAP nn.Parameters.

    Parameters
    ----------
    kernel : 1-D tensor h, the fixed transient shape (e.g. from
        `bi_exponential_kernel`).
    beta0 : baseline log-rate (generative value / prior-mean anchor).
    beta_inj : additive post-injection log-rate shift (anchor).
    history : optional FIR coefficients w_1..w_L over own past events; w_l < 0
        is refractory, w_l > 0 self-exciting. None / [] disables history.
    mu_a, sigma_a : log-normal mark location and (positive) scale; sigma_a is
        stored in log space so it stays positive.
    dt : bin width in seconds.
    """

    def __init__(
        self,
        kernel: torch.Tensor,
        *,
        beta0: float = -2.0,
        beta_inj: float = 0.0,
        history: list[float] | torch.Tensor | None = None,
        mu_a: float = 0.0,
        sigma_a: float = 0.5,
        dt: float = 1.0 / 25.0,
    ):
        super().__init__()
        self.dt = float(dt)
        self.register_buffer("kernel", torch.as_tensor(kernel, dtype=torch.float32))

        # latent under SingleTraceModel SVI -> buffers, not learnable params (see docstring)
        self.register_buffer("beta0", torch.tensor(float(beta0)))
        self.register_buffer("beta_inj", torch.tensor(float(beta_inj)))

        w = torch.as_tensor(history if history is not None else [], dtype=torch.float32)
        self.history_len = int(w.numel())
        self.w = nn.Parameter(w) if self.history_len > 0 else None

        self.mu_a = nn.Parameter(torch.tensor(float(mu_a)))
        self.log_sigma_a = nn.Parameter(torch.tensor(float(sigma_a)).log())
        # Prior on the mark distribution, None for none (the behaviour before
        # 2026-10-09). With no prior mu_a and sigma_a are point estimates the optimizer
        # moves freely, and at every SNR tested it trades amplitude for rate: mu_a falls
        # 0.6-1.2 below its simulated value and beta0 rises 0.7-2 above it, since the
        # data fix rate x amplitude^2 far better than either. `set_mark_prior` anchors
        # them on an independent estimate (peak prominences) with a stated width.
        self.mark_prior: tuple[float, float, float, float] | None = None
        # Where q(a)'s location starts: "noise" (min(mu_a, log sigma_y), the shipped
        # rule) or "prior" (mu_a).
        self.q_a_init = "noise"

    def set_mark_prior(self, mu: float, sigma: float, *, mu_sd: float = 0.2,
                       log_sigma_sd: float = 0.3) -> None:
        """mu_a ~ N(mu, mu_sd^2), log sigma_a ~ N(log sigma, log_sigma_sd^2); starts there."""
        self.mark_prior = (float(mu), float(mu_sd), math.log(float(sigma)), float(log_sigma_sd))
        with torch.no_grad():
            self.mu_a.fill_(float(mu))
            self.log_sigma_a.fill_(math.log(float(sigma)))

    def mark_prior_logp(self) -> torch.Tensor:
        """Log density of the mark prior at the current (mu_a, log sigma_a)."""
        m, ms, ls, lss = self.mark_prior
        return (-0.5 * ((self.mu_a - m) / ms) ** 2
                - 0.5 * ((self.log_sigma_a - ls) / lss) ** 2)

    @property
    def sigma_a(self) -> torch.Tensor:
        return self.log_sigma_a.exp()

    # ------------------------------------------------------------- intensity
    def log_intensity(self, s: torch.Tensor, post_injection: torch.Tensor) -> torch.Tensor:
        """log lambda_t for a given event tensor.

        `s` is [B, T] in {0, 1}; `post_injection` is the 1[t >= t_inj] mask,
        broadcastable to [B, T]. Returns log lambda of shape [B, T]. The history
        term sum_l w_l s_{t-l} is built by shifting `s` right by each lag l
        (the events at t-l), so the current bin never feeds its own rate.
        """
        log_lam = self.beta0 + self.beta_inj * post_injection
        if self.history_len > 0:
            log_lam = log_lam + self._history_term(s)
        return log_lam.expand_as(s) if log_lam.dim() == 0 else log_lam

    def _history_term(self, s: torch.Tensor) -> torch.Tensor:
        """sum_{l=1}^{L} w_l s_{t-l} for all t at once (right-shift per lag)."""
        B, T = s.shape
        hist = s.new_zeros(B, T)
        for lag in range(1, self.history_len + 1):
            shifted = F.pad(s, (lag, 0))[:, :T]  # s_{t-lag}, zero-filled at the start
            hist = hist + self.w[lag - 1] * shifted
        return hist

    # ------------------------------------------------------------- likelihood
    def event_logprob(self, s: torch.Tensor, post_injection: torch.Tensor) -> torch.Tensor:
        """Per-sequence timing log-likelihood log p(s_{1:T} | theta).

            sum_t [ s_t log p_t + (1 - s_t) log(1 - p_t) ],
            log(1 - p_t) = -lambda_t * dt,  p_t = 1 - exp(-lambda_t * dt).

        Uses expm1 for the occupied/empty split so both terms stay stable at
        small lambda*dt. Returns shape [B] (summed over time).
        """
        log_lam = self.log_intensity(s, post_injection)
        lam_dt = log_lam.exp() * self.dt
        log_1mp = -lam_dt                  # log(1 - p_t) = -lambda_t dt
        p = -torch.expm1(-lam_dt)          # 1 - exp(-lambda_t dt), stable
        log_p = p.clamp_min(1e-12).log()
        ll = s * log_p + (1.0 - s) * log_1mp
        return ll.sum(dim=-1)

    # ------------------------------------------------------------- convolution
    def convolve(self, u: torch.Tensor) -> torch.Tensor:
        """c_t = sum_{k=0}^{M-1} h_k u_{t-k}, the causal FIR convolution.

        `u` is the drive [B, T]; returns c [B, T]. Implemented as a conv1d with
        the kernel flipped and the input left-padded by M-1 so the output is
        causal (each c_t depends only on u at t and earlier).
        """
        M = self.kernel.numel()
        weight = self.kernel.flip(0).view(1, 1, M)
        u_in = F.pad(u, (M - 1, 0)).unsqueeze(1)
        c = F.conv1d(u_in, weight).squeeze(1)
        return c

    # ------------------------------------------------------------- generative
    @torch.no_grad()
    # Forward generative process.
    def sample(
        self,
        T: int,
        inj_t: int,
        batch: int = 1,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward-generate the phasic block.

        Walks t = 0..T-1 sampling s_t ~ Bernoulli(p_t) with the rate built from
        the events drawn so far (history is sequential), then draws all marks
        a_t ~ LogNormal at once (marks are independent of timing), forms the
        drive u_t = s_t a_t, and convolves with h. Returns a dict with `s`, `a`,
        `u`, `c`, each [batch, T].
        """
        device = self.kernel.device
        s = torch.zeros(batch, T, device=device) # event indicator to be filled
        post = (torch.arange(T, device=device) >= inj_t).float() # Sets 1 at and after the injection time

        for t in range(T):
            # Building conditional intensity
            log_lam = self.beta0 + self.beta_inj * post[t] # Indicator for post-injection time
            if self.history_len > 0:
                lo = max(0, t - self.history_len)
                past = s[:, lo:t]  # columns oldest -> newest (lag L .. lag 1)
                if past.shape[1] > 0:
                    w_slice = self.w[: past.shape[1]].flip(0)  # align newest -> w_1
                    log_lam = log_lam + (past * w_slice).sum(-1)
            lam_dt = log_lam.exp() * self.dt
            p = -torch.expm1(-lam_dt)  # 1 - exp(-lambda_t dt)
            s[:, t] = torch.bernoulli(p.expand(batch), generator=generator)

        eps = torch.randn(batch, T, generator=generator, device=device)
        a = (self.mu_a + self.sigma_a * eps).exp()  # LogNormal marks
        u = s * a                                    # drive
        c = self.convolve(u)                         # phasic fluorescence
        return {"s": s, "a": a, "u": u, "c": c}


def estimate_log_baseline_rate(
    y: torch.Tensor,
    inj_t: int,
    *,
    dt: float = 1.0 / 25.0,
    threshold: float = 2.0,
    min_rate: float = 1e-3,
) -> float:
    """Data-driven anchor for beta0: the empirical pre-injection log event-rate.

    A coarse proxy for the baseline log-intensity beta0 used to seed the prior
    mean (and guide loc) of SingleTraceModel's Bayesian phasic GLM. It counts the
    pre-injection *transients* of `y` -- local maxima rising more than
    `threshold` above the pre-injection median (the trace is z-scored dF/F, so
    `threshold` is in z-units) -- and converts the count to a rate:

        r_hat = n_peaks / (B * n_pre_bins * dt),   beta0 ~ log(max(r_hat, min_rate)).

    `y` is [B, T] or [T]; only bins t < inj_t are used. With multiple sequences
    the peaks are counted **per sequence and pooled** over the total
    pre-injection duration `B * n_pre_bins * dt` -- *not* on the averaged trace,
    which would cancel events landing at different bins across sequences. Each
    event is assumed to raise one peak, so overlapping transients undercount --
    fine for an *init* anchor, not a fit. `min_rate` floors an empty pre-window
    so the log is finite. Returns a Python float (the log-rate).

    Note this makes the beta0 prior weakly **empirical-Bayes** (its location is
    read off the data); the prior *scale* stays a fixed weakly-informative
    choice set on SingleTraceModel.
    """
    Y = y.unsqueeze(0) if y.dim() == 1 else y  # [B, T]
    pre = Y[:, :inj_t]                         # [B, n_pre]
    if pre.shape[1] < 3:
        return float(torch.log(torch.tensor(min_rate)))
    level = pre - pre.median(dim=1, keepdim=True).values  # detrend each sequence
    # interior local maxima above threshold: y_t > y_{t-1} and y_t >= y_{t+1}
    up = level[:, 1:-1] > level[:, :-2]
    down = level[:, 1:-1] >= level[:, 2:]
    is_peak = up & down & (level[:, 1:-1] > threshold)
    n_peaks = int(is_peak.sum().item())
    duration = pre.shape[0] * pre.shape[1] * float(dt)  # pooled over sequences
    rate = max(n_peaks / duration, float(min_rate))
    return float(torch.log(torch.tensor(rate)))
