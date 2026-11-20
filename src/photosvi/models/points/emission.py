"""Emission node of the points family: additive-mean Gaussian / Student-t observation.

The two carriers -- phasic c_t (events.py) and tonic x_hat(t) (tonic.py) -- are
deterministic nodes that add no density; ALL the observation density enters
here, through a single emission factor on their additive mean
m_t = x_hat(t) + c_t:

    Gaussian    y_t ~ Normal(m_t, sigma_y^2)
    Student-t   y_t ~ StudentT(nu, m_t, sigma_y)

The Student-t is the robust variant (Lange, Little & Taylor 1989): its
polynomial tails grow the NLL only *logarithmically* in the residual, so a
motion-artefact outlier is down-weighted instead of dominating the squared
error and biasing the carriers. As nu -> inf it recovers the Gaussian;
small nu (3-7) gives heavy tails.

Its Gamma-precision scale-mixture representation (West 1987, eqs 4-5) -- one
auxiliary precision w_t ~ Gamma(nu/2, nu/2) per bin, conditionally Gaussian
given w_t with inflated variance sigma_y^2 / w_t -- is what SVI exploits. That
auxiliary latent belongs to the model/guide, so for the *marginal* emission
density `log_prob` uses StudentT directly, which is exact; only the forward
`sample` materializes w (to draw a trace), and exposes it.

Like events.py / tonic.py this is a single building block, not a Pyro
model/guide: it exposes the emission log-likelihood block of the complete-data
joint and a forward `sample` for the generative loop. The mean m_t is
formed by the caller as x_hat + c; this node never sees the carriers, only
their sum -- exactly the additive-component (GAM) structure whose no-concurvity
property keeps the tonic and phasic from trading off.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Gamma, Normal, StudentT


def _ar_is_stationary(phi, margin: float = 1e-8) -> bool:
    """Every AR pole strictly inside the unit circle, with room to spare.

    The margin is not pedantry: phi = (1.9, -0.9) has a pole at exactly 1, which
    floating point returns as 0.9999999999999998 and a bare `< 1` accepts. That is a
    random walk -- no stationary variance, so nothing to scale to sigma_y, and the
    autocovariance solve below returns nonsense rather than failing.
    """
    import numpy as np
    if not phi:
        return True
    poles = np.roots(np.concatenate([[1.0], -np.asarray(phi, float)]))
    return bool(np.all(np.abs(poles) < 1.0 - margin))


def ar_conditional(obs, mean, scale, phi):
    """`(cond_mean, cond_scale)` for an AR(p) residual, exact at every bin.

    The AR(p) joint factorises as p(y_1) prod_t p(y_t | y_{t-1}, ..., y_{t-p}), and
    because y is observed each factor is an ordinary Normal: the mean carries the p
    previous residuals, the scale is the one-step prediction error. Past bin p those are
    `phi` and the innovation scale; BEFORE it they are not, because only t previous
    values are available -- so each of the first p bins gets its own regression, solved
    from the autocovariances. Scoring that block under the marginal instead is the usual
    shortcut and it is wrong by exactly the amount the first bins are correlated.

    Returns tensors broadcastable to `mean`, so the caller keeps its per-bin site.
    """
    import numpy as np
    p = len(phi)
    T = mean.shape[-1]
    if p == 0 or T <= 1:
        return mean, scale * torch.ones_like(mean)
    g = ar_autocovariance(phi, max(p + 1, 2))        # unit marginal variance
    r = obs - mean
    cm = mean.clone()
    sd = torch.ones_like(mean)
    for t in range(1, min(p, T)):                    # the ramp-up block, exactly
        G = np.array([[g[abs(i - j)] for j in range(t)] for i in range(t)])
        a = np.linalg.solve(G, g[1:t + 1])
        var = float(g[0] - a @ g[1:t + 1])
        for k in range(t):
            cm[..., t] = cm[..., t] + float(a[k]) * r[..., t - 1 - k]
        sd[..., t] = var**0.5
    if T > p:                                        # the stationary block
        innov = float(g[0] - np.dot(phi, g[1:p + 1]))
        acc = torch.zeros_like(mean[..., p:])
        for k in range(p):
            acc = acc + float(phi[k]) * r[..., p - 1 - k: T - 1 - k]
        cm[..., p:] = cm[..., p:] + acc
        sd[..., p:] = innov**0.5
    return cm, scale * sd


def ar_autocovariance(phi, n: int):
    """Stationary autocovariances gamma(0..n-1) for unit MARGINAL variance.

    Needed in two places and easy to get wrong in both. The conditional scale of an
    AR(p) is not `sqrt(1 - rho^2)` past p = 1, and the first p bins are not independent
    draws from the marginal -- they are a correlated block, and scoring them as if they
    were is the standard sloppy AR likelihood. Solving Yule-Walker once gives the exact
    quantity for both.
    """
    import numpy as np
    p = len(phi)
    if p == 0:
        g = np.zeros(max(n, 1)); g[0] = 1.0
        return g
    phi = np.asarray(phi, float)
    # Yule-Walker for gamma(0..p) at unit innovation variance:
    #   gamma(i) - sum_k phi_k gamma(|i - k|) = [i == 0]
    A = np.zeros((p + 1, p + 1))
    for i in range(p + 1):
        A[i, i] += 1.0
        for k in range(1, p + 1):
            A[i, abs(i - k)] -= phi[k - 1]
    rhs = np.zeros(p + 1); rhs[0] = 1.0
    g = np.linalg.solve(A, rhs)                      # gamma(0..p) for unit innovation
    out = np.zeros(max(n, p + 1))
    out[: p + 1] = g
    for t in range(p + 1, len(out)):
        out[t] = sum(phi[k] * out[t - 1 - k] for k in range(p))
    return (out / out[0])[:n]                        # rescaled to unit MARGINAL variance


class AdditiveEmission(nn.Module):
    """Additive-mean emission y_t ~ Normal / StudentT about m_t = x_hat + c.

    Holds the homoscedastic residual scale sigma_y (stored in log space so it
    stays positive, mirroring PhasicEventProcess's log_sigma_a) and, for the
    robust variant, the Student-t degrees of freedom nu (also log-stored). The
    node carries no carrier of its own: callers pass the additive mean
    m = x_hat + c and the observation y, both broadcastable to [B, T].

    Parameters
    ----------
    sigma_y : residual standard deviation, must be > 0. Stored as log_sigma_y.
        MAP-optimized by default; `SingleTraceModel` can instead treat it as a sampled
        latent under a narrow data-anchored prior (see `estimate_log_noise_scale`
        and `SingleTraceModel(sigma_y_prior_loc=...)`), which freezes this param.
    ar_rho : lag-one correlation of the residuals. Zero, the default, is the
        independent emission this class had until 2026-09-30 and reproduces every
        earlier result exactly. A positive value makes the residual an AR(1) process:

            y_t = m_t + e_t,   e_t = rho e_{t-1} + sqrt(1 - rho^2) sigma_y eps_t

        so the marginal scale stays sigma_y and only the correlation changes. It
        exists because real recordings do not have white residuals: on the raw
        `.doric` files the first differences of the trace correlate at +0.8 where
        white noise gives -0.5, and the effective sample size of a pre-injection
        window is about a tenth of its nominal length. An independent emission counts
        15,000 observations where there are the information of 1,400, and the
        direction of that error is over-confidence.

        It also replaces, rather than complements, a slow component added to the mean:
        `y = m + AR(1)` and `y = m + OU + white` describe the same distribution, and
        carrying both would be modelling one mechanism twice. Keeping it in the
        emission rather than the mean matters for what the fit does with it -- noise
        is not something the model tries to explain with events.
    nu : Student-t degrees of freedom. None selects the Gaussian emission;
        a positive value selects the robust Student-t with that nu.
        Stored as log_nu when present. Small nu (3-7) is heavy-tailed; large nu
        approaches the Gaussian.
    """

    def __init__(self, *, sigma_y=0.1, nu: float | None = None,
                 ar_rho: float = 0.0, ar_coeffs=None):
        super().__init__()
        # `sigma_y` is one scale for every sequence, or a sequence of them -- one per
        # session of a cohort, broadcast over its bins (2026-10-20). A float keeps the
        # scalar parameter every earlier fit used.
        sig = torch.as_tensor(sigma_y, dtype=torch.float32)
        if sig.dim() > 1 or not bool((sig > 0).all()):
            raise ValueError(
                f"sigma_y must be > 0 (got {sigma_y}); it is stored as log_sigma_y"
            )
        if ar_coeffs is not None and ar_rho:
            raise ValueError("give ar_rho or ar_coeffs, not both")
        phi = tuple(float(v) for v in (ar_coeffs if ar_coeffs is not None else
                                       ((ar_rho,) if ar_rho else ())))
        if phi and not _ar_is_stationary(phi):
            raise ValueError(
                f"AR coefficients {phi} are not stationary: a root of "
                f"1 - sum phi_k z^-k lies on or inside the unit circle, so the process "
                f"has no marginal variance to scale to sigma_y"
            )
        self.log_sigma_y = nn.Parameter(sig.log())
        self.ar_coeffs = phi
        # kept as the AR(1) name it has been used under everywhere; for a longer AR it
        # is the leading coefficient and NOT the lag-one correlation, so read
        # `ar_order` when what you mean is "is this coloured at all"
        self.ar_rho = phi[0] if len(phi) == 1 else 0.0

        if nu is None:
            self.log_nu = None
        else:
            if nu <= 0:
                raise ValueError(f"nu must be > 0 (got {nu}); Student-t needs positive dof")
            self.log_nu = nn.Parameter(torch.tensor(float(nu)).log())

    # ------------------------------------------------------- constrained params
    @property
    def sigma_y(self) -> torch.Tensor:
        """Residual scale sigma_y = exp(log_sigma_y) > 0; [S, 1] when per sequence."""
        s = self.log_sigma_y.exp()
        return s.reshape(-1, 1) if s.dim() == 1 else s

    @property
    def per_sequence(self) -> bool:
        """True iff each sequence has its own sigma_y."""
        return self.log_sigma_y.dim() == 1

    @property
    def nu(self) -> torch.Tensor | None:
        """Student-t dof nu = exp(log_nu) > 0, or None in Gaussian mode."""
        return None if self.log_nu is None else self.log_nu.exp()

    @property
    def robust(self) -> bool:
        """True iff this is the Student-t (heavy-tailed) emission."""
        return self.log_nu is not None

    def _dist(self, mean: torch.Tensor, scale_mul: float = 1.0):
        """The per-bin emission distribution about `mean` (Gaussian or StudentT).

        `scale_mul` narrows the scale, which the AR(1) conditionals need: given the
        previous residual the innovation has scale sigma_y * sqrt(1 - rho^2), while
        the marginal stays sigma_y.
        """
        scale = self.sigma_y * scale_mul
        if self.log_nu is None:
            return Normal(mean, scale)
        return StudentT(self.nu, mean, scale)

    # ------------------------------------------------------------- likelihood
    def log_prob(self, y: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """Emission block of the complete-data log-likelihood, summed over t.

            sum_t log p(y_t | m_t),   m_t = x_hat(t) + c_t,

        with p the Gaussian or, robustly, the Student-t marginal. `y` and
        `mean` are broadcastable to [B, T]; returns shape [B]
        (the per-sequence emission log-likelihood, matching the shape of
        PhasicEventProcess.event_logprob so the two blocks add directly).
        """
        if not self.ar_coeffs:
            return self._dist(mean).log_prob(y).sum(dim=-1)
        # AR(p): factorise into conditionals on the observed past. No latent is
        # introduced and the cost per step is unchanged -- the conditional mean just
        # carries the previous residuals. See `ar_conditional`.
        cm, sd = ar_conditional(y, mean, self.sigma_y, self.ar_coeffs)
        return self._dist_at(cm, sd).log_prob(y).sum(dim=-1)

    def _dist_at(self, loc, scale):
        """The emission family at an explicit per-bin (loc, scale).

        `_dist` takes a single multiplier, which is enough for AR(1) where every
        conditional past the first shares one scale. An AR(p) does not: the first p
        bins each have their own, so the scale arrives as a tensor.
        """
        if self.log_nu is None:
            return Normal(loc, scale)
        return StudentT(self.nu, loc, scale)

    def residual(self, y: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """Partial residual r_t = y_t - m_t -- the GAM diagnostic."""
        return y - mean

    # ------------------------------------------------------------- generative
    @torch.no_grad()
    def sample(
        self,
        mean: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Draw an observed trace y given the additive mean.

        Gaussian mode: y = m + sigma_y * eps, eps ~ N(0, 1).
        Robust mode: the scale mixture -- draw a per-bin precision
        w_t ~ Gamma(nu/2, nu/2), then y_t ~ N(m_t, sigma_y^2 / w_t); marginalizing
        w reproduces the StudentT. Returns a dict with `y` (shape of
        `mean`) and `w` (the auxiliary precisions, or None in Gaussian mode).

        `generator` seeds the Gaussian innovation eps for reproducibility; the
        Gamma precision draw uses the global RNG (torch.distributions.Gamma takes
        no generator), so seed with torch.manual_seed for a deterministic robust
        draw.
        """
        eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        if self.ar_coeffs:
            # The same recursion the density scores, so a trace drawn here is a trace
            # the log_prob above rates highly. lfilter rather than a Python loop:
            # 60000 steps per sequence is a minute of interpreter time otherwise.
            import numpy as np  # noqa: PLC0415
            from scipy.signal import lfilter  # noqa: PLC0415
            if self.log_nu is not None:
                raise NotImplementedError(
                    "a coloured emission with the Student-t is not defined here: the "
                    "scale mixture and the AR recursion interact and nobody has "
                    "decided how")
            phi = np.asarray(self.ar_coeffs, float)
            g = ar_autocovariance(self.ar_coeffs, len(phi) + 1)
            innov = float(g[0] - np.dot(phi, g[1:len(phi) + 1])) ** 0.5
            sig = self.sigma_y.detach().cpu().numpy()   # scalar, or [S, 1] per sequence
            e = eps.detach().cpu().numpy() * sig
            # burn in past the transient rather than trying to draw the first p bins
            # from their joint: 200 time constants of the slowest pole is far cheaper
            # than getting a p-variate stationary draw right, and is exact to
            # machine precision by the time it reaches bin 0.
            burn = 200 * len(phi) + int(
                10 / max(1e-9, -np.log(max(np.abs(np.roots(
                    np.concatenate([[1.0], -phi]))))))
            )
            burn = min(max(burn, 100), 20000)
            pad = np.random.default_rng(0).standard_normal(e.shape[:-1] + (burn,)) * sig
            full = lfilter([innov], np.concatenate([[1.0], -phi]),
                           np.concatenate([pad, e], axis=-1), axis=-1)
            noise = torch.as_tensor(full[..., burn:], device=mean.device, dtype=mean.dtype)
            return {"y": mean + noise, "w": None}
        if self.log_nu is None:
            return {"y": mean + self.sigma_y * eps, "w": None}
        half_nu = 0.5 * self.nu
        w = Gamma(half_nu, half_nu).sample(mean.shape).to(device=mean.device, dtype=mean.dtype)
        y = mean + (self.sigma_y / w.sqrt()) * eps
        return {"y": y, "w": w}


# MAD-to-sigma constant for a Gaussian: MAD = 0.6745 * sigma (0.6745 = Phi^{-1}(0.75)).
_MAD_TO_SIGMA = 1.0 / 0.6745


def estimate_log_noise_scale(
    y: torch.Tensor,
    inj_t: int | None = None,
    *,
    min_scale: float = 1e-3,
    ar_coeffs=None,
) -> float:
    """Data-driven anchor for sigma_y: the high-frequency noise floor of `y`.

    A coarse proxy for the residual scale sigma_y used to seed the prior mean
    (and guide loc) of SingleTraceModel's optional Bayesian emission scale -- the
    sigma_y sibling of `estimate_log_baseline_rate`. It is the robust scale of
    the **successive differences** of `y` (the DAUB / Gasser-Sroka-Jennen-Steinmetz
    white-noise estimator):

        d_t = y_t - y_{t-1},   sigma_hat = MAD(d) / (0.6745 * sqrt(2)),
        sigma_y ~ log(max(sigma_hat, min_scale)).

    Differencing is a high-pass filter, so the slow tonic structure the single
    -Bateman baseline can't represent is removed before the estimate -- the
    anchor tracks genuine measurement noise, not baseline misfit. The MAD makes
    it robust to the sparse phasic events (and the smooth convolved bumps they
    produce barely move the difference signal anyway), so it is not inflated by
    real signal. The `/sqrt(2)` undoes the variance doubling of differencing two
    independent noise samples; `0.6745` converts the MAD to a Gaussian std.

    That `/sqrt(2)` assumes the noise is WHITE, and silently ruins the estimate if it is
    not: for an AR residual var(y_t - y_{t-1}) is 2 sigma^2 (1 - rho_1), so this returns
    sigma * sqrt(1 - rho_1). On the AR(2) fitted to the recordings rho_1 is 0.998 and the
    factor is 0.045 -- a 22-fold underestimate, which would hand the fit a trace it
    believes is almost noiseless. Pass `ar_coeffs` and the correction is divided out.

    `y` is [B, T] or [T]; differences are taken along time and **pooled** over
    the batch. When `inj_t` is given, only the pre-injection window `t < inj_t`
    is used (the cleanest noise window -- no drug transient), mirroring
    `estimate_log_baseline_rate`. `min_scale` floors an empty/degenerate window
    so the log is finite. Returns a Python float (the log-scale).

    Like the rate anchor this makes the sigma_y prior weakly **empirical-Bayes**
    (its location is read off the data); the prior *scale* stays a fixed
    narrow choice set on SingleTraceModel -- the knob that keeps sigma_y from
    inflating.
    """
    Y = y.unsqueeze(0) if y.dim() == 1 else y  # [B, T]
    if inj_t is not None:
        Y = Y[:, :inj_t]
    if Y.shape[1] < 2:
        return float(math.log(min_scale))
    d = (Y[:, 1:] - Y[:, :-1]).reshape(-1)            # pooled successive differences
    med = d.median()
    mad = (d - med).abs().median()
    sigma_hat = float(mad.item()) * _MAD_TO_SIGMA / math.sqrt(2.0)
    if ar_coeffs:
        # undo the white-noise assumption: var(dy) = 2 sigma^2 (1 - rho_1)
        rho1 = float(ar_autocovariance(list(ar_coeffs), 2)[1])
        sigma_hat = sigma_hat / math.sqrt(max(1.0 - rho1, 1e-12))
    return float(math.log(max(sigma_hat, float(min_scale))))


def estimate_ar_coeffs(y, inj_t: int | None = None, p: int = 2) -> list[float]:
    """Yule-Walker AR(p) coefficients of the pre-injection trace, mean removed.

    This is how the paper's coefficients were obtained from the recordings
    (scripts/paper_calibration_numbers.py), made into a function so a user's fit reads its
    own. The slow tonic and the sparse transients are left in, as they were there: the
    residual the emission models is dominated by the slow correlated component, and that
    is what the coefficients have to describe. Returns [] (white noise) when the window is
    too short to estimate p lags or the estimate is not stationary.
    """
    import numpy as np
    from scipy.linalg import solve_toeplitz
    v = np.asarray(y.detach().cpu() if torch.is_tensor(y) else y, float).reshape(-1)
    if inj_t is not None:
        v = v[:inj_t]
    if v.size < 50 * (p + 1):
        return []
    v = v - v.mean()
    r = np.array([np.dot(v[:v.size - k], v[k:]) / v.size for k in range(p + 1)])
    if r[0] <= 0:
        return []
    phi = [float(x) for x in solve_toeplitz((r[:p], r[:p]), r[1:p + 1])]
    return phi if _ar_is_stationary(phi) else []
