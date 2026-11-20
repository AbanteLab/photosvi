"""Pluggable event-logit blocks for the SingleTraceModel guide.

`SingleTraceModel.guide` owns the global block (the phasic betas, the optional
Bayesian log sigma_y, the marks) and delegates exactly one thing: the per-bin
relaxed-Bernoulli logits ell of q(s). That delegation is this module. Each
block is a small `nn.Module` returning ell of shape [B, T]; swapping blocks
swaps the *variational family* over events without touching the generative
model or the rest of the guide.

Three blocks live here:

  * `FreeEventGuide` -- one free variational logit per (sequence, bin), the
    mean-field default. Maximal flexibility, no coupling to the trace beyond
    the prior-intensity init.
  * `WarmFreeEventGuide` -- the same family, initialized at the matched filter
    instead of the prior. Identical expressiveness, different basin; it exists so
    that "free vs amortized" can be read as a *family* difference rather than an
    initialization one (the amortized block is intrinsically matched-filter-warm).
  * `WindowedCategoricalEventGuide` -- per-bin marginals from a softmax over short
    windows, so timing alternatives inside a window compete instead of accumulating.
    The guide's mass then reads as an event count, which is what the intensity
    coefficients are fit against.
  * `MatchedFilterEventGuide` -- an *amortized* block: ell_t = alpha * m_t + b
    with alpha > 0, where m_t is a precomputed matched-filter score. Two global
    scalars replace the B*T free logits, so the trace itself carries the
    per-bin structure.

The matched-filter blocks take `mf` **precomputed** rather than computing it:
`photosvi.benchmark.detectors` already imports from `photosvi.models.points`,
so calling it here would close an import cycle. Callers score the trace with
`benchmark.matched_filter_score` and hand the result in.
"""

from __future__ import annotations

import pyro
import pyro.distributions as dist
import torch
from torch import nn


class FreeEventGuide(nn.Module):
    """Free per-(sequence, bin) event logits -- the mean-field default.

    `q_s_logits` is a `[B, T]` `pyro.param` initialized at the prior intensity
    so the guide starts honest. Every bin is an independent knob: the family
    can represent any per-bin belief, which is what makes it flexible and also
    what lets it drift away from the trace under a long fit.
    """

    def forward(self, y: torch.Tensor, inj_t: int, *, prior_logits: torch.Tensor) -> torch.Tensor:
        B, T = y.shape
        return pyro.param("q_s_logits", prior_logits.expand(B, T).contiguous())

    def fitted_logits(self, y: torch.Tensor) -> torch.Tensor:
        """The fitted logits, read back from the param store (call after training)."""
        return pyro.param("q_s_logits").detach()


class WarmFreeEventGuide(FreeEventGuide):
    """`FreeEventGuide` seeded from the matched filter instead of the prior intensity.

    Same variational family -- B*T free logits, the same `q_s_logits` param, the
    same readout -- so anything that differs from `FreeEventGuide` is the starting
    basin, not expressiveness. The warm logits are the classical detector init:

        ell_t = clamp(gain * (m_t - thresh), floor, hi)

    with `m_t` a standardized matched-filter score. `thresh` sets which score counts
    as evidence for an event, `gain` how sharply the logit responds either side of
    it, and the clamp keeps the sigmoid off its saturated tails where gradients die.

    Why it exists: `MatchedFilterEventGuide` *is* the matched filter, so comparing it
    against a prior-initialized free guide moves family and initialization together.
    This block holds initialization fixed so the two contrasts separate.

    Parameters
    ----------
    mf : per-bin matched-filter score, shape [T] (or [1, T]). Precomputed by the
        caller, typically `benchmark.matched_filter_score(y, kernel, inj_t)`. Must be
        scored on the same trace and bin grid as the `y` later passed to the guide.
    thresh, gain, hi, floor : the affine-and-clamp above. Defaults assume `mf` is
        standardized, as `matched_filter_score` returns it.
    """

    def __init__(
        self,
        mf: torch.Tensor,
        *,
        thresh: float = 2.5,
        gain: float = 2.0,
        hi: float = 6.0,
        floor: float = -6.0,
    ):
        super().__init__()
        if gain <= 0:
            raise ValueError(f"gain must be > 0 (got {gain})")
        if floor >= hi:
            raise ValueError(f"floor must be < hi (got [{floor}, {hi}])")
        m = torch.as_tensor(mf, dtype=torch.get_default_dtype())
        if m.dim() == 2 and m.shape[0] == 1:
            m = m[0]
        if m.dim() != 1:
            raise ValueError(f"mf must be [T] or [1, T] (got shape {tuple(m.shape)})")
        warm = (gain * (m - thresh)).clamp(floor, hi)
        self.register_buffer("warm_logits", warm.detach().clone())

    def forward(self, y: torch.Tensor, inj_t: int, *, prior_logits: torch.Tensor) -> torch.Tensor:
        B, T = y.shape
        if T > self.warm_logits.numel():
            raise ValueError(
                f"mf covers {self.warm_logits.numel()} bins but the trace has {T}: the "
                "matched filter must be scored on the same trace as the fit."
            )
        warm = self.warm_logits.to(device=y.device, dtype=y.dtype)[:T]
        return pyro.param("q_s_logits", warm.expand(B, T).contiguous())


class MatchedFilterEventGuide(nn.Module):
    """Amortized event logits ell_t = alpha * m_t + b, with alpha > 0.

    `m_t` is a fixed per-bin matched-filter score (the trace correlated with
    the transient kernel); SVI tunes only the gain `alpha` and offset `b`. The
    whole family is two scalars, so instead of learning B*T independent beliefs
    the guide learns *how strongly to trust the matched filter*.

    The positivity constraint on `alpha` is the load-bearing part. `ell` is then
    strictly increasing in `m`, so `q(s_t=1) = sigmoid(ell_t)` ranks bins in
    **exactly** matched-filter order for any (alpha, b). Detection ranking is
    therefore invariant to the fit: it cannot erode as SVI runs, which is what
    the free per-bin family does over a long fit. What the fit *can* still move
    is the calibration -- where the sigmoid puts the decision boundary.

    Parameters
    ----------
    mf : per-bin matched-filter score, shape [T] (or [1, T]). Precomputed by the
        caller, typically `benchmark.matched_filter_score(y, kernel, inj_t)`;
        stored as a buffer and never updated by the fit. Must be scored on the
        same trace and bin grid as the `y` later passed to the guide.
    alpha_init : initial gain (must be > 0). The default assumes `mf` is
        standardized, as `matched_filter_score` returns it.
    bias_init : initial offset, on the logit axis. The default is a sparse
        prior-ish start: sigmoid(-5) ~ 0.007 event probability at m = 0.

    Notes
    -----
    Unlike a conv-net guide, this block has no weight init to draw from the RNG
    -- `alpha` and `b` start at fixed scalars. The fit's basin therefore does
    not depend on RNG call order at construction time.
    """

    def __init__(
        self,
        mf: torch.Tensor,
        *,
        alpha_init: float = 2.0,
        bias_init: float = -5.0,
    ):
        super().__init__()
        if alpha_init <= 0:
            raise ValueError(f"alpha_init must be > 0 (got {alpha_init})")
        m = torch.as_tensor(mf, dtype=torch.get_default_dtype())
        if m.dim() == 2 and m.shape[0] == 1:
            m = m[0]
        if m.dim() != 1:
            raise ValueError(f"mf must be [T] or [1, T] (got shape {tuple(m.shape)})")
        self.register_buffer("mf", m.detach().clone())
        self.alpha_init = float(alpha_init)
        self.bias_init = float(bias_init)

    def _logits(self, T: int, device, dtype) -> torch.Tensor:
        if T > self.mf.numel():
            raise ValueError(
                f"mf covers {self.mf.numel()} bins but the trace has {T}: the matched filter "
                "must be scored on the same trace as the fit."
            )
        alpha = pyro.param(
            "q_s_alpha",
            torch.as_tensor(self.alpha_init, device=device, dtype=dtype),
            constraint=dist.constraints.positive,
        )
        b = pyro.param("q_s_bias", torch.as_tensor(self.bias_init, device=device, dtype=dtype))
        return alpha * self.mf.to(device=device, dtype=dtype)[:T] + b

    def forward(self, y: torch.Tensor, inj_t: int, *, prior_logits: torch.Tensor) -> torch.Tensor:
        B, T = y.shape
        return self._logits(T, y.device, y.dtype).expand(B, T)

    def fitted_logits(self, y: torch.Tensor) -> torch.Tensor:
        """The fitted logits, rebuilt from the fitted alpha/b (call after training)."""
        B, T = y.shape
        return self._logits(T, y.device, y.dtype).detach().expand(B, T)

    @torch.no_grad()
    def gain_posterior(self) -> dict[str, float]:
        """The fitted `{"alpha", "bias"}` -- how much the guide trusts the filter.

        `alpha` is the gain on the standardized matched-filter score and `bias`
        the logit offset. Read after training; both are MAP point estimates, not
        posteriors with a width.
        """
        return {
            "alpha": float(pyro.param("q_s_alpha").detach()),
            "bias": float(pyro.param("q_s_bias").detach()),
        }


class WindowedCategoricalEventGuide(nn.Module):
    """Per-bin marginals from a softmax over short windows -- timing alternatives compete.

    The blocks above all carry the events as an *independent per-bin* gate. One real
    transient produces a kernel-correlation lobe several bins wide, and an independent
    factor answers that lobe with fractional mass at every bin in it, so the guide's
    total mass is `lobe width x per-bin evidence` -- a number with no fixed relation to
    how many transients there were. Measured on the frozen GT under the K5 kernel:
    sum(q) = 787 against 318 true onsets.

    That inflation is not harmless, because the intensity coefficients are fit against
    it. With `p_t = 1 - exp(-lambda_t dt) ~ lambda_t dt`, the ELBO's cross term is a
    Poisson log-likelihood with the `q_t` as observations, so the stationary condition
    is `lambda * T = sum(q)`: the fitted rate *is* the mass density, and

        beta_inj = log(true rate ratio) + log(inflation_post / inflation_pre).

    The second term is pure bias, and it is nonzero exactly when the inflation differs
    either side of the injection -- which it does, because the post-injection lobes
    overlap more. Measured: +1.24 under the old non-negative kernel, -0.16 under K5.

    This block removes the term by construction. Each window of `window` bins gets one
    softmax over `window + 1` outcomes -- "no onset here", or "onset at position k" --
    so the per-bin marginals inside a window **sum to at most 1**:

        pi_w = softmax(z_w),  q_t = pi_w[k(t)],  sum_{t in w} q_t = 1 - pi_w[0] <= 1

    The mass becomes a genuine event count and the timing uncertainty moves into the
    within-window distribution, where it costs entropy and nothing else.

    **This is a reparameterization of the mean-field family, not a categorical joint.**
    `SingleTraceModel.guide` still samples one relaxed Bernoulli per bin; what changed is
    that the marginals are now produced by a competition rather than independently.
    That keeps the model, the sampling, and the ELBO untouched -- so the bound is on
    the same quantity and comparisons against the other arms stay valid -- at the cost
    of the joint still allowing two events in a window. Making the *joint* categorical
    needs a custom distribution and is the follow-up if this works.

    Choosing `window`: wide enough to contain the correlation lobe, so the alternatives
    genuinely compete; narrow enough that two real events rarely share one. Under K5 the
    lobe is 3 bins (0.12 s) and only 1.6% of true inter-event intervals fall below 0.2 s,
    so 5 bins is comfortable. Under the old kernel the lobe was 81 bins and no such
    window existed -- this block only became viable once the kernel was fixed.

    Parameters
    ----------
    mf : per-bin matched-filter score, shape [T] (or [1, T]) when every sequence in the
        batch shares one trace's warm start, or [B, T] to give each sequence its own.
        The per-sequence form is what the hierarchical fit needs: a cohort is B
        *different* recordings, and expanding one animal's matched filter across all of
        them would warm-start every session at another animal's transients. The scores
        go in as `gain * (m_t - thresh)` against a fixed 0 for "no onset", so the
        initial softmax reproduces the classical detector's calls.
    window : bins per window. `T` need not divide evenly; the tail window is shorter.
    """

    def __init__(
        self,
        mf: torch.Tensor,
        *,
        window: int = 10,
        thresh: float = 2.5,
        gain: float = 2.0,
    ):
        super().__init__()
        if window < 2:
            raise ValueError(f"window must be >= 2 bins (got {window})")
        if gain <= 0:
            raise ValueError(f"gain must be > 0 (got {gain})")
        m = torch.as_tensor(mf, dtype=torch.get_default_dtype())
        if m.dim() == 2 and m.shape[0] == 1:
            m = m[0]
        if m.dim() not in (1, 2):
            raise ValueError(f"mf must be [T], [1, T] or [B, T] (got shape {tuple(m.shape)})")
        self.window = int(window)
        # [T] -> broadcast to every sequence; [B, T] -> one warm start per sequence.
        self.per_sequence = m.dim() == 2
        self.register_buffer("warm_scores", (gain * (m - thresh)).detach().clone())

    def _init_scores(self, B: int, T: int, device, dtype) -> torch.Tensor:
        """[B, n_win, window + 1] warm-start scores; slot 0 is 'no onset', fixed at zero.

        Shared across the batch when `mf` came in as [T], one row per sequence when it
        came in as [B, T].
        """
        W, n_win = self.window, -(-T // self.window)          # ceil
        pad = n_win * W - T
        s = self.warm_scores.to(device=device, dtype=dtype)[..., :T]
        if s.dim() == 1:
            s = s.expand(B, -1)
        if s.shape[0] != B:
            raise ValueError(
                f"mf has {s.shape[0]} sequences but the batch has {B}: a per-sequence "
                "warm start must be scored on the same traces as the fit."
            )
        if pad:                       # the tail window is short: pad with -inf-ish
            s = torch.cat([s, torch.full((B, pad), -30.0, device=device, dtype=dtype)], dim=-1)
        return torch.cat([torch.zeros(B, n_win, 1, device=device, dtype=dtype),
                          s.reshape(B, n_win, W)], dim=-1)

    def _marginals(self, B: int, T: int, device, dtype) -> torch.Tensor:
        z = pyro.param(
            "q_s_window_scores",
            lambda: self._init_scores(B, T, device, dtype).contiguous(),
        )
        q = torch.softmax(z, dim=-1)[..., 1:]     # drop the 'no onset' slot
        return q.reshape(B, -1)[:, :T]

    def forward(self, y: torch.Tensor, inj_t: int, *, prior_logits: torch.Tensor) -> torch.Tensor:
        B, T = y.shape
        if T > self.warm_scores.shape[-1]:
            raise ValueError(
                f"mf covers {self.warm_scores.shape[-1]} bins but the trace has {T}: the "
                "matched filter must be scored on the same trace as the fit."
            )
        q = self._marginals(B, T, y.device, y.dtype)
        return q.clamp(1e-6, 1 - 1e-6).logit()

    def fitted_logits(self, y: torch.Tensor) -> torch.Tensor:
        """The fitted logits, rebuilt from the fitted scores (call after training)."""
        B, T = y.shape
        return self._marginals(B, T, y.device, y.dtype).clamp(1e-6, 1 - 1e-6).logit().detach()
