"""Pyro model/guide assembling the points joint for relaxed-discrete SVI.

Wires the three generative building blocks -- the phasic event process
(events.py), the tonic Bateman baseline (tonic.py), and the additive-mean
emission (emission.py) -- into a Pyro `model`/`guide` pair fit by SVI on the
ELBO. Discrete events are handled by a binary-Concrete (Gumbel-Softmax)
relaxation so every latent is reparameterizable and `Trace_ELBO` applies.

Assumptions:

  * One shared global parameter set (a batch of sequences, no drug->animal
    hierarchy); globals are MAP point estimates registered via `pyro.module`,
    as in PhotoHMM.
  * Events relaxed with `RelaxedBernoulli` at a fixed temperature `tau`.
  * Local guide: log-normal marks with free per-(sequence,bin) params, and event
    logits from a pluggable block (`event_guides.py`) -- free per-bin by default,
    or an amortized family.
  * No FIR self-excitation (history_len must be 0), so the prior intensity is
    event-independent and the whole time axis vectorizes.
"""

from __future__ import annotations

import math

import pyro
import pyro.distributions as dist
import torch
from torch import nn

from photosvi.models.points.emission import AdditiveEmission, ar_conditional
from photosvi.models.points.event_guides import FreeEventGuide
from photosvi.models.points.events import PhasicEventProcess
from photosvi.models.points.tonic import TonicBaseline

#: Width the per-bin mark posterior starts at. Deliberately wider than any plausible
#: sigma_a: the loc starts near the noise, so at windows that do hold an event the
#: initial claim is wrong by several units, and it must not also be confident.
_MARK_SCALE_INIT = 0.5
#: Upper bound on the mark posterior's scale, in log-amplitude units.
#:
#: Not a tuning knob: a guard against a failure mode that is silent until it is fatal. The
#: mark KL is gated by the event probability, so in the bins holding no event -- almost all
#: of them -- the gate sits at its 1e-6 floor and nothing constrains q(a)'s WIDTH. Where the
#: prior is too narrow for the data, the scale drifts; on the opioid cohort it sat near 1
#: until step 10000 and then ran 3 -> 17 -> 30, at which point exp(loc + 4 scale) passed the
#: largest float32 and one infinite mark met the gate meant to make it harmless, giving
#: 0 * inf = NaN in the mean. Every parameter was still finite, so a finiteness check on
#: them caught nothing and the fit died 11000 steps in with no diagnosis.
#:
#: Five is far outside anything meaningful -- it is a posterior claiming marks spread over
#: e^20 either way -- so it cannot bind on a fit that is working, and when it does bind the
#: parameter is visibly pinned instead of the run crashing. The cure for a fit that reaches
#: it is a prior that matches the data, not a larger bound.
_MARK_SCALE_MAX = 5.0

#: Smallest weight the gated mark divergence may carry at any bin.
#:
#: Gating the mark KL by q(s_t) is what the generative story implies -- a mark at a bin with
#: no event is never used, so charging its divergence there is charging for nothing -- and
#: it is what removed the guide's incentive to empty the trace. But it also removes the only
#: force holding q(a)'s WIDTH anywhere near the prior, and at a bin with no event the right
#: answer is precisely the prior, not a free parameter. With the gate at its 1e-6 clamp the
#: width is unconstrained, and on the opioid cohort it drifted from 0.7 to 30 over eleven
#: thousand steps until exp(loc + 4 sigma) passed the largest float32 and one infinite mark
#: met the near-zero gate: 0 * inf = NaN.
#:
#: A floor on the gate charges the empty bins a small fraction of their divergence, which
#: pulls them back to the prior -- where they belong -- while leaving the bins that do hold
#: an event paying in full, since their gate is near one either way. Zero restores the
#: previous behaviour exactly.
_MARK_GATE_FLOOR = 0.0



def observed_bins(obs_mask, shape, device, ar_order):
    """The mask actually applied to the emission, widened for an AR(p) residual.

    A masked bin is not enough under a correlated emission. `AdditiveEmission`
    factorises an AR(p) residual as sum_t p(y_t | y_{t-1}, ..., y_{t-p}), so dropping
    bin t leaves the terms at t + 1, ..., t + p still conditioning on the value at t --
    and that value is exactly the one we decided not to trust. Measured on a 50-sigma
    spike with an AR(1) rho = 0.9: masking the bin alone leaves -5331 of the -11908 it
    cost, while masking the bin and its successor gives -1.54 against -1.72 for a
    clean trace.

    So every masked bin takes the `ar_order` following ones with it. This is what the
    injection window needs: the handling artefact is large and broadband, and an
    unmasked one is read as a burst of events at precisely the bins where the drug
    contrast is computed.

    `ar_order` is the number of AR coefficients, `len(emission.ar_coeffs)`. It used to
    be `emission.ar_rho`, which is zero for any order above one, so under AR(2) the
    mask was not widened at all (found 2026-10-07).
    """
    m = torch.as_tensor(obs_mask, device=device, dtype=torch.bool)
    if m.dim() == 1:
        m = m.reshape(-1, 1) if m.shape[0] == shape[0] else m.reshape(1, -1)
    m = m.expand(shape)
    base = m
    for k in range(1, min(int(ar_order), m.shape[-1] - 1) + 1):
        keep = torch.ones_like(base)
        keep[..., k:] = base[..., :-k]
        m = m & keep
    return m


class SingleTraceModel(nn.Module):
    """Points-family Pyro program: relaxed-event SVI over the additive-mean joint.

    Global parameters (marks mu_a/sigma_a, tonic psi, robustness nu) are
    MAP-optimized `nn.Parameter`s registered via `pyro.module("points", self)`.
    The phasic coefficients **beta0 / beta_inj are always Bayesian** -- under a
    Normal prior (default) or a Uniform prior on [beta_low, beta_high]
    (`beta_prior="uniform"`), each with a matching mean-field posterior in
    `guide` (Normal, or a sigmoid-affine-transformed Normal whose support is
    exactly the prior interval). The **emission
    scale sigma_y is optionally Bayesian**: passing `sigma_y_prior_loc` switches
    it from a MAP point estimate to a sampled latent under a narrow Normal prior
    on `log sigma_y`, anchored on the data's high-frequency noise floor so it
    can't inflate to absorb tonic misfit. Latent sample sites are therefore the
    globals `beta0`, `beta_inj` (and `log_sigma_y` when Bayesian) plus the
    per-bin events `s` (relaxed) and marks `a`; the guide covers exactly those.

    Parameters
    ----------
    phasic : PhasicEventProcess supplying the intensity GLM, mark params, fixed
        kernel and `convolve`. Must have `history_len == 0` for this slice.
    tonic : TonicBaseline supplying the deterministic x_hat(t).
    emission : AdditiveEmission supplying sigma_y (and nu for the robust variant).
    tau : Concrete relaxation temperature lambda.
    beta0_prior_loc, beta_inj_prior_loc : prior means for the Bayesian phasic
        coefficients (Normal prior on the unconstrained log-rate axis). Default
        to the phasic block's own `beta0` / `beta_inj` anchors -- pass a value
        from `estimate_log_baseline_rate(y, inj_t)` for a data-driven beta0.
    beta0_prior_scale, beta_inj_prior_scale : prior std-devs (weakly-informative;
        ~1 log-unit covers a wide event-rate range). Fixed, not data-driven.
        Only used when `beta_prior="normal"`.
    beta_prior : "normal" (default) or "uniform". Selects the prior family for
        BOTH beta0 and beta_inj. "uniform" replaces the informative Normal with
        a flat prior on [beta_low, beta_high]; the guide switches to a
        transformed Normal supported on that interval so guide samples can
        never fall outside the prior's support.
    beta_low, beta_high : bounds of the Uniform prior on the log-rate axis.
        Must bracket the plausible beta range -- values outside are assigned
        zero prior mass. Only used when `beta_prior="uniform"`.
    sigma_y_prior_loc : prior mean for the Bayesian emission scale on the
        `log sigma_y` axis. None (default) keeps the legacy MAP point estimate
        (`emission.log_sigma_y`); pass a value -- e.g.
        `estimate_log_noise_scale(y, inj_t)` -- to make sigma_y Bayesian with a
        data-driven anchor. Supplying it freezes `emission.log_sigma_y`.
    sigma_y_prior_scale : prior std-dev on the `log sigma_y` axis (default 0.1).
        *Narrow on purpose* -- this is the anti-inflation knob: a tight prior
        pins sigma_y near the high-frequency anchor so it can't soak up slow
        baseline misfit. Only used when `sigma_y_prior_loc` is given.
    event_guide : the block supplying q(s)'s per-bin logits (`event_guides.py`).
        None (default) installs `FreeEventGuide` -- one free variational logit
        per (sequence, bin). Pass `MatchedFilterEventGuide(mf)` for the amortized
        family, whose event ranking is pinned to the matched filter's. The choice
        changes only the variational family; `model` is untouched either way.
    """

    def __init__(
        self,
        phasic: PhasicEventProcess,
        tonic: TonicBaseline,
        emission: AdditiveEmission,
        *,
        tau: float = 0.5,
        beta0_prior_loc: float | None = None,
        beta0_prior_scale: float = 1.0,
        beta_inj_prior_loc: float | None = None,
        beta_inj_prior_scale: float = 1.0,
        sigma_y_prior_loc: float | None = None,
        sigma_y_prior_scale: float = 0.1,
        beta_prior: str = "normal",
        beta_low: float = -2.0,
        beta_high: float = 2.0,
        beta_inj_low: float | None = None,
        beta_inj_high: float | None = None,
        event_guide: nn.Module | None = None,
        couple_events: bool = False,
        analytic_s_kl: bool = True,
        mark_kl: str = "gated",
        mark_gate_floor: float = _MARK_GATE_FLOOR,
        mark_window: int | None = None,
        # The 2x2 joint Gaussian over (beta0, beta_inj), NOT a mean-field pair.
        #
        # Default since 2026-09-14. The two coefficients trade off by construction --
        # a higher baseline with a smaller injection effect explains the same
        # post-injection rate -- and their posterior correlation is -0.90. A
        # factorised family has to call that zero, and reports 0.46x the exact width
        # against a NUTS reference where the block reports 1.03x.
        #
        # The old default was "meanfield", and it cost the study its whole two-stage
        # arm: `cohort_twostage.py` built this class without naming a guide and 519
        # per-session fits silently took the family we had already rejected. A
        # default that is wrong for every production use is not a neutral choice.
        #
        # It is also free. Measured on a 40-minute trace: 96,017 variational
        # parameters against 96,011, and no time difference outside the noise of a
        # loaded machine. "meanfield" stays reachable because S1 Text compares
        # against it and `linear_response.py` corrects it, not because there is a
        # case for choosing it -- detection reads q(s_t) and never touches these
        # coefficients, so it is identical either way.
        beta_guide: str = "fullrank",
    ):
        super().__init__()
        if phasic.history_len != 0:
            raise NotImplementedError(
                "SingleTraceModel slice requires history_len == 0 (no FIR self-excitation): "
                "relaxed events feeding back into the intensity is a deferred slice."
            )
        if tau <= 0:
            raise ValueError(f"relaxation temperature tau must be > 0 (got {tau})")
        if beta0_prior_scale <= 0 or beta_inj_prior_scale <= 0:
            raise ValueError("beta prior scales must be > 0")
        if sigma_y_prior_scale <= 0:
            raise ValueError("sigma_y_prior_scale must be > 0")
        if beta_prior not in ("normal", "uniform"):
            raise ValueError(f"beta_prior must be 'normal' or 'uniform' (got {beta_prior!r})")
        if beta_guide not in ("meanfield", "fullrank"):
            raise ValueError(
                f"beta_guide must be 'meanfield' or 'fullrank' (got {beta_guide!r})")
        if beta_low >= beta_high:
            raise ValueError(f"beta_low must be < beta_high (got [{beta_low}, {beta_high}])")
        bi_low = beta_low if beta_inj_low is None else beta_inj_low
        bi_high = beta_high if beta_inj_high is None else beta_inj_high
        if bi_low >= bi_high:
            raise ValueError(f"beta_inj_low must be < beta_inj_high (got [{bi_low}, {bi_high}])")
        self.phasic = phasic
        self.tonic = tonic
        self.emission = emission
        self.event_guide = FreeEventGuide() if event_guide is None else event_guide
        # When False (the default, and the shipped behaviour) the guide's event logits
        # are anchored on a *detached* prior intensity: the rate reaches q(s) only as a
        # constant init, so the event factor and the betas are estimated by separate
        # blocks. When True the anchor is rebuilt at the betas *sampled in the guide* and
        # left differentiable, so q(s) tracks the rate being estimated and the two chase
        # each other. See `guide()`.
        self.couple_events = bool(couple_events)
        if mark_kl not in ("sampled", "analytic", "gated"):
            raise ValueError(f"mark_kl must be 'sampled', 'analytic' or 'gated' (got {mark_kl!r})")
        self.analytic_s_kl = bool(analytic_s_kl)
        self.mark_kl = mark_kl
        if not 0.0 <= mark_gate_floor < 1.0:
            raise ValueError(
                f"mark_gate_floor must be in [0, 1), got {mark_gate_floor!r}")
        self.mark_gate_floor = float(mark_gate_floor)
        # Marks are tied within a window when the event guide is windowed: at most
        # one event per window means at most one mark, so a free mark per bin is
        # W-fold redundant. Picked up from the guide unless given explicitly.
        if mark_window is None:
            mark_window = getattr(self.event_guide, "window", None)
        self.mark_window = int(mark_window) if mark_window else None
        self.tau = float(tau)
        self.beta_prior = beta_prior
        # A mean-field q cannot represent the correlation between beta0 and beta_inj,
        # and those two trade off against each other by construction: raising the
        # baseline and lowering the injection effect explain the same post-injection
        # rate. Measured against NUTS on the same data, the mean-field posterior is
        # 2.2x too narrow; a 2x2 full-rank block is the cheapest thing that could
        # account for it.
        self.beta_guide = beta_guide
        self.beta_low = float(beta_low)
        self.beta_high = float(beta_high)
        # beta0 is a log-rate anchored near the matched filter's estimate; beta_inj is
        # a log-ratio centred near zero. One interval for both has to be wide enough
        # for the union, which either clips a plausible beta_inj or makes beta0's
        # prior far wider than intended -- and a value outside the support scores
        # -inf and fails silently. So they get their own bounds.
        self.beta_inj_low = float(bi_low)
        self.beta_inj_high = float(bi_high)

        # Bayesian phasic GLM coefficients: Normal prior, mean-field posterior
        # (theory/points-inference eq 3), or Uniform prior with an
        # interval-supported transformed-Normal posterior. Prior means default
        # to the phasic anchors (under "uniform" they only seed the guide
        # init); scales are fixed weakly-informative on the log-rate axis.
        loc0 = phasic.beta0 if beta0_prior_loc is None else beta0_prior_loc
        loc_inj = phasic.beta_inj if beta_inj_prior_loc is None else beta_inj_prior_loc
        self.beta0_prior_loc = float(loc0)
        self.beta0_prior_scale = float(beta0_prior_scale)
        self.beta_inj_prior_loc = float(loc_inj)
        self.beta_inj_prior_scale = float(beta_inj_prior_scale)

        # Optional Bayesian emission scale: Normal prior on log sigma_y (= LogNormal
        # on sigma_y), mean-field posterior, same form as the betas
        # (theory/points-inference: Theta priors are Gaussian on the unconstrained
        # transform). `sigma_y_prior_loc is None` keeps the legacy MAP point
        # estimate. A *narrow* prior_scale anchored on the data's high-frequency
        # noise floor is what stops sigma_y inflating to absorb tonic misfit.
        self.bayesian_sigma_y = sigma_y_prior_loc is not None
        self.sigma_y_prior_loc = None if sigma_y_prior_loc is None else float(sigma_y_prior_loc)
        self.sigma_y_prior_scale = float(sigma_y_prior_scale)
        if self.bayesian_sigma_y:
            # sigma_y is now a sampled latent, not a MAP param: freeze the emission's
            # point param so pyro.module doesn't optimize a value the likelihood
            # no longer reads (the model passes the sampled sigma_y instead).
            self.emission.log_sigma_y.requires_grad_(False)

    # ------------------------------------------------------------------ helpers
    def _prior_event_prob(
        self,
        T: int,
        inj_t: int,
        *,
        device,
        dtype,
        beta0: torch.Tensor | None = None,
        beta_inj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prior per-bin event probability p_t = 1 - exp(-lambda_t dt), shape [T].

        History-free intensity log lambda_t = beta0 + beta_inj * 1[t>=t_inj], so
        p_t is event-independent and the whole time axis is one vectorized term.
        `beta0` / `beta_inj` override the phasic-block anchors with the values
        *sampled* in `model()`; left None (e.g. the guide's logit init) they fall
        back to the phasic buffers.
        """
        t = torch.arange(T, device=device, dtype=dtype)
        post = (t >= inj_t).to(dtype)
        b0 = self.phasic.beta0 if beta0 is None else beta0
        b_inj = self.phasic.beta_inj if beta_inj is None else beta_inj
        # exp(5) is 148 events per second -- every bin at 25 Hz -- and exp(-20) is
        # one event per fifteen years. Outside that the quantity is not a rate, and
        # the clamp keeps p strictly inside (0, 1), where the relaxed Bernoulli's
        # logits are finite. With the unbounded Normal prior there is no longer a
        # support constraint doing this implicitly.
        log_lam = (b0 + b_inj * post).clamp(-20.0, 5.0)
        return -torch.expm1(-log_lam.exp() * self.phasic.dt)

    def _emission_dist(self, mean: torch.Tensor, sigma_y: torch.Tensor | None = None,
                       obs: torch.Tensor | None = None):
        """The additive-mean emission distribution about `mean` (Gaussian/StudentT).

        `sigma_y`, when given, overrides the emission's point scale with a value
        *sampled* in `model()` (the Bayesian-scale path); left None it falls back
        to the emission's MAP `sigma_y`.

        `obs` is the observed trace, and is what makes `ar_rho` reach INFERENCE rather
        than only generation. The AR(1) likelihood factorises into conditionals,
        p(y_1) prod_t p(y_t | y_{t-1}), and because y is observed each conditional is an
        ordinary Normal whose mean carries the previous residual -- so the per-bin site
        stays per-bin and their product is the exact joint, no new latent and no
        approximation. Without `obs` there is nothing to condition on and this returns
        the white marginal, which is also the right thing for a prior predictive.

        Passing `obs` was missing until 2026-09-30, which made `ar_rho` a
        generation-only setting: `AdditiveEmission.log_prob` implemented the AR(1) and
        the SVI path never called it, so a fit with rho set scored exactly as a white
        fit. Two arms of a noise-misspecification grid came back bit-identical, which is
        how it was found.
        """
        scale = self.emission.sigma_y if sigma_y is None else sigma_y
        if self.emission.ar_coeffs and obs is not None and mean.shape[-1] > 1:
            mean, scale = ar_conditional(obs, mean, scale, self.emission.ar_coeffs)
        if self.emission.robust:
            return dist.StudentT(self.emission.nu, mean, scale)
        return dist.Normal(mean, scale)

    def _beta_guide_init(self, anchor: float, *, inj: bool = False) -> float:
        """Unconstrained-axis guide init for the uniform-prior path.

        The logit of the anchor's relative position inside
        [beta_low, beta_high], so the transformed guide starts at the same
        anchor the Normal path would. Clamped 1% inside the bounds so an
        anchor at (or outside) an endpoint still yields a finite init.
        """
        lo, hi = (self.beta_inj_low, self.beta_inj_high) if inj else (self.beta_low, self.beta_high)
        margin = 0.01 * (hi - lo)
        a = min(max(anchor, lo + margin), hi - margin)
        p = (a - lo) / (hi - lo)
        return math.log(p / (1.0 - p))

    def _beta_guide_dist(self, loc, scale, *, device, dtype, inj: bool = False):
        """Interval-supported mean-field posterior for the uniform-prior path.

        A base Normal(loc, scale) pushed through sigmoid + affine onto
        (beta_low, beta_high) -- the construction autoguides use for
        interval-constrained sites. Guide samples can never leave the Uniform
        prior's support (an unbounded Normal guide would score -inf prior
        log-prob whenever it strays outside), and the transform chain keeps
        the sample reparameterizable for the ELBO gradient.
        """
        lo, hi = (self.beta_inj_low, self.beta_inj_high) if inj else (self.beta_low, self.beta_high)
        low = torch.as_tensor(lo, device=device, dtype=dtype)
        width = torch.as_tensor(hi - lo, device=device, dtype=dtype)
        return dist.TransformedDistribution(
            dist.Normal(loc, scale),
            [dist.transforms.SigmoidTransform(), dist.transforms.AffineTransform(low, width)],
        )

    # -------------------------------------------------------------------- model
    def model(self, y: torch.Tensor, inj_t: int, tilt: tuple[float, float] | None = None,
              obs_mask=None):
        """Joint p(y, s, a | Theta) with relaxed events.

        `y` is [B, T] observed traces sharing one injection bin `inj_t`. Samples
        relaxed events s_t ~ RelaxedBernoulli(tau, p_t) and marks
        a_t ~ LogNormal(mu_a, sigma_a) in a (sequences x time) plate, forms the
        deterministic drive u = s a, phasic c = h * u and additive mean
        m = x_hat + c, then scores y under the emission.
        """
        pyro.module("points", self)
        B, T = y.shape
        device, dtype = y.device, y.dtype

        if self.beta_prior == "uniform":
            beta0 = pyro.sample("beta0", dist.Uniform(
                torch.as_tensor(self.beta_low, device=device, dtype=dtype),
                torch.as_tensor(self.beta_high, device=device, dtype=dtype)))
            beta_inj = pyro.sample("beta_inj", dist.Uniform(
                torch.as_tensor(self.beta_inj_low, device=device, dtype=dtype),
                torch.as_tensor(self.beta_inj_high, device=device, dtype=dtype)))
        else:
            beta0 = pyro.sample(
                "beta0",
                dist.Normal(
                    torch.as_tensor(self.beta0_prior_loc, device=device, dtype=dtype),
                    torch.as_tensor(self.beta0_prior_scale, device=device, dtype=dtype),
                ),
            )
            beta_inj = pyro.sample(
                "beta_inj",
                dist.Normal(
                    torch.as_tensor(self.beta_inj_prior_loc, device=device, dtype=dtype),
                    torch.as_tensor(self.beta_inj_prior_scale, device=device, dtype=dtype),
                ),
            )

        if tilt is not None:
            # Linear-response tilt. Adding t.theta to the log joint and differentiating
            # the fitted mean recovers the covariance: for the exact posterior
            # dE[theta]/dt IS Var(theta), by the cumulant-generating identity. The
            # claim linear response makes is that a variational fit's *response* to
            # the tilt approximates the exact one even when its own covariance does
            # not -- so this is the whole method, expressed as one extra term.
            pyro.factor("lr_tilt", tilt[0] * beta0 + tilt[1] * beta_inj)

        # optional Bayesian emission scale: sample log sigma_y on the unconstrained
        # (Gaussian) axis, exp into the positive scale used by the emission factor.
        sigma_y = None
        if self.bayesian_sigma_y:
            log_sigma_y = pyro.sample(
                "log_sigma_y",
                dist.Normal(
                    torch.as_tensor(self.sigma_y_prior_loc, device=device, dtype=dtype),
                    torch.as_tensor(self.sigma_y_prior_scale, device=device, dtype=dtype),
                ),
            )
            sigma_y = log_sigma_y.exp()

        p = self._prior_event_prob(
            T, inj_t, device=device, dtype=dtype, beta0=beta0, beta_inj=beta_inj
        )  # [T]
        x_hat = self.tonic.baseline(T, inj_t)                             # [T]
        tau = torch.as_tensor(self.tau, device=device, dtype=dtype)
        mu_a, sigma_a = self.phasic.mu_a, self.phasic.sigma_a
        if getattr(self.phasic, "mark_prior", None) is not None:
            pyro.factor("mark_prior", self.phasic.mark_prior_logp())

        seq_plate = pyro.plate("sequences", B, dim=-2)
        time_plate = pyro.plate("time", T, dim=-1)

        with seq_plate, time_plate:
            s_dist = dist.RelaxedBernoulli(temperature=tau, probs=p)
            s = pyro.sample("s", s_dist)
            a_dist = dist.LogNormal(mu_a, sigma_a)
            a = pyro.sample("a", a_dist)

        if self.mark_kl != "sampled":
            pyro.factor("a_sampled_logp_undo", -a_dist.log_prob(a).sum())

        if self.analytic_s_kl:
            # Cancel this site's relaxed log p(s). The guide cancels its own log q(s)
            # and adds the analytic Bernoulli KL in their place, so the events'
            # contribution to the ELBO becomes -KL(Bernoulli(q) || Bernoulli(p(beta)))
            # exactly. See the class docstring for why.
            pyro.factor("s_relaxed_logp_undo", -s_dist.log_prob(s).sum())

        u = s * a                          # relaxed drive [B, T]
        c = self.phasic.convolve(u)        # phasic fluorescence [B, T]
        mean = x_hat + c                   # additive mean [B, T]

        # The structural mean, recorded BEFORE the emission so a diagnostic can read it.
        # Under a coloured emission the mean at the `y` site is the CONDITIONAL mean --
        # it carries the previous residual -- so reading that as "the fitted signal"
        # turns every residual diagnostic into a one-step-ahead prediction error, which
        # for a smooth trace is near zero and says nothing. `fitted_mean` prefers this.
        pyro.deterministic("mean_struct", mean)
        with seq_plate, time_plate:
            em = self._emission_dist(mean, sigma_y=sigma_y, obs=y)
            if obs_mask is not None:
                em = em.mask(observed_bins(obs_mask, y.shape, y.device,
                                           len(self.emission.ar_coeffs)))
            pyro.sample("y", em, obs=y)

    # -------------------------------------------------------------------- guide
    def guide(self, y: torch.Tensor, inj_t: int, tilt: tuple[float, float] | None = None,
              obs_mask=None):
        """Mean-field guide over the per-bin events and marks (free params).

        q(s_t) = RelaxedBernoulli(tau, logits=ell_t) and q(a_t) =
        LogNormal(loc_t, scale_t), with ell, loc, scale free `pyro.param`s of
        shape [B, T]. The event guide is binary-Concrete at the SAME temperature as
        the model prior, so the relaxed ELBO is valid.
        """
        B, T = y.shape
        device, dtype = y.device, y.dtype
        tau = torch.as_tensor(self.tau, device=device, dtype=dtype)

        # global beta posterior. Normal-prior path: mean-field Normal on the
        # log-rate axis, loc initialized at the prior mean. Uniform-prior path:
        # the same free (loc, scale) pair lives on an UNCONSTRAINED base axis
        # and is pushed through sigmoid+affine onto (beta_low, beta_high) so
        # q's support matches the prior's; loc is initialized so the
        # transformed guide starts at the anchor. Either way scale starts
        # small-but-free (positive).
        uniform = self.beta_prior == "uniform"
        b0_init = self._beta_guide_init(self.beta0_prior_loc) if uniform else self.beta0_prior_loc
        bi_init = (
            self._beta_guide_init(self.beta_inj_prior_loc, inj=True)
            if uniform
            else self.beta_inj_prior_loc
        )
        b0_loc = pyro.param(
            "q_beta0_loc",
            torch.as_tensor(b0_init, device=device, dtype=dtype),
        )
        b0_scale = pyro.param(
            "q_beta0_scale",
            torch.tensor(0.1, device=device, dtype=dtype),
            constraint=dist.constraints.positive,
        )
        bi_loc = pyro.param(
            "q_beta_inj_loc",
            torch.as_tensor(bi_init, device=device, dtype=dtype),
        )
        bi_scale = pyro.param(
            "q_beta_inj_scale",
            torch.tensor(0.1, device=device, dtype=dtype),
            constraint=dist.constraints.positive,
        )
        if self.beta_guide == "fullrank":
            # One correlated draw for the pair, unpacked onto the model's two sites.
            # The joint is sampled as an auxiliary site so the ELBO takes its log q
            # (including the transform's Jacobian, which is why the transform is
            # applied to the distribution rather than to the sample); the two Delta
            # sites then contribute nothing of their own and simply carry the values
            # to the model.
            loc = pyro.param(
                "q_beta_loc",
                torch.stack([torch.as_tensor(b0_init, device=device, dtype=dtype),
                             torch.as_tensor(bi_init, device=device, dtype=dtype)]),
            )
            tril = pyro.param(
                "q_beta_tril",
                0.1 * torch.eye(2, device=device, dtype=dtype),
                constraint=dist.constraints.lower_cholesky,
            )
            base = dist.MultivariateNormal(loc, scale_tril=tril)
            if uniform:
                lo = torch.tensor([self.beta_low, self.beta_inj_low],
                                  device=device, dtype=dtype)
                width = torch.tensor(
                    [self.beta_high - self.beta_low,
                     self.beta_inj_high - self.beta_inj_low], device=device, dtype=dtype)
                q_beta = dist.TransformedDistribution(
                    base, [dist.transforms.SigmoidTransform(),
                           dist.transforms.AffineTransform(lo, width)])
            else:
                q_beta = base
            pair = pyro.sample("_beta_joint", q_beta, infer={"is_auxiliary": True})
            beta0 = pyro.sample("beta0", dist.Delta(pair[0]))
            beta_inj = pyro.sample("beta_inj", dist.Delta(pair[1]))
        elif uniform:
            beta0 = pyro.sample(
                "beta0", self._beta_guide_dist(b0_loc, b0_scale, device=device, dtype=dtype)
            )
            beta_inj = pyro.sample(
                "beta_inj",
                self._beta_guide_dist(bi_loc, bi_scale, device=device, dtype=dtype, inj=True),
            )
        else:
            beta0 = pyro.sample("beta0", dist.Normal(b0_loc, b0_scale))
            beta_inj = pyro.sample("beta_inj", dist.Normal(bi_loc, bi_scale))

        # mean-field posterior over log sigma_y, mirroring the beta block (only
        # when the Bayesian-scale path is active so model/guide sites match).
        if self.bayesian_sigma_y:
            sy_loc = pyro.param(
                "q_log_sigma_y_loc",
                torch.as_tensor(self.sigma_y_prior_loc, device=device, dtype=dtype),
            )
            sy_scale = pyro.param(
                "q_log_sigma_y_scale",
                torch.tensor(0.1, device=device, dtype=dtype),
                constraint=dist.constraints.positive,
            )
            pyro.sample("log_sigma_y", dist.Normal(sy_loc, sy_scale))

        # event logits are delegated to the pluggable event-guide block (free
        # per-bin by default, or an amortized family).
        if self.couple_events:
            # coupled: the anchor is rebuilt at the betas SAMPLED just above and left
            # differentiable, so a gradient on the rate reshapes q(s). The block then
            # supplies a residual on top of that anchor, hence the zero init -- passing
            # the prior logit here too would double-count it.
            p_live = self._prior_event_prob(
                T, inj_t, device=device, dtype=dtype, beta0=beta0, beta_inj=beta_inj
            )
            anchor = p_live.clamp(1e-9, 1 - 1e-9).logit()
            zeros = torch.zeros(T, device=device, dtype=dtype)
            s_logits = self.event_guide(y, inj_t, prior_logits=zeros) + anchor
        else:
            # decoupled (default): the prior intensity is passed as the init anchor so
            # the free block starts honest, detached because the param-store init must
            # be a leaf, not a function of Theta.
            p0 = self._prior_event_prob(T, inj_t, device=device, dtype=dtype).detach()
            ell0 = p0.clamp(1e-6, 1 - 1e-6).logit()
            s_logits = self.event_guide(y, inj_t, prior_logits=ell0)
        # Per-window marks when the event factor is windowed, per-bin otherwise.
        # The point of tying them is not economy: a mark at a bin with no event is a
        # latent the data cannot speak to, and 60000 of them held only by a
        # divergence against a *fitted* prior is what dragged that prior away.
        # Measured before this change: mu_a fell 1.767 -> 0.368 and sigma_a rose
        # 0.212 -> 0.877, the latter landing on the empty-bin posterior width.
        W = self.mark_window
        mu_a_init = (float(self.phasic.mu_a.detach())
                     if getattr(self.phasic, "q_a_init", "noise") == "prior" else
                     min(float(self.phasic.mu_a.detach()),
                         float(torch.log(self.emission.sigma_y.detach()))))
        if W and W > 1:
            n_win = -(-T // W)
            # Initialised at the NOISE SCALE, capped by the prior. Neither of the
            # obvious choices works in both regimes, and the reason is the same one:
            # what matters is not how the starting mark compares to the prior but how
            # the SPURIOUS DRIVE it creates compares to the noise. q(s_t) is small but
            # not zero at the ~90% of windows holding no event, so every one of them
            # contributes q.E[a] to the fitted mean from step one.
            #
            #   q_a_loc = 0 puts E[a] at 1.0 whatever the units are: 1.7 sigma_y on
            #   simulated cohorts, but 82 sigma_y on real dF/F recordings, where it
            #   drove the marks to E[a] = 8 and the fit never recovered.
            #
            #   q_a_loc = mu_a puts it at the prior: 0.02 (1.7 sigma_y) on real
            #   recordings, which is right, but 6.0 (57 sigma_y) on simulated ones,
            #   where the phasic component reached 10^6 by step 1000 and the loss rose
            #   six orders of magnitude before failing.
            #
            # Starting at the smaller of the two is what both regimes actually want:
            # every mark begins at the level below which it cannot disturb the fit,
            # and the windows that hold an event are raised from there by the data.
            # Measured on a simulated cohort, marks at event windows climb from the
            # noise scale to 4.6 against a true 5.85 within 1000 steps, while the
            # empty ones fall away -- which is the behaviour a gated divergence is
            # supposed to produce and which an anchor at the prior prevents.
            a_loc = pyro.param(
                "q_a_loc", torch.full((B, n_win), mu_a_init, device=device, dtype=dtype)
            ).repeat_interleave(W, dim=-1)[:, :T]
            # The SCALE stays deliberately diffuse and is NOT initialised at
            # sigma_a. A narrow q(a) is a confident claim, and at step zero the claim
            # is wrong wherever a window does hold an event -- the mark there is the
            # prior's, several units above the noise the loc starts at. Confident and
            # wrong together is the combination that kills the fit: measured on a
            # simulated cohort, loc at the noise scale with sigma_a's width failed
            # within 2000 steps, the same loc with a width of 0.5 survived, and the
            # old loc with either width survived. Neither axis does it alone.
            a_scale = pyro.param(
                "q_a_scale",
                _MARK_SCALE_INIT * torch.ones(B, n_win, device=device, dtype=dtype),
                constraint=dist.constraints.interval(0.0, _MARK_SCALE_MAX),
            ).repeat_interleave(W, dim=-1)[:, :T]
        else:
            a_loc = pyro.param(
                "q_a_loc", torch.full((B, T), mu_a_init, device=device, dtype=dtype)
            )
            a_scale = pyro.param(
                "q_a_scale",
                _MARK_SCALE_INIT * torch.ones(B, T, device=device, dtype=dtype),
                constraint=dist.constraints.positive,
            )

        seq_plate = pyro.plate("sequences", B, dim=-2)
        time_plate = pyro.plate("time", T, dim=-1)
        with seq_plate, time_plate:
            q_dist = dist.RelaxedBernoulli(temperature=tau, logits=s_logits)
            s = pyro.sample("s", q_dist)
            qa_dist = dist.LogNormal(a_loc, a_scale)
            a_draw = pyro.sample("a", qa_dist)

        if self.mark_kl != "sampled":
            # A marked point process has marks only where it has points. The model
            # draws a_t at every bin because u_t = s_t * a_t is convenient to write,
            # but where s_t = 0 the mark does not exist -- it is a latent kept for
            # notation. Charging its full divergence at all 60000 bins estimates the
            # mark distribution from ~59700 bins that contain no transient, and the
            # hyperparameters follow: measured, mu_a collapsed 1.767 -> 0.368 and
            # sigma_a inflated 0.212 -> 0.877, the latter landing exactly on the
            # empty-bin posterior width. "gated" weights the divergence by q(s_t),
            # which is what the generative story implies.
            mk = self._lognormal_kl(
                a_loc, a_scale, self.phasic.mu_a, self.phasic.sigma_a
            )
            if self.mark_kl == "gated":
                mk = torch.sigmoid(s_logits).clamp_min(self.mark_gate_floor) * mk
            pyro.factor("a_sampled_logq_undo", -qa_dist.log_prob(a_draw).sum(),
                        has_rsample=True)
            pyro.factor("a_mark_kl", mk.sum(), has_rsample=True)

        if self.analytic_s_kl:
            # A factor here adds to log q, hence subtracts from the ELBO. Undo this
            # site's relaxed log q(s) (so the sampling stays reparameterized for the
            # reconstruction path but stops scoring beta), then charge the analytic
            # Bernoulli KL in its place.
            p_beta = self._prior_event_prob(
                T, inj_t, device=device, dtype=dtype, beta0=beta0, beta_inj=beta_inj
            )
            # has_rsample: both are differentiable functions of reparameterized draws
            # -- `s` from the relaxed guide, and the betas from their Normal posteriors.
            pyro.factor(
                "s_relaxed_logq_undo", -q_dist.log_prob(s).sum(), has_rsample=True
            )
            pyro.factor(
                "s_analytic_kl",
                self._bernoulli_kl(s_logits, p_beta).sum(),
                has_rsample=True,
            )

    @staticmethod
    def _lognormal_kl(loc_q, scale_q, loc_p, scale_p) -> torch.Tensor:
        """KL(LogNormal(loc_q, scale_q) || LogNormal(loc_p, scale_p)), elementwise.

        A LogNormal is a Normal pushed through exp and KL is invariant under a
        diffeomorphism, so this is the Normal-to-Normal KL of the parameters.
        """
        return (
            torch.log(scale_p / scale_q)
            + (scale_q ** 2 + (loc_q - loc_p) ** 2) / (2 * scale_p ** 2)
            - 0.5
        )

    @staticmethod
    def _bernoulli_kl(q_logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """KL(Bernoulli(sigmoid(q_logits)) || Bernoulli(p)), elementwise and exact.

        Written from logits and `log1p` throughout rather than as
        `q*log(q/p) + (1-q)*log((1-q)/(1-p))`: `q` reaches sigmoid(-6) here and `1 - p`
        sits within a rounding error of 1, which is where the naive form starts
        shedding precision. Matches `torch.distributions.kl_divergence` to 2e-16.
        """
        qq = torch.sigmoid(q_logits)
        log_q1 = torch.nn.functional.logsigmoid(q_logits)
        log_q0 = torch.nn.functional.logsigmoid(-q_logits)
        p = p.clamp(1e-12, 1 - 1e-12)
        return qq * (log_q1 - torch.log(p)) + (1 - qq) * (log_q0 - torch.log1p(-p))

    # --------------------------------------------------------- posterior readout
    @torch.no_grad()
    def event_posterior(self, y: torch.Tensor, inj_t: int | None = None) -> torch.Tensor:
        """Variational event probabilities q(s_t = 1) = sigmoid(ell_t), shape [B, T].

        Asks the event-guide block for its fitted logits (call after training).
        These are the relaxed-Bernoulli means, the model's per-bin belief that an
        event occurred -- the points family's whole output. Identical readout
        whichever event-guide family is installed.

        Under `couple_events` the block's logits are only *part* of q(s): `guide()`
        adds the live prior anchor on top, so the block alone is not the posterior.
        This reconstructs the sum at the posterior-mean betas, which is why `inj_t`
        becomes required there -- the anchor is a function of the injection bin.
        Reading the block alone under coupling silently over-states q by the whole
        prior log-odds (order 7 logits at these rates), which shows up as a wildly
        over-confident detector rather than as an error.
        """
        ell = self.event_guide.fitted_logits(y)
        if not self.couple_events:
            return torch.sigmoid(ell)
        if inj_t is None:
            raise ValueError(
                "event_posterior(y, inj_t) requires inj_t when couple_events=True: the "
                "guide's event logits include a prior anchor that depends on it."
            )
        T = y.shape[-1]
        bp = self.beta_posterior()
        b0 = torch.as_tensor(bp["beta0"][0], device=y.device, dtype=y.dtype)
        bi = torch.as_tensor(bp["beta_inj"][0], device=y.device, dtype=y.dtype)
        p_live = self._prior_event_prob(
            T, inj_t, device=y.device, dtype=y.dtype, beta0=b0, beta_inj=bi
        )
        return torch.sigmoid(ell + p_live.clamp(1e-9, 1 - 1e-9).logit())

    @torch.no_grad()
    def reconstruct(self, y: torch.Tensor, inj_t: int | None = None) -> dict[str, torch.Tensor]:
        """The fitted trace, split into its tonic and phasic parts. Call after training.

        Returns `{"tonic", "phasic", "mean"}`, each [B, T], where `mean = tonic +
        phasic` is the model's posterior-mean fluorescence -- the recording with the
        emission noise removed, which is the only part of `y` the model claims to
        explain. What is left over, `y - mean`, is the residual, and a residual that
        still looks like transients is the diagnostic that says so.

        Both parts are posterior means and the split is exact rather than a
        decomposition chosen afterwards: the tonic is the fitted pharmacokinetic
        baseline, the phasic is the kernel convolved with the expected drive
        `E[u_t] = q(s_t = 1) . E[a_t]`. Expectation passes through the convolution
        because it is linear, so this is the mean of the model's mean and not a
        single posterior draw -- it will look smoother than any trace the generative
        model would produce, which is the point of plotting it against the raw data.

        The marks are read from the variational LogNormal, `E[a] = exp(loc +
        scale^2 / 2)`, and where the event factor is windowed each window's mark
        applies to all its bins -- the same broadcast `guide` performs, so the
        reconstruction is the drive the fit actually used.
        """
        q = self.event_posterior(y, inj_t)                                  # [B, T]
        B, T = y.shape
        W = self.mark_window
        loc = pyro.param("q_a_loc")
        scale = pyro.param("q_a_scale")
        if W and W > 1:
            loc = loc.repeat_interleave(W, dim=-1)[:, :T]
            scale = scale.repeat_interleave(W, dim=-1)[:, :T]
        a_mean = torch.exp(loc + 0.5 * scale**2).to(device=y.device, dtype=y.dtype)
        phasic = self.phasic.convolve(q * a_mean)
        tonic = self.tonic.baseline(T, inj_t).expand(B, T)
        return dict(tonic=tonic, phasic=phasic, mean=tonic + phasic)

    @torch.no_grad()
    def beta_posterior(self) -> dict[str, float]:
        """Fitted mean-field posterior over the phasic GLM coefficients.

        Returns `{"beta0": (loc, scale), "beta_inj": (loc, scale)}` read from the
        Pyro param store (call after training). `beta_inj`'s loc is the inferred
        post-injection log-rate shift -- the drug-effect-on-event-rate estimate
        -- and its scale is the posterior std, the credible width on that effect.

        Under `beta_prior="uniform"` the guide params live on the unconstrained
        base axis, so they are mapped back to the log-rate interval first: loc
        becomes the exact posterior median (monotone transform), scale a
        delta-method std about it. Same return shape either way.
        """
        if self.beta_guide == "fullrank":
            fr_loc = pyro.param("q_beta_loc").detach()
            fr_cov = pyro.param("q_beta_tril").detach()
            fr_cov = fr_cov @ fr_cov.T
            marg = [(float(fr_loc[i]), float(fr_cov[i, i].sqrt())) for i in range(2)]

        out = {}
        for i, name in enumerate(("beta0", "beta_inj")):
            if self.beta_guide == "fullrank":
                # The marginal of a correlated joint, which is what a credible
                # interval on one coefficient means. The correlation itself is the
                # point of the block and is reported by `beta_correlation`.
                loc, scale = marg[i]
            else:
                loc = float(pyro.param(f"q_{name}_loc").detach())
                scale = float(pyro.param(f"q_{name}_scale").detach())
            if self.beta_prior == "uniform":
                # Each coefficient has its own interval, so each must be read back
                # through its own transform: beta0's bounds track the trace's rate
                # anchor, beta_inj's are centred on zero. Using one pair for both
                # silently reports beta_inj on beta0's axis.
                lo, hi = (
                    (self.beta_inj_low, self.beta_inj_high)
                    if name == "beta_inj"
                    else (self.beta_low, self.beta_high)
                )
                width = hi - lo
                p = 1.0 / (1.0 + math.exp(-loc))
                loc = lo + width * p
                scale = scale * width * p * (1.0 - p)
            out[name] = (loc, scale)
        return out

    @torch.no_grad()
    def beta_correlation(self) -> float:
        """Posterior correlation between beta0 and beta_inj under the full-rank guide.

        The quantity a mean-field guide is forced to set to zero. The two trade off
        by construction -- a higher baseline and a smaller injection effect explain
        the same post-injection rate -- so a large negative value here is the
        correlation whose omission narrows the mean-field marginals.
        """
        if self.beta_guide != "fullrank":
            raise RuntimeError("beta_correlation requires beta_guide='fullrank'")
        tril = pyro.param("q_beta_tril").detach()
        cov = tril @ tril.T
        return float(cov[0, 1] / (cov[0, 0].sqrt() * cov[1, 1].sqrt()))

    @torch.no_grad()
    def scale_posterior(self) -> dict[str, float]:
        """Fitted mean-field posterior over the emission scale sigma_y.

        Only meaningful when the Bayesian-scale path is active
        (`sigma_y_prior_loc` was supplied); raises otherwise. Returns
        `{"log_sigma_y_loc", "log_sigma_y_scale", "sigma_y"}` read from the Pyro
        param store (call after training), where `sigma_y = exp(log_sigma_y_loc)`
        is the posterior median residual scale. A value that stays near the
        high-frequency anchor (rather than inflating) is the win this path buys.
        """
        if not self.bayesian_sigma_y:
            raise RuntimeError(
                "scale_posterior() requires the Bayesian sigma_y path "
                "(pass sigma_y_prior_loc to SingleTraceModel); sigma_y is a MAP point "
                "estimate here -- read emission.sigma_y instead."
            )
        loc = float(pyro.param("q_log_sigma_y_loc").detach())
        scale = float(pyro.param("q_log_sigma_y_scale").detach())
        return {
            "log_sigma_y_loc": loc,
            "log_sigma_y_scale": scale,
            "sigma_y": float(torch.tensor(loc).exp()),
        }
