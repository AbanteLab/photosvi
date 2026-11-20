"""Cohort model: a crossed animal x drug hierarchy over the injection effect.

`SingleTraceModel` (points.py) fits **one trace** with one global `beta0` / `beta_inj`. The
scientific question is a population one -- does the drug shift the event rate, and by
how much, across a cohort -- so the estimand should be a population parameter with
partial pooling, not N independent point estimates someone averages afterwards. That
average is not the same quantity: it carries no between-animal variance, so its
interval is wrong in a direction that always favours the claim.

This module is a **separate class**, not a generalization of `SingleTraceModel`. The
single-trace model is frozen (it is what the benchmark and the coverage study
measure) and the changes here -- per-session injection bins, a batched tonic, latent
per-animal coefficients -- would touch every line of its model and guide. The two
share the generative building blocks (events.py, tonic.py, emission.py) and the
event-guide blocks, which is where the modelling actually lives.

Design
------

Each session is one recording of one animal under one drug. Any assignment of drugs
to animals is allowed -- one drug per animal, all drugs in every animal, or anything
between -- and it is given simply as the per-session `animal` and `drug` index
vectors. The model does not need to be told which design it is; it reads it off the
indices and **switches on only the variance components that design can identify**.

    log lambda^(s)_t = beta0[animal(s)] + beta_inj[s] * 1[t >= inj_t[s]]

    beta0[s]    = mu_b0 + sigma_b0 * z_b0[animal(s)]             per animal
                        + sigma_b0_session * z_b0s[s]           per session

    beta_inj[s] = mu_drug[drug(s)]                               the estimand
                + sigma_animal  * z_anim[animal(s)]              animal responsiveness
                + sigma_cell    * z_cell[(animal, drug)(s)]      animal x drug
                + sigma_session * z_sess[s]                      residual session noise

`mu_drug[d1] - mu_drug[d2]` is the drug contrast, and its interval carries the
between-animal component -- the thing a two-stage analysis cannot produce.

Which levels exist
------------------

A variance component that the design cannot identify does not merely fail to be
estimated: it absorbs signal from the one below it. So each is switched on only when
the design supports it, and the rule is a property of the index vectors, not a user
option.

**Which component sets a drug contrast's width depends on the design.** Crossed --
every animal seeing every drug -- the animal effect enters both terms and cancels,
leaving `sigma_cell`. Uncrossed, nothing cancels and `sigma_animal` enters directly.
An earlier version of this paragraph named `sigma_animal` unconditionally; see
`variance_components` and its correction of 2026-09-14.

  * **animal** (`sigma_animal`) -- whenever there is more than one animal.
  * **animal x drug** (`sigma_cell`) -- only when some animal received more than one
    drug. With one drug per animal each animal *is* a single cell, so `z_anim` and
    `z_cell` index the same partition and are perfectly confounded; the model then
    collapses to the nested one, where the drug contrast is a **between-animal**
    comparison and its precision is governed by the number of animals per drug.
    When animals do receive several drugs the comparison becomes **within-animal**:
    `z_anim` cancels from the contrast, and the same number of recordings buys a
    much tighter interval. That is a real property of the design, and the model
    reports it rather than assuming it.
  * **session** (`sigma_session`) -- only when some (animal, drug) cell has more than
    one recording. Without replicates inside a cell there is no residual to estimate.
  * **baseline session** (`sigma_b0_session`) -- whenever an animal has more than one
    recording. This one is on `beta0`, not on the injection effect, and in a
    repeated-measures design it is not optional: sessions are recorded on different
    days, with different bleaching and different fibre placement, so an animal's
    *baseline* event rate genuinely differs between them. Forcing one `beta0` across
    an animal's sessions makes the pre-injection misfit go somewhere, and the only
    place left is `beta_inj` -- which is the estimand.

`design_summary()` returns what was detected, so a fit can state its own design.

Non-centred parameterization
----------------------------

Every level is non-centred: the latent is a standard normal `z` and the coefficient
is `mu + sigma * z`. Under a mean-field guide the centred form has Neal's funnel
geometry -- the posterior over `z` narrows as `sigma` shrinks, and a factorized `q`
that cannot represent that dependence collapses `sigma` toward zero. Non-centring
makes the prior geometry independent of `sigma`, so the mean-field family fits it far
better. This is the standard remedy and it is not optional here: the whole point of
the hierarchy is the variance components.

The tonic is drug-structured
----------------------------

The Bateman carrier is not a nuisance curve: it **is** the drug's pharmacokinetics,
and its rate constants are absorption and elimination, which differ between opioids.
Fitting five free parameters per session pools nothing and therefore cannot report
that difference -- and the flexibility runs in the risky direction, because the
phasic component is identified only by its timescale separation from the tonic, so
an unconstrained per-session tonic is free to absorb phasic structure.

The five unconstrained parameters are therefore split by what they mean, not pooled
uniformly:

  * `alpha`, `rho`, `gamma` (peak height, elimination, absorption) get a **per-drug
    mean** with a per-session deviation. These are the pharmacokinetics, and the
    per-drug means are reported quantities.
  * `b` and `drift` (baseline level, bleaching slope) stay **per session and
    unpooled**. They are recording nuisance -- a different day, a different fibre --
    and pooling them by drug would tie bleaching slopes across animals, which is
    wrong for a reason that has nothing to do with the drug.

The split matters: pooling everything would be as wrong as pooling nothing, in the
opposite direction.
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
from photosvi.models.points.points import (
    _MARK_GATE_FLOOR,
    _MARK_SCALE_INIT,
    _MARK_SCALE_MAX,
    SingleTraceModel,
    observed_bins,
)
from photosvi.models.points.tonic import BatchedTonicBaseline

# The random-effect levels, in the order they nest. Each entry is
# (name, z-site suffix, plate name) and is switched on by `_detect_levels`.
_LEVELS = ("animal", "cell", "session", "b0_session")

# Tonic components that get a per-drug mean plus a per-session deviation. `b` and
# `drift` are deliberately absent: see the module docstring.
_TONIC_COMPONENTS = ("alpha", "rho", "gamma")
# Unit of the level and drift coordinates inside the tonic block, as a fraction of their
# prior scale. The data pin both to far below their prior width (the level to about a
# hundredth of its 0.5 prior), and with one unit the block's Cholesky had to represent
# widths of 1e-2 next to the effects' O(1): the fit ran 1300 steps and diverged. With the
# unit at the posterior's own order the block starts near it (2026-10-20).
_OFFSET_BLOCK_UNIT = 0.01

# Per-session tonic offsets. Not pooled -- they are recording nuisance -- but they can
# be either MAP point estimates or sampled latents; see `bayesian_offsets`.
_TONIC_OFFSETS = ("b", "drift")

# Physical bounds on the variance components, applied before the exponential.
#
# A between-animal spread outside [1e-3, 10] on the log-rate axis is not a variance
# component: 10 means a factor e^10, twenty thousand, between two animals. The bound
# exists because the scales became coordinates of an unbounded joint Gaussian when
# the block was widened, and the Cholesky factor can carry one far during early
# optimisation. Measured 2026-09-10: without it, 2 of the first 7 cohort fits died
# with an infinite intensity, which reaches the emission as a NaN mean.
_SIGMA_RANGE = (1e-3, 10.0)

# The same guard for the TONIC guide's posterior scales, which had none.
#
# These bound `q_*_scale`, the width of the variational posterior over the tonic
# hierarchy -- not a model quantity, so this is a guide constraint and changes no
# density. Added 2026-09-16 after one uncrossed cohort diverged reproducibly across
# two launches with `q_z_tonic_rho_scale` reaching 58.6: a posterior that wide on
# log(k_decay) proposes elimination rates differing by e^58 between sessions, and the
# samples it draws reach the emission as a non-finite mean.
#
# 2.0 is generous rather than tight: on the log axis these parameters inhabit it
# allows a factor of e^2 ~ 7 between sessions, which is more spread than any
# pharmacokinetics has. The phasic components got their bound when the joint block
# made them coordinates of an unbounded Gaussian; these are mean-field and outside
# the block, so the mechanism is different -- weak identifiability of a per-session
# deviation the likelihood barely constrains.
#
# **A bound makes the fit finish; it does not make the parameter identified.** If a
# scale runs to the ceiling the estimate there is the ceiling, not a measurement, and
# `tonic_scales()` is now recorded per run so that is visible rather than inferred.
_TONIC_SCALE_RANGE = (1e-4, 2.0)

# And a bound on the log intensity itself, for the same reason and independently of
# where a large value came from. exp(5) is 148 events per second, so every bin at
# 25 Hz; exp(-20) is one event per 15 years. Outside that interval the quantity is
# not a rate, and clamping keeps p = 1 - exp(-lambda dt) strictly inside (0, 1),
# where the relaxed Bernoulli's logits are finite.
_LOG_LAM_RANGE = (-20.0, 5.0)


class HierarchicalModel(nn.Module):
    """Crossed animal x drug hierarchy over the phasic rate coefficients.

    Parameters
    ----------
    phasic : PhasicEventProcess supplying the kernel, the mark hyperparameters and
        `convolve`. Its own `beta0` / `beta_inj` are unused: the coefficients are
        latent here. Must have `history_len == 0`.
    tonic : BatchedTonicBaseline with one parameter row per session.
    emission : AdditiveEmission supplying sigma_y, shared across the cohort.
    animal : length-S integer index, `animal[s]` = which animal recorded session s.
    drug : length-S integer index, `drug[s]` = which drug session s received. A
        length-A vector is also accepted as shorthand for the one-drug-per-animal
        design. Any assignment is allowed; see "Which levels exist" above for what
        each design can identify.
    inj_t : length-S integer tensor of injection bins, one per session.
    beta0_anchor : prior mean for `mu_b0` on the log-rate axis -- the cohort's
        matched-filter baseline estimate. The prior is deliberately wide.
    n_animals, n_drugs : sizes of the two index sets. Inferred from the indices when
        omitted, which is right only if every level appears in the data; pass them
        explicitly when a level can be empty.
    levels : optional override of the detected random-effect levels, as a subset of
        ("animal", "cell", "session"). Detection is the documented behaviour and an
        override that asks for an unidentifiable level is an error, not a warning --
        the failure mode is silent variance stealing, which no diagnostic catches.
    tau, event_guide, couple_events, analytic_s_kl, mark_kl, mark_window : as in
        `SingleTraceModel`, and with the same defaults -- the fixed model, unchanged. The
        hierarchy is over the coefficients only.
    """

    def __init__(
        self,
        phasic: PhasicEventProcess,
        tonic: BatchedTonicBaseline,
        emission: AdditiveEmission,
        *,
        animal,
        drug,
        inj_t,
        beta0_anchor: float,
        n_animals: int | None = None,
        n_drugs: int | None = None,
        levels: tuple[str, ...] | None = None,
        mu_b0_prior_scale: float = 2.0,
        sigma_b0_prior_loc: float = 0.5,
        sigma_b0_prior_scale: float = 0.5,
        sigma_b0_session_prior_loc: float = 0.25,
        sigma_b0_session_prior_scale: float = 0.5,
        mu_drug_prior_scale: float = 1.0,
        # Seed the block's diagonal for mu_b0 and mu_drug at their prior scales
        # rather than at 0.1. See `_block_init`. Off by default: every cohort
        # result in the paper was fitted from 0.1.
        block_init_at_prior: bool = False,
        sigma_animal_prior_loc: float = 0.5,
        sigma_animal_prior_scale: float = 0.5,
        sigma_cell_prior_loc: float = 0.35,
        sigma_cell_prior_scale: float = 0.5,
        sigma_session_prior_loc: float = 0.25,
        sigma_session_prior_scale: float = 0.5,
        tonic_hierarchy: bool = True,
        tonic_mu_prior_scale: dict | None = None,
        tonic_sigma_prior_loc: dict | None = None,
        tonic_sigma_prior_scale: float = 0.5,
        tonic_block: bool = False,
        bayesian_offsets: bool = False,
        offset_prior_scale: dict | None = None,
        tau: float = 0.5,
        event_guide: nn.Module | None = None,
        couple_events: bool = False,
        analytic_s_kl: bool = True,
        mark_kl: str = "gated",
        mark_gate_floor: float = _MARK_GATE_FLOOR,
        mark_window: int | None = None,
        # The joint block over the globals AND every non-centred effect.
        #
        # Default since 2026-09-14, matching `SingleTraceModel`'s move to "fullrank"
        # and for the same reason: a default that is wrong for every production use
        # is not neutral. Over 24 null crossed cohorts this family reports a contrast
        # interval 0.99x the estimator's realised spread where a block over the
        # globals alone reports 1.57x, and it is the only one that recovers the
        # animal x drug interaction (0.104 against a simulated 0.10). It costs
        # nothing: 37.7 ms/step against 40.2, a 72x72 Cholesky being free beside
        # 2.9 M per-bin latents.
        #
        # `build_cohort_model` reads the config and never falls through to this, so
        # the default only reaches someone constructing the class directly. That is
        # exactly the caller the old default caught out once already.
        #
        # NOT a universal answer: it is calibrated on a CROSSED design, where the
        # animal effect cancels from a drug contrast. It reports sigma_animal at 60%
        # of truth, which is the component an uncrossed design leans on.
        beta_guide: str = "fullrank_z",
    ):
        super().__init__()
        if phasic.history_len != 0:
            raise NotImplementedError(
                "hierarchical slice requires history_len == 0 (no FIR self-excitation)"
            )
        if tau <= 0:
            raise ValueError(f"relaxation temperature tau must be > 0 (got {tau})")
        if mark_kl not in ("sampled", "analytic", "gated"):
            raise ValueError(f"mark_kl must be 'sampled', 'analytic' or 'gated' (got {mark_kl!r})")
        if beta_guide not in ("meanfield", "fullrank", "fullrank_za", "fullrank_z"):
            raise ValueError(
                "beta_guide must be 'meanfield', 'fullrank', 'fullrank_za' or "
                f"'fullrank_z' (got {beta_guide!r})")
        priors = {
            "b0": (sigma_b0_prior_loc, sigma_b0_prior_scale),
            "animal": (sigma_animal_prior_loc, sigma_animal_prior_scale),
            "cell": (sigma_cell_prior_loc, sigma_cell_prior_scale),
            "session": (sigma_session_prior_loc, sigma_session_prior_scale),
            "b0_session": (sigma_b0_session_prior_loc, sigma_b0_session_prior_scale),
        }
        for name, (loc, scale) in priors.items():
            if loc <= 0 or scale <= 0:
                raise ValueError(f"sigma_{name} prior loc and scale must be > 0")
        if mu_b0_prior_scale <= 0 or mu_drug_prior_scale <= 0:
            raise ValueError("mu prior scales must be > 0")

        a, d, inj, A, D = self._parse_indices(animal, drug, inj_t, n_animals, n_drugs)
        if a.numel() != tonic.n_seq:
            raise ValueError(
                f"tonic carries {tonic.n_seq} sessions but the indices name {a.numel()}"
            )
        cell, n_cells = self._cell_index(a, d)
        detected = self._detect_levels(a, d, cell, n_cells, A)
        if levels is None:
            active = detected
        else:
            active = tuple(x for x in _LEVELS if x in set(levels))
            bad = set(levels) - set(_LEVELS)
            if bad:
                raise ValueError(f"unknown level(s) {sorted(bad)}; choose from {_LEVELS}")
            unidentifiable = set(active) - set(detected)
            if unidentifiable:
                raise ValueError(
                    f"level(s) {sorted(unidentifiable)} are not identifiable from this "
                    f"design (detected: {detected}). An unidentifiable variance "
                    "component absorbs the one below it -- usually sigma_animal, which "
                    "is what sets the drug contrast's width."
                )

        self.phasic = phasic
        self.tonic = tonic
        self.emission = emission
        self.event_guide = FreeEventGuide() if event_guide is None else event_guide
        self.register_buffer("animal", a)
        self.register_buffer("drug", d)
        self.register_buffer("cell", cell)
        self.register_buffer("inj_t", inj)
        self.n_sessions = int(a.numel())
        self.n_animals = A
        self.n_drugs = D
        self.n_cells = n_cells
        self.levels = active
        self.detected_levels = detected
        self.level_size = {"animal": A, "cell": n_cells, "session": self.n_sessions,
                           "b0_session": self.n_sessions}
        self.sigma_prior = priors
        self.beta0_anchor = float(beta0_anchor)

        # ---- drug-structured tonic. The per-drug means are anchored on the
        # least-squares pre-fits that built `tonic`, averaged within drug: the same
        # move as anchoring beta0 on the matched filter, and for the same reason --
        # a hard-coded pharmacokinetic constant would eventually sit outside a real
        # drug's range and the fit would start from nowhere.
        self.tonic_hierarchy = bool(tonic_hierarchy)
        # A second full-rank Gaussian block over the per-drug tonic means (alpha, rho,
        # gamma) x drugs, instead of a factorised Normal per coordinate (opt-in,
        # 2026-10-20). Only the means: the per-drug amplitude and rate contrasts are read
        # off them, and the three components of one drug's curve trade against each other
        # in the likelihood (a later, lower peak against an earlier, higher one), which a
        # factorised q calls independent. The tonic scales are sampled on their
        # constrained axis in the model, so blocking them would need a Jacobian term, and
        # the per-session deviations number 3S; both stay factorised.
        # Modes (2026-10-20): "means" (also True) blocks the per-drug means alone;
        # "with_sessions" adds the per-session deviations z^psi_j; and
        # "with_sessions_level_drift" also makes the per-session level b and drift d
        # variational latents (the `bayesian_offsets` sites) inside the same block, instead
        # of MAP point estimates.
        _modes = {False: "off", None: "off", "off": "off", True: "means", "means": "means",
                  "with_sessions": "with_sessions",
                  "with_sessions_level_drift": "with_sessions_level_drift"}
        if tonic_block not in _modes:
            raise ValueError(f"unknown tonic_block {tonic_block!r}")
        self.tonic_block_mode = _modes[tonic_block] if self.tonic_hierarchy else "off"
        self.tonic_block = self.tonic_block_mode != "off"
        if self.tonic_block_mode == "with_sessions_level_drift":
            bayesian_offsets = True
        # Wide by default: exp(1.0) is a 2.7x amplitude factor and exp(0.5) a 1.6x
        # rate factor, so the prior locates the curve without pinning it.
        self.tonic_mu_prior_scale = {"alpha": 1.0, "rho": 0.5, "gamma": 0.5} | (
            tonic_mu_prior_scale or {})
        self.tonic_sigma_prior_loc = {"alpha": 0.3, "rho": 0.2, "gamma": 0.2} | (
            tonic_sigma_prior_loc or {})
        self.tonic_sigma_prior_scale = float(tonic_sigma_prior_scale)
        # A MAP parameter has zero uncertainty by construction, so its error never
        # reaches the estimand. With one `b` and one `drift` per session that is 2S
        # point estimates whose uncertainty is silently discarded -- a source of
        # posterior overconfidence entirely separate from the mean-field
        # factorization, and worth being able to switch on and off to see which is
        # which. Anchored per session on the least-squares pre-fit, unpooled, because
        # bleaching and fibre placement are properties of the recording.
        self.bayesian_offsets = bool(bayesian_offsets)
        if self.bayesian_offsets and not self.tonic_hierarchy:
            raise ValueError(
                "bayesian_offsets requires tonic_hierarchy: the offsets are sampled "
                "inside the same block that builds the drug-structured carrier, and "
                "silently ignoring the request would make the comparison it exists "
                "for meaningless"
            )
        self.offset_prior_scale = {"b": 0.5, "drift": 0.005} | (offset_prior_scale or {})
        self.offset_anchor = {n: getattr(tonic, n).detach().clone()
                              for n in _TONIC_OFFSETS}
        if self.bayesian_offsets:
            for name in _TONIC_OFFSETS:
                getattr(tonic, name).requires_grad_(False)
        self.tonic_anchor = self._tonic_anchors(tonic, drug_of_session=d, n_drugs=D)
        # The per-session deviation needs a drug with more than one session; with one
        # session per drug it is confounded with the per-drug mean itself.
        self.tonic_session_level = bool(
            self.tonic_hierarchy and (torch.bincount(d, minlength=D) > 1).any())
        if self.tonic_hierarchy:
            # alpha/rho/gamma are now latents. Freeze the module's own copies so
            # pyro.module does not optimise parameters the likelihood no longer reads;
            # b and drift stay trainable, since they remain per-session MAP.
            for name in _TONIC_COMPONENTS:
                getattr(tonic, name).requires_grad_(False)
        self.mu_b0_prior_scale = float(mu_b0_prior_scale)
        self.block_init_at_prior = bool(block_init_at_prior)
        self.mu_drug_prior_scale = float(mu_drug_prior_scale)
        self.tau = float(tau)
        # The single-trace analogue of this block closed the whole gap between the
        # variational and the exact posterior: beta0 and beta_inj correlate at -0.88
        # because a higher baseline with a smaller injection effect explains the same
        # post-injection rate, and a mean-field q must call that zero. `mu_b0` and
        # `mu_drug` trade off here for the same reason, so the block covers exactly
        # those. The variance components stay mean-field until there is evidence they
        # need more -- adding them untested would make a null result uninterpretable.
        self.beta_guide = beta_guide
        self.couple_events = bool(couple_events)
        self.analytic_s_kl = bool(analytic_s_kl)
        self.mark_kl = mark_kl
        if not 0.0 <= mark_gate_floor < 1.0:
            raise ValueError(
                f"mark_gate_floor must be in [0, 1), got {mark_gate_floor!r}")
        self.mark_gate_floor = float(mark_gate_floor)
        if mark_window is None:
            mark_window = getattr(self.event_guide, "window", None)
        self.mark_window = int(mark_window) if mark_window else None

    # ---------------------------------------------------------------- design
    @staticmethod
    def _parse_indices(animal, drug, inj_t, n_animals, n_drugs):
        a = torch.as_tensor(animal, dtype=torch.long).reshape(-1)
        d = torch.as_tensor(drug, dtype=torch.long).reshape(-1)
        inj = torch.as_tensor(inj_t, dtype=torch.long).reshape(-1)
        S = a.numel()
        if inj.numel() == 1:
            inj = inj.expand(S).clone()
        if inj.numel() != S:
            raise ValueError(
                f"inj_t must be a scalar or have one entry per session ({S}), "
                f"got {inj.numel()}"
            )
        if S == 0:
            raise ValueError("need at least one session")
        if (a < 0).any() or (d < 0).any():
            raise ValueError("animal and drug indices must be non-negative")
        A = int(a.max()) + 1 if n_animals is None else int(n_animals)
        if int(a.max()) >= A:
            raise ValueError("an animal index exceeds the declared number of animals")
        # A length-A drug vector is the one-drug-per-animal shorthand; expand it.
        if d.numel() == A and d.numel() != S:
            d = d[a]
        if d.numel() != S:
            raise ValueError(
                f"drug must have one entry per session ({S}) -- or per animal ({A}) as "
                f"shorthand for the one-drug-per-animal design -- got {d.numel()}"
            )
        D = int(d.max()) + 1 if n_drugs is None else int(n_drugs)
        if int(d.max()) >= D:
            raise ValueError("a drug index exceeds the declared number of drugs")
        return a, d, inj, A, D

    @staticmethod
    def _cell_index(a: torch.Tensor, d: torch.Tensor):
        """Map each session to its (animal, drug) cell; returns (cell[S], n_cells).

        Cells are indexed over the pairs that actually occur, not the full A x D grid:
        an unobserved combination has no data and would be a latent held only by its
        prior, contributing nothing but a wider posterior everywhere it touches.
        """
        keys = a * (int(d.max()) + 1) + d
        uniq, cell = torch.unique(keys, sorted=True, return_inverse=True)
        return cell, int(uniq.numel())

    @staticmethod
    def _detect_levels(a, d, cell, n_cells, A) -> tuple[str, ...]:
        """Which random-effect levels this design can identify. See the module docstring."""
        out = []
        if A > 1:
            out.append("animal")
        # animal x drug is identifiable only if some animal saw more than one drug --
        # otherwise cells and animals are the same partition and the two are confounded.
        if n_cells > A:
            out.append("cell")
        # a session residual needs replicates inside a cell.
        if bool((torch.bincount(cell, minlength=n_cells) > 1).any()):
            out.append("session")
        # a per-session baseline needs only that some animal was recorded twice --
        # a different day is a different baseline even under a different drug.
        if bool((torch.bincount(a, minlength=A) > 1).any()):
            out.append("b0_session")
        return tuple(out)

    @staticmethod
    def _tonic_anchors(tonic, *, drug_of_session, n_drugs: int) -> dict:
        """Per-drug mean of each pre-fitted tonic component, on the unconstrained axis.

        The pre-fit is a least-squares fit per trace, so averaging within drug gives
        a data-driven starting point for that drug's pharmacokinetics. A drug with no
        sessions falls back to the cohort mean rather than to a constant, so an empty
        level does not start the fit somewhere arbitrary.
        """
        out = {}
        for name in _TONIC_COMPONENTS:
            v = getattr(tonic, name).detach()
            per_drug = torch.full((n_drugs,), float(v.mean()))
            for d in range(n_drugs):
                m = drug_of_session == d
                if bool(m.any()):
                    per_drug[d] = float(v[m].mean())
            out[name] = per_drug
        return out

    def design_summary(self) -> dict:
        """What the model read off the indices -- sizes, active levels, and the shape.

        `within_animal` says whether the drug contrast is a within-animal comparison
        (some animal saw several drugs) or a between-animal one. It is the single fact
        that most changes how much the cohort can resolve, so a fit should report it.
        """
        per_animal_drugs = torch.zeros(self.n_animals, dtype=torch.long)
        for a_i in range(self.n_animals):
            m = self.animal == a_i
            per_animal_drugs[a_i] = int(torch.unique(self.drug[m]).numel()) if m.any() else 0
        return {
            "n_sessions": self.n_sessions,
            "n_animals": self.n_animals,
            "n_drugs": self.n_drugs,
            "n_cells": self.n_cells,
            "levels": self.levels,
            "detected_levels": self.detected_levels,
            "drugs_per_animal": per_animal_drugs.tolist(),
            "within_animal": "cell" in self.detected_levels,
            "tonic_hierarchy": self.tonic_hierarchy,
            "tonic_session_level": self.tonic_session_level,
            "bayesian_offsets": self.bayesian_offsets,
            "beta_guide": self.beta_guide,
        }

    # ------------------------------------------------------------------ helpers
    def _index_of(self, level: str) -> torch.Tensor:
        """Session -> latent index for a level."""
        per_session = torch.arange(self.n_sessions, device=self.animal.device)
        return {"animal": self.animal, "cell": self.cell,
                "session": per_session, "b0_session": per_session}[level]

    def _gate(self, T: int, *, device, dtype) -> torch.Tensor:
        """1[t >= inj_t[s]] as [S, T] -- the per-session injection mask."""
        t = torch.arange(T, device=device, dtype=dtype).unsqueeze(0)
        return (t >= self.inj_t.to(device).unsqueeze(-1).to(dtype)).to(dtype)

    def _prior_event_prob(self, beta0_seq, beta_inj_seq, gate) -> torch.Tensor:
        """p = 1 - exp(-lambda dt), [S, T], from per-session coefficients."""
        log_lam = (beta0_seq.unsqueeze(-1)
                   + beta_inj_seq.unsqueeze(-1) * gate).clamp(*_LOG_LAM_RANGE)
        return -torch.expm1(-log_lam.exp() * self.phasic.dt)

    def _session_coeffs(self, mu_b0, sigma_b0, z_b0, mu_drug, sigmas, zs):
        """Map the non-centred latents onto per-session (beta0, beta_inj).

        `sigmas` and `zs` are dicts keyed by level name. Each active level adds its
        own `sigma * z[index_of_level]` to every session it covers, so sessions that
        share an animal share that animal's draw -- which is what stops them counting
        as independent evidence about the drug.
        """
        beta0_seq = (mu_b0 + sigma_b0 * z_b0)[self.animal]        # [S]
        beta_inj_seq = mu_drug[self.drug]                          # [S]
        for lvl in self.levels:
            term = sigmas[lvl] * zs[lvl][self._index_of(lvl)]
            # b0_session is the only level acting on the baseline rather than on the
            # injection effect: it is the day-to-day drift of an animal's own
            # baseline, not part of its drug response.
            if lvl == "b0_session":
                beta0_seq = beta0_seq + term
            else:
                beta_inj_seq = beta_inj_seq + term
        return beta0_seq, beta_inj_seq

    def _tonic_sites(self, T: int, *, guide: bool, device, dtype, drug_plate):
        """Sample the drug-structured tonic and return x_hat, [S, T].

        Called from `model` and `guide` with identical site names and plate
        structure; only the distributions differ. The drug plate is passed in rather
        than created here so the tonic means share the one `mu_drug` uses -- they are
        per-drug in the same sense.

        With `tonic_hierarchy` off this samples nothing and falls back to the block's
        own MAP parameters, which is the single-trace behaviour.
        """
        if not self.tonic_hierarchy:
            return self.tonic.baseline(T, self.inj_t)
        as_t = lambda v: torch.as_tensor(v, device=device, dtype=dtype)  # noqa: E731
        sess_plate = (pyro.plate("session_tonic", self.n_sessions, dim=-1)
                      if self.tonic_session_level else None)
        psi = {}
        tjoint = None
        tw = self._tonic_block_layout()
        if guide and self.tonic_block:
            init_loc, init_diag = self._tonic_block_init(device, dtype)
            t_loc = pyro.param("q_tonic_block_loc", init_loc)
            t_tril = pyro.param("q_tonic_block_tril", torch.diag(init_diag),
                                constraint=dist.constraints.lower_cholesky)
            tjoint = pyro.sample("_tonic_joint",
                                 dist.MultivariateNormal(t_loc, scale_tril=t_tril),
                                 infer={"is_auxiliary": True})
        for k_, name in enumerate(_TONIC_COMPONENTS):
            anchor = self.tonic_anchor[name].to(device=device, dtype=dtype)
            if tjoint is not None:
                mu_d = dist.Delta(tjoint[tw[f"mu_{name}"]])
            elif guide:
                # The posterior mean starts at the same per-drug anchor the prior is
                # centred on, so the fit begins at the pre-fit rather than at zero --
                # which for `rho` would be a 1 s^-1 elimination rate, three orders of
                # magnitude off, and the Bateman curve would start flat.
                loc = pyro.param(f"q_tonic_mu_{name}_loc", anchor.clone())
                scale = pyro.param(
                    f"q_tonic_mu_{name}_scale",
                    torch.full((self.n_drugs,), 0.1, device=device, dtype=dtype),
                    constraint=dist.constraints.positive)
                mu_d = dist.Normal(loc, scale)
            else:
                mu_d = dist.Normal(anchor, as_t(self.tonic_mu_prior_scale[name]))
            with drug_plate:
                mu = pyro.sample(f"tonic_mu_{name}", mu_d)
            val = mu[self.drug]

            if self.tonic_session_level:
                sname, zname = f"tonic_sigma_{name}", f"z_tonic_{name}"
                med = self.tonic_sigma_prior_loc[name]
                if guide:
                    sl = pyro.param(f"q_{sname}_loc",
                                    torch.tensor(math.log(med), device=device, dtype=dtype))
                    _bound = dist.constraints.interval(*_TONIC_SCALE_RANGE)
                    ss = pyro.param(f"q_{sname}_scale",
                                    torch.tensor(0.1, device=device, dtype=dtype),
                                    constraint=_bound)
                    if tjoint is not None and f"z_{name}" in tw:
                        zl = zs = None
                    else:
                        zl = pyro.param(f"q_{zname}_loc",
                                        torch.zeros(self.n_sessions, device=device,
                                                    dtype=dtype))
                        zs = pyro.param(f"q_{zname}_scale",
                                        torch.full((self.n_sessions,), 0.5,
                                                   device=device, dtype=dtype),
                                        constraint=_bound)
                    sigma_d = dist.LogNormal(sl, ss)
                    z_d = None if zl is None else dist.Normal(zl, zs)
                    if tjoint is not None and f"z_{name}" in tw:
                        z_d = dist.Delta(tjoint[tw[f"z_{name}"]])
                else:
                    sigma_d = dist.LogNormal(as_t(med).log(),
                                             as_t(self.tonic_sigma_prior_scale))
                    z_d = dist.Normal(as_t(0.0), as_t(1.0))
                sigma = pyro.sample(sname, sigma_d)
                with sess_plate:
                    z = pyro.sample(zname, z_d)
                val = val + sigma * z
            psi[name] = val

        # b and drift are per-session and unpooled either way; `bayesian_offsets`
        # decides whether their uncertainty is carried or discarded.
        off = {}
        # One plate object, entered by both offsets: re-creating it per component
        # makes two plates with the same name, which Pyro rejects.
        off_plate = (pyro.plate("session_offset", self.n_sessions, dim=-1)
                     if self.bayesian_offsets else None)
        for name in _TONIC_OFFSETS:
            if not self.bayesian_offsets:
                off[name] = getattr(self.tonic, name)
                continue
            anchor = self.offset_anchor[name].to(device=device, dtype=dtype)
            sc = self.offset_prior_scale[name]
            if tjoint is not None and name in tw:
                # In the block on a standardized axis, offset = anchor + scale * u: the
                # drift's natural scale is 1e-3 and the block shares one optimizer step
                # size across coordinates, so a raw drift coordinate (and its covariance
                # entries) moved ten times its own width per step and the fit diverged by
                # step 8 (2026-10-20). The Jacobian of this map is a constant.
                d_ = dist.Delta(anchor + as_t(sc * _OFFSET_BLOCK_UNIT) * tjoint[tw[name]])
            elif guide:
                loc = pyro.param(f"q_tonic_{name}_loc", anchor.clone())
                scale = pyro.param(
                    f"q_tonic_{name}_scale",
                    torch.full((self.n_sessions,), sc / 5, device=device, dtype=dtype),
                    constraint=dist.constraints.positive)
                d_ = dist.Normal(loc, scale)
            else:
                d_ = dist.Normal(anchor, as_t(sc))
            with off_plate:
                off[name] = pyro.sample(f"tonic_{name}", d_)

        return self.tonic.baseline_from(
            off["b"], psi["alpha"], psi["rho"], psi["gamma"], off["drift"],
            T, self.inj_t)

    @torch.no_grad()
    def session_tonic(self) -> dict[str, list[float]]:
        """The tonic actually used per session, in constrained units, [S] each.

        Distinct from `tonic_posterior`, which reports the per-drug *means*. Under a
        non-centred parameterization the mean and the per-session deviations are only
        jointly identified, so a drifting mean with compensating deviations looks
        wrong at the population level while every session is still fitted well. The
        two readouts have to be compared to tell those cases apart.
        """
        if not self.tonic_hierarchy:
            pk, kd, kr = self.tonic.peak, self.tonic.k_decay, self.tonic.k_rise
        else:
            vals = {}
            for name in _TONIC_COMPONENTS:
                v = self._tonic_mu_moments(name)[0][self.drug]
                if self.tonic_session_level:
                    v = v + (pyro.param(f"q_tonic_sigma_{name}_loc").detach().exp()
                             * self._tonic_z_loc(name))
                vals[name] = v
            pk = vals["alpha"].exp()
            kd = vals["rho"].exp()
            kr = kd + vals["gamma"].exp()
        t_peak = (kr / kd).log() / (kr - kd)
        # "peak" is the HEIGHT of the bump in the trace's units, "peak_s" the TIME it
        # occurs at. Both are reported because the pair is what distinguishes the
        # drugs: how much tone the compound puts in, and how fast it gets there.
        return {"peak": [float(x) for x in pk], "k_decay": [float(x) for x in kd],
                "k_rise": [float(x) for x in kr], "peak_s": [float(x) for x in t_peak]}

    @torch.no_grad()
    def tonic_scales(self) -> dict[str, float]:
        """Posterior median of each tonic variance component."""
        if not (self.tonic_hierarchy and self.tonic_session_level):
            return {}
        return {n: float(pyro.param(f"q_tonic_sigma_{n}_loc").detach().exp())
                for n in _TONIC_COMPONENTS}

    @torch.no_grad()
    def tonic_guide_widths(self) -> dict[str, float]:
        """How wide the tonic guide's own posteriors got, and how close to the bound.

        The bound of `_TONIC_SCALE_RANGE` stops a fit from diverging; it does not make
        the parameter identified. A scale sitting on the ceiling is the ceiling being
        reported, not something measured, and that has to be visible in the run record
        rather than reconstructed later -- which it could not be, because nothing was
        stored and the diverging cohort had to be re-run under instrumentation to find
        out what ran away.
        """
        if not (self.tonic_hierarchy and self.tonic_session_level):
            return {}
        out: dict[str, float] = {}
        for n in _TONIC_COMPONENTS:
            for kind in ("tonic_sigma", "z_tonic"):
                if kind == "z_tonic" and f"z_{n}" in self._tonic_block_layout():
                    continue                       # inside the tonic block, no own scale
                v = pyro.param(f"q_{kind}_{n}_scale").detach()
                out[f"{kind}_{n}_scale_max"] = float(v.max())
        out["at_bound"] = float(
            max(v for k, v in out.items() if k.endswith("_max"))
            >= 0.99 * _TONIC_SCALE_RANGE[1])
        return out

    @torch.no_grad()
    def tonic_posterior(self) -> dict[str, dict[str, float]]:
        """Per-drug pharmacokinetics read off the fit, in constrained units.

        Returns amplitude, elimination and absorption rates and the Bateman peak time
        for each drug -- the quantities a per-session tonic fit cannot produce.
        """
        if not self.tonic_hierarchy:
            return {}
        out = {}
        mom = {n: self._tonic_mu_moments(n)[0] for n in _TONIC_COMPONENTS}
        for d in range(self.n_drugs):
            a = float(mom["alpha"][d])
            r = float(mom["rho"][d])
            g = float(mom["gamma"][d])
            kd, kr = math.exp(r), math.exp(r) + math.exp(g)
            out[d] = {"peak": math.exp(a), "k_decay": kd, "k_rise": kr,
                      "peak_s": math.log(kr / kd) / (kr - kd)}
        return out

    def _emission_dist(self, mean: torch.Tensor, obs: torch.Tensor | None = None):
        """Per-bin emission about `mean`; `obs` is what lets `ar_rho` reach inference.

        See `SingleTraceModel._emission_dist` for why conditioning on the observed
        previous residual is exact rather than an approximation, and for the bug that
        having no `obs` here caused.
        """
        scale = self.emission.sigma_y
        if self.emission.ar_coeffs and obs is not None and mean.shape[-1] > 1:
            mean, scale = ar_conditional(obs, mean, scale, self.emission.ar_coeffs)
        if self.emission.robust:
            return dist.StudentT(self.emission.nu, mean, scale)
        return dist.Normal(mean, scale)

    # Which scales the joint block covers, in order, after (mu_b0, mu_drug...).
    def _block_scales(self) -> list[str]:
        return ["b0", *self.levels] if self.beta_guide.startswith("fullrank") else []

    def _blocked_z(self) -> list[str]:
        """Which non-centred effects the joint block covers, in order.

        Declared once, because which latents are correlated is the entire difference
        between the three block modes and it must not be inferred from a string
        comparison in five places.

        `fullrank_z` takes all of them and overshoots: it widens the cohort contrast
        past nominal while `sigma_animal` falls from 0.265 to 0.158 against a truth
        of 0.25, in every cohort measured. The wider interval is variance displaced
        out of the between-animal level, not variance added.

        `fullrank_za` takes only the animal-level effects. The suspicion it tests is
        that the displacement comes from the levels with the least data per
        coefficient -- 24 cell effects and 24 session effects, against 8 animals --
        whose block coordinates have little to constrain them and can absorb spread
        that belongs elsewhere.
        """
        if self.beta_guide == "fullrank_z":
            return ["z_b0", *(f"z_{lvl}" for lvl in self.levels)]
        if self.beta_guide == "fullrank_za":
            return ["z_b0", *(f"z_{lvl}" for lvl in self.levels if lvl == "animal")]
        return []

    def _block_layout(self) -> tuple[int, dict[str, slice]]:
        """Size of the joint block and where each site sits in it.

        Built once and read everywhere, because a block whose coordinates are indexed
        by hand in six places is a block that will silently unpack the wrong latent
        the first time a level is added.

            [ mu_b0 | mu_drug ... | log sigma ... | z_b0 | z_<level> ... ]

        Under `fullrank` it stops after the scales -- eight coordinates, the globals.
        Under `fullrank_z` it continues through the non-centred effects, which is the
        channel by which between-animal spread reaches `mu_drug`: they are a priori
        independent and a posteriori coupled, and a block that omits them must call
        that dependence zero.
        """
        at, where = 0, {}
        where["mu_b0"] = slice(at, at + 1); at += 1
        where["mu_drug"] = slice(at, at + self.n_drugs); at += self.n_drugs
        for lvl in self._block_scales():
            where[f"sigma_{lvl}"] = slice(at, at + 1); at += 1
        for key in self._blocked_z():
            n = self.n_animals if key == "z_b0" else self.level_size[key[2:]]
            where[key] = slice(at, at + n); at += n
        return at, where

    def _block_init(self, device, dtype):
        """Mean and per-coordinate scale to start the block at.

        The diagonal is seeded with the same values the factorised guide would use --
        0.1 on the globals, 0.5 on the non-centred effects -- rather than one number
        everywhere. Starting 64 unit-scale latents at 0.1 is a much tighter posterior
        than the fit begins from otherwise, and it takes thousands of steps to undo.

        `block_init_at_prior` seeds `mu_b0` and `mu_drug` at their own prior scales
        instead. The argument for it is the sentence above, applied to the coordinates it
        was not applied to: `mu_drug`'s prior scale is 1.0 and it starts at 0.1, so the fit
        has to grow that scale tenfold, and measured on the opioid cohort it was still
        growing after seventeen thousand steps -- the point estimate settled by seven
        thousand while the interval was still inflating. Starting a posterior at its prior
        and letting the data tighten it is the ordinary choice; starting it ten times
        tighter is the one that needs a reason.

        It is off by default because every cohort result in the paper was fitted from 0.1,
        and whether the difference is only speed or also the converged answer is still
        open.
        """
        n, where = self._block_layout()
        loc = torch.zeros(n, device=device, dtype=dtype)
        diag = torch.full((n,), 0.1, device=device, dtype=dtype)
        loc[where["mu_b0"]] = self.beta0_anchor
        for lvl in self._block_scales():
            loc[where[f"sigma_{lvl}"]] = math.log(self.sigma_prior[lvl][0])
        for key in self._blocked_z():
            diag[where[key]] = 0.5
        if self.block_init_at_prior:
            diag[where["mu_b0"]] = self.mu_b0_prior_scale
            diag[where["mu_drug"]] = self.mu_drug_prior_scale
        return loc, diag

    def _sample_log_scale(self, level: str, *, guide: bool, device, dtype,
                          joint=None, index=None):
        """Sample `sigma_<level>` through its log, which is where the prior lives.

        LogNormal(log m, s) on a positive scale is Normal(log m, s) on its log, so
        this is the same prior written on the axis the latent actually inhabits. It
        matters because a joint Gaussian block can then include the scales without a
        transform: a block mixing constrained and unconstrained coordinates needs a
        per-coordinate Jacobian, which is exactly the kind of term that is easy to
        omit and produces a silently wrong objective.
        """
        loc, scale = self.sigma_prior[level]
        name = f"log_sigma_{level}"
        lo, hi = (math.log(_SIGMA_RANGE[0]), math.log(_SIGMA_RANGE[1]))
        if joint is not None:
            return pyro.sample(name, dist.Delta(joint[index])).clamp(lo, hi).exp()
        if guide:
            ll = pyro.param(f"q_{name}_loc",
                            torch.tensor(math.log(loc), device=device, dtype=dtype))
            ls = pyro.param(f"q_{name}_scale",
                            torch.tensor(0.1, device=device, dtype=dtype),
                            constraint=dist.constraints.positive)
            return pyro.sample(name, dist.Normal(ll, ls)).clamp(lo, hi).exp()
        return pyro.sample(name, dist.Normal(
            torch.as_tensor(math.log(loc), device=device, dtype=dtype),
            torch.as_tensor(scale, device=device, dtype=dtype))).clamp(lo, hi).exp()

    def _check_inj(self, inj_t) -> None:
        if inj_t is None:
            return
        v = torch.as_tensor(inj_t, dtype=torch.long).reshape(-1)
        if v.numel() == 1:
            v = v.expand(self.n_sessions)
        if not torch.equal(v, self.inj_t.cpu()):
            raise ValueError(
                "inj_t was set at construction (one bin per session); the value passed "
                "to model/guide disagrees with it. Rebuild the model instead."
            )

    # -------------------------------------------------------------------- model
    def _apply_tilt(self, tilt, sites: dict[str, torch.Tensor]) -> None:
        """Add `sum_k tilt[k] . site[k]` to the log joint, for linear response.

        Adding `t . theta` to the log joint and differentiating the fitted mean
        recovers the covariance: for the exact posterior `dE[theta]/dt` IS
        `Cov(theta)`, by the cumulant-generating identity. Linear response claims a
        variational fit's *response* to that tilt approximates the exact one even when
        its own covariance does not -- which is the situation here, where the block
        gets the cohort contrast's mean right and its width wrong.

        Only the cohort-level sites are tiltable. The per-animal and per-session
        effects are non-centred, so a tilt on them would be a tilt on `z` rather than
        on anything the study reports, and the variance components live on a log axis
        where a linear tilt is not the cumulant-generating one.
        """
        unknown = set(tilt) - set(sites)
        if unknown:
            raise ValueError(
                f"cannot tilt {sorted(unknown)}; tiltable sites are {sorted(sites)}")
        total = None
        for name, coeff in tilt.items():
            site = sites[name]
            c = torch.as_tensor(coeff, device=site.device, dtype=site.dtype)
            if c.shape != site.shape and c.numel() != 1:
                raise ValueError(
                    f"tilt for {name!r} has shape {tuple(c.shape)}, expected "
                    f"{tuple(site.shape)} or a scalar")
            term = (c * site).sum()
            total = term if total is None else total + term
        pyro.factor("lr_tilt", total)

    def model(self, y: torch.Tensor, inj_t=None, tilt=None, obs_mask=None):
        """Joint p(y, z, s, a | Theta) for the cohort. `y` is [S, T].

        `inj_t` is accepted so the signature matches `SingleTraceModel.model`; the injection
        bins are per-session state set at construction, and a value that disagrees
        with them is an error rather than a silent override.

        `tilt` adds a linear term to the log joint for the linear-response correction,
        as a mapping from global site name to coefficient: `{"mu_drug": [0, -1, 1]}`
        tilts along the drug-2-minus-drug-1 contrast, which is the estimand this model
        reports. See `_apply_tilt`.

        `obs_mask` selects which bins contribute to the likelihood. `[S]` or `[S, 1]`
        drops whole sessions, which is what a delete-one-cluster resampling estimate
        needs: the design, and therefore the identifiable variance components, is left
        exactly as it was. `[S, T]` or `[T]` drops individual bins, which is what the
        injection handling window needs -- see `observed_bins`, which also widens the
        mask by one bin under an AR(1) emission.

        Either way it masks the OBSERVATION only, not the latents: a global
        `poutine.mask` would broadcast onto the scalar cohort parameters and break
        their log-probability shapes.
        """
        self._check_inj(inj_t)
        pyro.module("hpoints", self)
        S, T = y.shape
        if S != self.n_sessions:
            raise ValueError(f"y has {S} sequences but the model was built for {self.n_sessions}")
        device, dtype = y.device, y.dtype
        as_t = lambda v: torch.as_tensor(v, device=device, dtype=dtype)  # noqa: E731
        std_normal = dist.Normal(as_t(0.0), as_t(1.0))

        # ---- cohort level
        mu_b0 = pyro.sample("mu_b0", dist.Normal(
            as_t(self.beta0_anchor), as_t(self.mu_b0_prior_scale)))
        sigma_b0 = self._sample_log_scale("b0", guide=False, device=device, dtype=dtype)
        drug_plate = pyro.plate("drug", self.n_drugs, dim=-1)
        with drug_plate:
            mu_drug = pyro.sample("mu_drug", dist.Normal(
                as_t(0.0), as_t(self.mu_drug_prior_scale)))

        if tilt is not None:
            self._apply_tilt(tilt, {"mu_b0": mu_b0, "mu_drug": mu_drug})

        # ---- non-centred random effects, one block per active level
        with pyro.plate("animal", self.n_animals, dim=-1):
            z_b0 = pyro.sample("z_b0", std_normal)
        sigmas, zs = {}, {}
        for lvl in self.levels:
            sigmas[lvl] = self._sample_log_scale(lvl, guide=False, device=device,
                                                 dtype=dtype)
            with pyro.plate(f"{lvl}_effect", self.level_size[lvl], dim=-1):
                zs[lvl] = pyro.sample(f"z_{lvl}", std_normal)

        b0_seq, bi_seq = self._session_coeffs(mu_b0, sigma_b0, z_b0, mu_drug, sigmas, zs)

        gate = self._gate(T, device=device, dtype=dtype)
        p = self._prior_event_prob(b0_seq, bi_seq, gate)          # [S, T]
        x_hat = self._tonic_sites(T, guide=False, device=device, dtype=dtype,
                                  drug_plate=drug_plate)
        tau = as_t(self.tau)
        mu_a, sigma_a = self.phasic.mu_a, self.phasic.sigma_a
        if getattr(self.phasic, "mark_prior", None) is not None:
            pyro.factor("mark_prior", self.phasic.mark_prior_logp())

        seq_plate = pyro.plate("sequences", S, dim=-2)
        time_plate = pyro.plate("time", T, dim=-1)
        with seq_plate, time_plate:
            s_dist = dist.RelaxedBernoulli(temperature=tau, probs=p)
            s = pyro.sample("s", s_dist)
            a_dist = dist.LogNormal(mu_a, sigma_a)
            a = pyro.sample("a", a_dist)

        if self.mark_kl != "sampled":
            pyro.factor("a_sampled_logp_undo", -a_dist.log_prob(a).sum())
        if self.analytic_s_kl:
            pyro.factor("s_relaxed_logp_undo", -s_dist.log_prob(s).sum())

        mean = x_hat + self.phasic.convolve(s * a)
        # See SingleTraceModel.model: under a coloured emission the mean at the `y` site
        # is the conditional one, so the structural mean is recorded separately.
        pyro.deterministic("mean_struct", mean)
        with seq_plate, time_plate:
            em = self._emission_dist(mean, obs=y)
            if obs_mask is not None:
                em = em.mask(observed_bins(obs_mask, y.shape, y.device,
                                           len(self.emission.ar_coeffs)))
            pyro.sample("y", em, obs=y)

    # -------------------------------------------------------------------- guide
    def guide(self, y: torch.Tensor, inj_t=None, tilt=None, obs_mask=None):
        """Mean-field guide: Normals on every latent, on their unconstrained axes.

        `tilt` and `obs_mask` are accepted and ignored: both are terms of the log
        JOINT, not of `q`, and Pyro requires the two signatures to match.
        """
        self._check_inj(inj_t)
        S, T = y.shape
        device, dtype = y.device, y.dtype
        tau = torch.as_tensor(self.tau, device=device, dtype=dtype)

        def free(name, loc_init, scale_init=0.1, shape=()):
            loc = pyro.param(f"q_{name}_loc",
                             torch.full(shape, float(loc_init), device=device, dtype=dtype))
            scale = pyro.param(f"q_{name}_scale",
                               torch.full(shape, float(scale_init), device=device, dtype=dtype),
                               constraint=dist.constraints.positive)
            return loc, scale

        fr = self.beta_guide.startswith("fullrank")
        if fr:
            # One correlated draw, unpacked onto the model's sites. Every coordinate
            # is unconstrained -- the scales because they are sampled through their
            # log -- so no transform and no Jacobian are involved. Including the
            # variance components matters because they correlate with mu_drug: a
            # larger between-animal spread means the cohort mean is less determined,
            # and a factorised q must call that dependence zero.
            n, where = self._block_layout()
            init_loc, init_diag = self._block_init(device, dtype)
            g_loc = pyro.param("q_global_loc", init_loc)
            g_tril = pyro.param("q_global_tril", torch.diag(init_diag),
                                constraint=dist.constraints.lower_cholesky)
            joint = pyro.sample("_global_joint",
                                dist.MultivariateNormal(g_loc, scale_tril=g_tril),
                                infer={"is_auxiliary": True})
            mu_b0 = pyro.sample("mu_b0", dist.Delta(joint[where["mu_b0"]][0]))
        else:
            joint = None
            mu_b0 = pyro.sample("mu_b0", dist.Normal(*free("mu_b0", self.beta0_anchor)))
        # Scales are sampled through their log, where the prior lives, so the guide
        # matches the model's axis and the block can include them untransformed.
        sigma_b0 = self._sample_log_scale(
            "b0", guide=True, device=device, dtype=dtype, joint=joint,
            index=(where["sigma_b0"].start if fr else None))
        drug_plate = pyro.plate("drug", self.n_drugs, dim=-1)
        if fr:
            with drug_plate:
                mu_drug = pyro.sample("mu_drug",
                                      dist.Delta(joint[where["mu_drug"]]))
        else:
            md_loc, md_scale = free("mu_drug", 0.0, 0.1, (self.n_drugs,))
            with drug_plate:
                mu_drug = pyro.sample("mu_drug", dist.Normal(md_loc, md_scale))

        # The non-centred effects join the block only under `fullrank_z`; under
        # `fullrank` they stay factorised, which is what the shipped results used.
        with pyro.plate("animal", self.n_animals, dim=-1):
            if "z_b0" in self._blocked_z():
                z_b0 = pyro.sample("z_b0", dist.Delta(joint[where["z_b0"]]))
            else:
                zb_loc, zb_scale = free("z_b0", 0.0, 0.5, (self.n_animals,))
                z_b0 = pyro.sample("z_b0", dist.Normal(zb_loc, zb_scale))

        sigmas, zs = {}, {}
        for lvl in self.levels:
            sigmas[lvl] = self._sample_log_scale(
                lvl, guide=True, device=device, dtype=dtype, joint=joint,
                index=(where[f"sigma_{lvl}"].start if fr else None))
            with pyro.plate(f"{lvl}_effect", self.level_size[lvl], dim=-1):
                if f"z_{lvl}" in self._blocked_z():
                    zs[lvl] = pyro.sample(f"z_{lvl}",
                                          dist.Delta(joint[where[f"z_{lvl}"]]))
                else:
                    zl, zsc = free(f"z_{lvl}", 0.0, 0.5, (self.level_size[lvl],))
                    zs[lvl] = pyro.sample(f"z_{lvl}", dist.Normal(zl, zsc))

        # Every latent above is reparameterized, so the per-session coefficients they
        # imply carry a gradient to all levels. Use the SAMPLES, not the param medians:
        # the analytic event KL below is the only term that sees the betas, so
        # substituting a point value there would cut the hierarchy off from the data --
        # the variance components would then be estimated by their priors alone.
        b0_seq, bi_seq = self._session_coeffs(mu_b0, sigma_b0, z_b0, mu_drug, sigmas, zs)

        # The tonic latents must be sampled in the guide too, with the same site
        # names and plate structure as the model -- the curve itself is not used
        # here, only the sites.
        self._tonic_sites(T, guide=True, device=device, dtype=dtype,
                          drug_plate=drug_plate)

        # ---- events and marks: identical to the single-trace guide, per session.
        gate = self._gate(T, device=device, dtype=dtype)
        if self.couple_events:
            anchor = self._prior_event_prob(b0_seq, bi_seq, gate).clamp(1e-9, 1 - 1e-9).logit()
            zeros = torch.zeros(S, T, device=device, dtype=dtype)
            s_logits = self.event_guide(y, self.inj_t, prior_logits=zeros) + anchor
        else:
            b0_0 = torch.full((S,), self.beta0_anchor, device=device, dtype=dtype)
            p0 = self._prior_event_prob(b0_0, torch.zeros_like(b0_0), gate).detach()
            s_logits = self.event_guide(
                y, self.inj_t, prior_logits=p0.clamp(1e-6, 1 - 1e-6).logit())

        W = self.mark_window
        # Initialised at the NOISE SCALE, capped by the prior. Neither of the obvious
        # choices works in both regimes, and for the same reason: what matters is not
        # how the starting mark compares to the prior but how the SPURIOUS DRIVE it
        # creates compares to the noise. q(s_t) is small but not zero at the ~90% of
        # windows holding no event, so every one contributes q.E[a] from step one.
        # Zero puts E[a] at 1.0 whatever the units -- 1.7 sigma_y on simulated
        # cohorts, 82 on real dF/F. The prior puts it at 0.02 on real recordings,
        # which is right, and 6.0 (57 sigma_y) on simulated ones, where the phasic
        # component reached 10^6 by step 1000. The smaller of the two is what both
        # regimes want: every mark starts below the level at which it could disturb
        # the fit, and the data raises the ones that hold an event. See the same
        # comment in points.py for the measurements.
        # A tensor, [S, 1] when the marks or the noise are per session (2026-10-20), so
        # each session's q(a) starts at its own min(mu_a, log sigma_y).
        _mu = self.phasic.mu_a.detach().to(device=device, dtype=dtype)
        mu_a_init = (_mu if getattr(self.phasic, "q_a_init", "noise") == "prior" else
                     torch.minimum(_mu, torch.log(self.emission.sigma_y.detach()).to(
                         device=device, dtype=dtype)))
        mu_a_init = torch.broadcast_to(mu_a_init.reshape(-1, 1) if mu_a_init.dim() else mu_a_init,
                                       (S, 1))
        if W and W > 1:
            n_win = -(-T // W)
            a_loc = pyro.param(
                "q_a_loc", mu_a_init.expand(S, n_win).clone()
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
                _MARK_SCALE_INIT * torch.ones(S, n_win, device=device, dtype=dtype),
                constraint=dist.constraints.interval(0.0, _MARK_SCALE_MAX),
            ).repeat_interleave(W, dim=-1)[:, :T]
        else:
            a_loc = pyro.param(
                "q_a_loc", mu_a_init.expand(S, T).clone())
            a_scale = pyro.param(
                "q_a_scale", _MARK_SCALE_INIT * torch.ones(S, T, device=device, dtype=dtype),
                constraint=dist.constraints.interval(0.0, _MARK_SCALE_MAX))

        seq_plate = pyro.plate("sequences", S, dim=-2)
        time_plate = pyro.plate("time", T, dim=-1)
        with seq_plate, time_plate:
            q_dist = dist.RelaxedBernoulli(temperature=tau, logits=s_logits)
            s = pyro.sample("s", q_dist)
            qa_dist = dist.LogNormal(a_loc, a_scale)
            a_draw = pyro.sample("a", qa_dist)

        if self.mark_kl != "sampled":
            mk = SingleTraceModel._lognormal_kl(a_loc, a_scale, self.phasic.mu_a, self.phasic.sigma_a)
            if self.mark_kl == "gated":
                mk = torch.sigmoid(s_logits).clamp_min(self.mark_gate_floor) * mk
            pyro.factor("a_sampled_logq_undo", -qa_dist.log_prob(a_draw).sum(),
                        has_rsample=True)
            pyro.factor("a_mark_kl", mk.sum(), has_rsample=True)

        if self.analytic_s_kl:
            p_beta = self._prior_event_prob(b0_seq, bi_seq, gate)
            pyro.factor("s_relaxed_logq_undo", -q_dist.log_prob(s).sum(), has_rsample=True)
            pyro.factor("s_analytic_kl",
                        SingleTraceModel._bernoulli_kl(s_logits, p_beta).sum(), has_rsample=True)

    # ----------------------------------------------------------------- readouts
    def _global_moments(self):
        """(loc, cov) over (mu_b0, mu_drug...) -- from the block, or assembled."""
        if self.beta_guide.startswith("fullrank"):
            loc = pyro.param("q_global_loc").detach()
            tril = pyro.param("q_global_tril").detach()
            return loc, tril @ tril.T
        loc = torch.cat([pyro.param("q_mu_b0_loc").detach().reshape(1),
                         pyro.param("q_mu_drug_loc").detach()])
        sc = torch.cat([pyro.param("q_mu_b0_scale").detach().reshape(1),
                        pyro.param("q_mu_drug_scale").detach()])
        return loc, torch.diag(sc ** 2)

    @torch.no_grad()
    def drug_posterior(self) -> dict[int, tuple[float, float]]:
        """(loc, scale) of q(mu_drug[d]) for each drug -- the headline estimand."""
        loc, cov = self._global_moments()
        return {d: (float(loc[1 + d]), float(cov[1 + d, 1 + d].sqrt()))
                for d in range(self.n_drugs)}

    @torch.no_grad()
    def _tonic_block_layout(self) -> dict[str, slice]:
        """Where each tonic latent sits in the tonic block, by mode."""
        out, i, D, S = {}, 0, self.n_drugs, self.n_sessions
        if not self.tonic_block:
            return out
        for n in _TONIC_COMPONENTS:
            out[f"mu_{n}"] = slice(i, i + D); i += D
        if self.tonic_block_mode in ("with_sessions", "with_sessions_level_drift") \
                and self.tonic_session_level:
            for n in _TONIC_COMPONENTS:
                out[f"z_{n}"] = slice(i, i + S); i += S
        if self.tonic_block_mode == "with_sessions_level_drift":
            for n in _TONIC_OFFSETS:
                out[n] = slice(i, i + S); i += S
        out["_n"] = i
        return out

    def _tonic_block_init(self, device, dtype):
        """Start the block where the factorised guide starts: same locations and widths."""
        tw = self._tonic_block_layout()
        loc = torch.zeros(tw["_n"], device=device, dtype=dtype)
        diag = torch.full((tw["_n"],), 0.1, device=device, dtype=dtype)
        for n in _TONIC_COMPONENTS:
            loc[tw[f"mu_{n}"]] = self.tonic_anchor[n].to(device=device, dtype=dtype)
            if f"z_{n}" in tw:
                diag[tw[f"z_{n}"]] = 0.5
        for n in _TONIC_OFFSETS:
            if n in tw:       # standardized: anchor + prior scale * u, u starting at 0
                loc[tw[n]] = 0.0
                diag[tw[n]] = 0.2
        return loc, diag

    @torch.no_grad()
    def _tonic_z_loc(self, name: str) -> torch.Tensor:
        """Posterior mean of the per-session tonic deviation z^psi, block or factorised."""
        tw = self._tonic_block_layout()
        if f"z_{name}" in tw:
            return pyro.param("q_tonic_block_loc").detach()[tw[f"z_{name}"]]
        return pyro.param(f"q_z_tonic_{name}_loc").detach()

    @torch.no_grad()
    def tonic_alpha_contrast(self, d1: int = 1, d2: int = 0) -> tuple[float, float]:
        """(loc, sd) of the log-height contrast alpha[d1] - alpha[d2] (see tonic_contrast)."""
        return self.tonic_contrast(d1, d2, "alpha")

    @torch.no_grad()
    def _tonic_mu_moments(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """(loc [D], cov [D, D]) of the per-drug tonic mean `name`, block or factorised."""
        if self.tonic_block:
            loc = pyro.param("q_tonic_block_loc").detach()
            L = pyro.param("q_tonic_block_tril").detach()
            cov = L @ L.T
            sl = self._tonic_block_layout()[f"mu_{name}"]
            return loc[sl].cpu(), cov[sl, sl].cpu()
        loc = pyro.param(f"q_tonic_mu_{name}_loc").detach().cpu()
        sc = pyro.param(f"q_tonic_mu_{name}_scale").detach().cpu()
        return loc, torch.diag(sc ** 2)

    @torch.no_grad()
    def tonic_contrast(self, d1: int, d2: int, name: str = "alpha") -> tuple[float, float]:
        """(loc, sd) of the per-drug tonic mean `name`[d1] - [d2]; alpha is the log height."""
        loc, cov = self._tonic_mu_moments(name)
        var = float(cov[d1, d1] + cov[d2, d2] - 2 * cov[d1, d2])
        return float(loc[d1] - loc[d2]), math.sqrt(max(var, 0.0))

    def drug_contrast(self, d1: int, d2: int) -> tuple[float, float]:
        """(loc, scale) of mu_drug[d1] - mu_drug[d2] under the mean-field posterior.

        The two are independent in `q`, so the contrast's variance is the sum. This is
        the quantity the drug comparison reports, and unlike a two-stage difference of
        per-trace fits its interval carries the between-animal component. How wide it
        is depends on the design: with one drug per animal the comparison is between
        animals and `sigma_animal` sets the floor; when animals receive several drugs
        the animal effect cancels and the same recordings buy a tighter interval.
        """
        loc, cov = self._global_moments()
        i, j = 1 + d1, 1 + d2
        # Var(a - b) = Var(a) + Var(b) - 2 Cov(a, b). Under the mean-field guide the
        # covariance is zero by construction and this reduces to the sum; under the
        # full-rank block it does not, and ignoring it would misreport the very
        # quantity the block was added to get right.
        var = float(cov[i, i] + cov[j, j] - 2 * cov[i, j])
        return (float(loc[i] - loc[j]), math.sqrt(max(var, 0.0)))

    @torch.no_grad()
    def _z_loc(self, name: str) -> torch.Tensor:
        """Posterior mean of a non-centred effect, from the block or its own site.

        One accessor because the readouts below must not care which guide is in use;
        reading `q_z_<name>_loc` directly is what would break the moment the effect
        moves into the block.
        """
        if name in self._blocked_z():
            _, where = self._block_layout()
            return pyro.param("q_global_loc").detach()[where[name]]
        return pyro.param(f"q_{name}_loc").detach()

    @torch.no_grad()
    def session_beta_inj(self) -> torch.Tensor:
        """Posterior mean of beta_inj for each session, [S] -- the shrunk estimates."""
        bi = self._global_moments()[0][1:][self.drug]
        for lvl in self.levels:
            if lvl == "b0_session":        # acts on the baseline, not the effect
                continue
            sigma = self._scale_median(lvl)
            bi = bi + sigma * self._z_loc(f"z_{lvl}")[self._index_of(lvl)]
        return bi

    @torch.no_grad()
    def animal_beta0(self) -> torch.Tensor:
        """Posterior mean of the animal-level baseline log-rate, [A]."""
        return (self._global_moments()[0][0]
                + self._scale_median("b0")
                * self._z_loc("z_b0"))

    @torch.no_grad()
    def session_beta0(self) -> torch.Tensor:
        """Posterior mean of the baseline log-rate actually used per session, [S].

        The animal's level plus its day-to-day deviation, when the design supports
        estimating one.
        """
        b0 = self.animal_beta0()[self.animal]
        if "b0_session" in self.levels:
            b0 = b0 + (self._scale_median("b0_session")
                       * self._z_loc("z_b0_session"))
        return b0

    @torch.no_grad()
    def _scale_median(self, level: str) -> float:
        """Posterior median of sigma_<level>, from the block or its own site."""
        if self.beta_guide.startswith("fullrank"):
            # Through the layout, not by re-deriving the offset: the same arithmetic
            # written twice is the thing that breaks when a level is added.
            _, where = self._block_layout()
            i = where[f"sigma_{level}"].start
            return float(pyro.param("q_global_loc").detach()[i].exp())
        return float(pyro.param(f"q_log_sigma_{level}_loc").detach().exp())

    @torch.no_grad()
    def variance_components(self) -> dict[str, float]:
        """Posterior median of each active variance component, on the log-rate axis.

        **Which component governs a drug contrast depends on the design**, and
        getting this backwards cost this project a week. In a CROSSED design -- every
        animal receiving every drug -- `sigma_animal` *cancels from the contrast*:
        the animal effect enters `mu_drug[a]` and `mu_drug[b]` identically and
        subtracts out, exactly as it does in a paired t-test. What remains is
        `sigma_cell`, the animal x drug interaction, which is the spread the contrast
        is actually estimated against and the reason its interval does not shrink
        with the number of sessions.

        In an UNCROSSED design -- each animal receiving one drug -- there is no
        pairing to exploit, nothing cancels, and `sigma_animal` governs the contrast
        directly. The real cohort is partially crossed and sits between the two, so
        neither component can be read as "the" one without checking `design_summary`.

        This docstring previously named `sigma_animal` unconditionally, and variants
        of the guide were then ranked on how well they recovered it -- selecting, on
        a crossed design, for the one parameter that does not enter the estimand. It was
        corrected on 2026-09-14.
        """
        return {f"sigma_{lvl}": self._scale_median(lvl)
                for lvl in ("b0", *self.levels)}

    @torch.no_grad()
    def event_posterior(self, y: torch.Tensor) -> torch.Tensor:
        """Per-bin event probabilities q(s_t = 1), [S, T]."""
        if hasattr(self.event_guide, "fitted_logits"):
            return torch.sigmoid(self.event_guide.fitted_logits(y))
        return torch.sigmoid(pyro.param("q_s_logits").detach())
