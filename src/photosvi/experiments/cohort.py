"""Simulate a cohort under the crossed design, for the two-stage vs hierarchical study.

Both arms of the comparison must see **exactly the same recordings**, or the study
measures the simulator rather than the methods. So the cohort is a pure function of
its seed: `simulate_cohort(spec, seed)` is deterministic and is called independently
by the CPU worker (which fits one session at a time) and the GPU worker (which fits
the whole cohort), with no cached intermediate to fall out of sync.

The design is the one that gets run in practice: **every animal receives vehicle and
both opioids**, three sessions on three different days. Three consequences that the
simulator has to reproduce, because each of them is something a method could get
wrong:

  * **Saline is not a null session.** Handling and the injection itself raise the
    event rate, so `mu_drug[saline] > 0`. A pre/post contrast that ignores vehicle
    attributes that rise to the drug.
  * **The two opioids have different pharmacokinetics.** The tonic carrier is the
    drug's absorption and elimination, and opioids differ in exactly that: here the
    first peaks at ~7 min and is still at a third of peak at the end of the
    recording, the second peaks at ~2.4 min and is essentially gone. Giving both
    drugs the same curve, as an earlier version of this simulator did, makes a
    drug-structured tonic un-testable -- there would be nothing to find, and a null
    result would mean nothing.
  * **Baseline drifts between sessions.** Different day, different bleaching,
    different fibre placement, so an animal's *pre-injection* rate differs across
    its own three recordings (`sigma_b0_session`). A model with one baseline per
    animal has to put that misfit somewhere, and the only place left is the
    injection effect.
  * **Animals differ in responsiveness** (`sigma_animal`), which is what makes the
    within-animal design worth running: that term cancels from a within-animal
    contrast and does not cancel from a between-animal one.

**Session order is NOT modelled here.** Real repeated-measures opioid designs have
an order confound -- tolerance and sensitization are real -- and the remedy is
counterbalancing in the experiment, not a term in the simulator. Adding an order
effect here would let the hierarchical arm demonstrate that it can absorb a
confound that the actual cohort may or may not have. When the covariate extension
lands, order becomes a column of the design matrix and *then* it is worth
simulating, together with the counterbalanced and non-counterbalanced cases.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field

import torch
from scipy.signal import lfilter

from photosvi.models.points import (
    AdditiveEmission,
    HierarchicalModel,
    PhasicEventProcess,
    TonicBaseline,
    WindowedCategoricalEventGuide,
    build_kernel,
    fit_tonic_baseline,
    stack_tonic_baselines,
)

SALINE, OPIOID1, OPIOID2 = 0, 1, 2
DRUG_NAMES = ("saline", "opioid1", "opioid2")


@dataclass
class CohortSpec:
    """The generative truth for one simulated cohort.

    `mu_drug` is on the log-rate-ratio axis: `mu_drug[d]` is the mean log ratio of
    post- to pre-injection event rate under drug d. The estimand of the study is the
    contrast `mu_drug[OPIOID2] - mu_drug[OPIOID1]`; `mu_drug[SALINE]` is the
    handling response the vehicle sessions control for.
    """

    n_animals: int = 8
    # "crossed": every animal receives every drug. "uncrossed": every animal
    # receives vehicle and one opioid. See `indices` -- it changes what the estimand
    # is a contrast *of*, not merely how many recordings there are.
    design: str = "crossed"
    mu_drug: tuple[float, ...] = (0.20, 1.10, 1.39)     # saline, 3.0x, 4.0x
    sigma_animal: float = 0.25       # between-animal responsiveness, within drug
    sigma_cell: float = 0.10         # animal x drug interaction
    beta0_mean: float = -2.85        # cohort mean baseline log-rate
    sigma_b0: float = 0.22           # between-animal baseline spread
    sigma_b0_session: float = 0.20   # day-to-day baseline drift within an animal
    # emission and marks, frozen from the morphine ground truth (nb65 -> nb66)
    sigma_y: float = 0.1062
    mu_a: float = 1.7666
    sigma_a: float = 0.2120
    # timing
    fs: int = 25
    T: int = 60000                   # 40 min
    inj_t: int = 15000               # 10 min
    inj_jitter: int = 1500           # +/- bins, so no method can key off a fixed bin
    # Tonic: one Bateman curve per drug, each rising AND falling inside the
    # recording. Indexed by the drug codes above (saline, opioid1, opioid2).
    tonic_b: float = -0.3406
    # Peak height of each drug's tonic bump, in the trace's units -- see tonic.py for
    # why this and not the textbook amplitude. Converted from the amplitudes this
    # field held until 2026-09-17 (1.2, 7.0, 6.0) by multiplying each by its own
    # Bateman peak factor, so the simulated recordings are unchanged to float
    # precision and every cohort already fitted remains a cohort from this generator.
    # `test_peak_parameterisation_reproduces_the_old_curves` asserts that.
    tonic_peak: tuple[float, ...] = (0.4618802153517006,   # vehicle carries a small one
                                     3.939445759311477,
                                     3.209953463886826)
    tonic_k_rise: tuple[float, ...] = (0.0060, 0.0050, 0.0140)
    tonic_k_decay: tuple[float, ...] = (0.0020, 0.0009, 0.0028)
    # Per HOUR (was -0.0022 per second until 2026-09-17; same drift, times 3600).
    # Bleaching: a property of the recording, not the drug.
    tonic_drift: float = -7.92
    # A slow stochastic component, ABSENT from this simulator until 2026-09-30 and
    # present in every real recording. Before the injection the Bateman tonic above is
    # a constant -- measured, its pre-injection standard deviation is exactly zero --
    # so a simulated baseline is a flat line plus events plus white noise, and a real
    # one is not. Fitting an Ornstein-Uhlenbeck process to the slow part of two raw
    # .doric recordings gives a corner near 0.23 Hz and an amplitude that puts the
    # slow band's spread at about 1.2 times the transient band's; the simulator sat
    # at 0.04.
    #
    # DEPRECATED 2026-09-30, superseded by `ar_rho`. An Ornstein-Uhlenbeck sampled at
    # a fixed step IS an AR(1) with rho = exp(-dt/tau): tau = 0.7 s at 25 Hz is exactly
    # rho = 0.944, and at matched scale the two agree on every statistic we measure
    #. Keeping both was keeping one process under two names.
    #
    # The AR(1) is the one to keep, and not only for parsimony. This term adds to the
    # MEAN, where the model has no matching term, so it is structure the fit cannot see
    # and can only absorb into events. `ar_rho` is in the EMISSION, which the model
    # scores. The same process moves from unmodelled to modelled.
    #
    # Still honoured so that cohorts generated with it reproduce exactly; setting it
    # warns and tells you the equivalent pair. The translation is deliberately NOT
    # automatic -- it would change what the fit sees, which is the whole point.
    ou_sd: float = 0.0
    ou_tau: float = 0.7

    # Lag-one correlation of the emission residual; see `AdditiveEmission`. Zero is the
    # white emission, so every cohort before this date is unchanged. `sigma_y` carries
    # the amplitude and `ar_rho` the colour, which is the pair that replaces
    # (`ou_sd`, `ou_tau`).
    ar_rho: float = 0.0

    # The full autoregressive order, for an emission with more than one pole. `ar_rho` is
    # the AR(1) spelling and stays for configs that use it; `ar_coeffs` wins when both are
    # given. The recordings want two poles -- about 2 s and 50 ms -- and one cannot carry
    # both: fitted to the data an AR(1) spends its only pole on the slow one and leaves the
    # trace as rough as white noise.
    ar_coeffs: tuple | list | None = None
    kernel_length: int = 125
    kernel: dict = field(default_factory=lambda: dict(
        family="biphasic", k_rise_fast=20.0, k_decay_fast=9.0,
        k_rise_slow=8.0, k_decay_slow=3.5))

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def n_drugs(self) -> int:
        return len(self.mu_drug)

    def tonic_peak_s(self, d: int) -> float:
        """Bateman peak time for drug `d`, in seconds after injection."""
        kr, kd = self.tonic_k_rise[d], self.tonic_k_decay[d]
        return math.log(kr / kd) / (kr - kd)

    @property
    def n_sessions(self) -> int:
        if self.design == "crossed":
            return self.n_animals * self.n_drugs
        # Uncrossed: one condition per animal when there are two, vehicle plus one opioid
        # when there are three. See `indices` for why they differ.
        return self.n_animals * (1 if self.n_drugs == 2 else 2)

    def indices(self) -> tuple[list[int], list[int]]:
        """(animal, drug) per session, for whichever design this spec names.

        The two designs are the extremes an experimenter actually chooses between,
        and they differ in what the opioid-versus-opioid contrast *is*:

        * `crossed` -- every animal receives every drug. The contrast is
          within-animal, so the animal effect enters both of its terms and cancels,
          exactly as it does in a paired test. What is left governing the width is
          the animal x drug interaction.
        * `uncrossed` -- every animal receives vehicle and exactly one opioid,
          alternating which. The contrast is now between-animal: nothing cancels,
          and the between-animal spread enters it directly. It is also the design in
          which each animal x drug cell holds a single session, so the interaction
          and a session residual are the same partition and only one of them can be
          identified -- `_detect_levels` is what decides that, from the indices.

        Both are generated at the same number of RECORDINGS, which is the thing an
        experiment pays for: 8 animals x 3 sessions and 12 animals x 2 sessions are
        both 24. Comparing them at equal animal counts instead would confound the
        design with the sample size and answer a question nobody is asking.
        """
        if self.design == "crossed":
            animal = [a for a in range(self.n_animals) for _ in range(self.n_drugs)]
            drug = [d for _ in range(self.n_animals) for d in range(self.n_drugs)]
            return animal, drug
        if self.design != "uncrossed":
            raise ValueError(f"design must be 'crossed' or 'uncrossed', got {self.design!r}")
        # Alternate which opioid an animal sees, so the two groups are equal in size and
        # neither is confounded with position in the cohort.
        #
        # With TWO conditions there is no vehicle session to add: breaking the crossing
        # means one condition per animal and nothing else, which is exactly what hiding
        # half of a two-condition cohort produces, and is the design the paper's uncrossed
        # arm on the real recordings simulates. The three-drug form keeps the vehicle
        # session because its contrast is between two opioids and vehicle is a third arm.
        animal, drug = [], []
        for a in range(self.n_animals):
            if self.n_drugs == 2:
                animal += [a]
                drug += [a % 2]
            else:
                animal += [a, a]
                drug += [SALINE, OPIOID1 if a % 2 == 0 else OPIOID2]
        return animal, drug

    def true_contrast(self, d1: int = OPIOID2, d2: int = OPIOID1) -> float:
        return self.mu_drug[d1] - self.mu_drug[d2]


#: Environment overrides on the simulated cohort, and what each one is for.
#:
#: `SIM_*` change what the data CONTAINS, which is how one asks what carrying a level costs
#: when the data does not have it. `DESIGN` and `N_ANIMALS` change the experiment itself.
_COHORT_ENV = {
    "SIM_SIGMA_CELL": ("sigma_cell", float),
    "SIM_SIGMA_ANIMAL": ("sigma_animal", float),
    "SIM_SIGMA_B0SES": ("sigma_b0_session", float),
    "DESIGN": ("design", str),
    "N_ANIMALS": ("n_animals", int),
    # The emission scale of the simulation, to sweep the per-event SNR (2026-10-09).
    "SIM_SIGMA_Y": ("sigma_y", float),
}


def apply_cohort_env(cfg: dict) -> dict:
    """Apply the environment's overrides to a cohort config, in place, and return it.

    **Every arm must call this, and that is the whole point.** The three arms of the
    comparison are only comparable because they regenerate the same cohort from
    `(spec, seed)`; an override applied by one and not the others silently hands them
    different recordings, and nothing downstream could detect it -- the numbers would
    simply be wrong in a way that looks like a result. Until 2026-10-02 the hierarchical
    arm applied these and the two-stage and GuPPy arms did not. It never bit, because the
    only study that used an override ran the hierarchical arm alone.
    """
    c = cfg["cohort"]
    for env, (key, cast) in _COHORT_ENV.items():
        raw = os.environ.get(env)
        if raw is not None:
            c[key] = cast(raw)
    return cfg


def spec_from_config(cfg: dict, delta: float) -> CohortSpec:
    """Build the cohort truth from the study config and one effect size.

    **The single definition of the simulated truth.** Both arms of the comparison
    call this; duplicating the construction in each worker is how two arms end up
    quietly fitting different data while every test still passes, and the resulting
    comparison would be meaningless in a way nothing downstream could detect.

    `delta` places the second opioid above the first on the log-rate axis, so
    `delta = 0` is the null: both drugs active, indistinguishable from each other.

    **Two conditions instead of three** when the config omits `mu_opioid1`, which is the
    design the recordings actually have: a vehicle arm and a treated arm, with `delta` the
    difference between them. The three-drug form is kept because the shipped study uses it
    and its tonic and effect vectors are indexed by drug; a config that names `mu_opioid1`
    gets exactly the cohorts it always did.
    """
    c = cfg["cohort"]
    if "mu_opioid1" in c:
        mu_drug = (c["mu_saline"], c["mu_opioid1"], c["mu_opioid1"] + float(delta))
    else:
        mu_drug = (c["mu_saline"], c["mu_saline"] + float(delta))
    # Noise-unit marks (opt-in, experiments.marks): the simulated marks are drawn from the
    # same fixed distribution in units of sigma_y that the fit assumes.
    from photosvi.experiments.marks import noise_unit_generation, noise_unit_mode
    nu = noise_unit_mode(cfg.get("fit"))
    # (A sigma_y of 0 is the real cohort's placeholder, set from the recordings after.)
    if nu is not None and c["sigma_y"] > 0:
        c = dict(c)
        c["mu_a"], c["sigma_a"] = noise_unit_generation(cfg.get("fit"), c["sigma_y"],
                                                        c["mu_a"], c["sigma_a"])
    return CohortSpec(
        n_animals=c["n_animals"],
        # Defaulted, not indexed, because the shipped study is crossed and the
        # uncrossed arm is a variant that names itself: a config that predates the
        # field should keep producing exactly the cohorts it always produced.
        design=c.get("design", "crossed"),
        mu_drug=mu_drug,
        sigma_animal=c["sigma_animal"], sigma_cell=c["sigma_cell"],
        beta0_mean=c["beta0_mean"], sigma_b0=c["sigma_b0"],
        sigma_b0_session=c["sigma_b0_session"], sigma_y=c["sigma_y"],
        # Defaulted rather than indexed, like `design`: the simulated study has no
        # reason to set them and a config that predates the keys should keep producing
        # the cohorts it always did. Real recordings do set them, because the mark
        # scale is in the units of the trace and the shipped value assumes transients
        # tens of times more prominent than a dF/F recording carries.
        mu_a=c.get("mu_a", CohortSpec.mu_a), sigma_a=c.get("sigma_a", CohortSpec.sigma_a),
        T=c["T"], inj_t=c["inj_t"], inj_jitter=c["inj_jitter"],
        # Defaulted rather than required, like the keys above, but it must be READ: a
        # config that names 48 Hz and a 48 Hz kernel length while the spec silently stays
        # at 25 produces recordings at one rate with a basis built for another, and
        # nothing downstream says so. Five errors of exactly this shape have been found
        # in this project already.
        fs=int(c.get("fs", CohortSpec.fs)),
        tonic_b=float(c.get("tonic_b", CohortSpec.tonic_b)),
        tonic_peak=tuple(c["tonic_peak"]),
        ou_sd=c.get("ou_sd", CohortSpec.ou_sd),
        ou_tau=c.get("ou_tau", CohortSpec.ou_tau),
        ar_rho=c.get("ar_rho", CohortSpec.ar_rho),
        ar_coeffs=c.get("ar_coeffs", CohortSpec.ar_coeffs),
        # The transient basis, defaulted rather than indexed so that a config written
        # before 2026-09-30 still produces K5. Exposed because K5 is no longer
        # obviously the right one: it was selected on simulation-side criteria in
        # September, and the first raw recordings put 1.16% of its power above 10 Hz
        # against 0.024% in the data. K7 in scripts/kernel_candidates.py is the same
        # shape with every rate constant scaled by 0.25, which lands at 0.022% --
        # `kernel: {family: biphasic, k_rise_fast: 5.0, k_decay_fast: 2.25,
        # k_rise_slow: 2.0, k_decay_slow: 0.875}` selects it.
        #
        # Within the biphasic family, `kernel_length` does NOT need raising with it,
        # which is worth stating because the opposite seems obvious. The builder sets
        # the slow lobe's scale from the sums over whatever length it is given, so the
        # basis is mean-free at any truncation: measured across 125 to 700 taps, sum(h)
        # stays at 1e-6 and the only thing that moves is the undershoot, from -0.247 to
        # -0.257 at the shortest. Peak, width and high-frequency share are identical
        # to three decimals.
        #
        # `family: bigamma` is the exception, and the default 125 taps is NOT enough for
        # it -- `bigamma_kernel` refuses that length rather than returning a kernel whose
        # undershoot has grown to swallow the transient. At 25 Hz it wants 143 taps and
        # this default gives 125, so a bigamma config must raise `kernel_length` too.
        # K9 in scripts/kernel_candidates.py is the fitted member of that family:
        # `kernel: {family: bigamma, shape_fast: 20.34, rate_fast: 15.65,
        # shape_slow: 15.40, rate_slow: 3.95}` with `kernel_length: 200`.
        kernel=c.get("kernel", CohortSpec().kernel),
        kernel_length=c.get("kernel_length", CohortSpec.kernel_length),
        tonic_k_rise=tuple(c["tonic_k_rise"]),
        tonic_k_decay=tuple(c["tonic_k_decay"]),
        tonic_drift=c["tonic_drift"])


def simulate_cohort(spec: CohortSpec, seed: int) -> dict:
    """Generate one cohort. Deterministic in `(spec, seed)`.

    Returns `y` [S, T] plus the indices, the injection bins and the per-session
    truth: the coefficients actually drawn, and the count MLE of `beta_inj` from the
    realized onsets -- which is the best any method could do on that recording, and
    therefore the right reference for a per-session estimate.
    """
    g = torch.Generator().manual_seed(int(seed))
    animal, drug = spec.indices()
    S, T, dt = spec.n_sessions, spec.T, spec.dt
    kern = build_kernel(dict(spec.kernel), spec.kernel_length, dt)

    # animal-level draws
    z_anim = torch.randn(spec.n_animals, generator=g)
    z_b0 = torch.randn(spec.n_animals, generator=g)
    z_cell = torch.randn(S, generator=g)
    z_b0s = torch.randn(S, generator=g)
    jitter = torch.randint(-spec.inj_jitter, spec.inj_jitter + 1, (S,), generator=g)

    ys, inj_ts, beta_inj_true, beta0_true, mles, n_pre, n_post = [], [], [], [], [], [], []
    for s in range(S):
        a, d = animal[s], drug[s]
        inj = int(spec.inj_t + jitter[s])
        bi = (spec.mu_drug[d]
              + spec.sigma_animal * float(z_anim[a])
              + spec.sigma_cell * float(z_cell[s]))
        b0 = (spec.beta0_mean
              + spec.sigma_b0 * float(z_b0[a])
              + spec.sigma_b0_session * float(z_b0s[s]))
        tonic = TonicBaseline(
            dt=dt, b=spec.tonic_b, peak=spec.tonic_peak[d],
            k_rise=spec.tonic_k_rise[d], k_decay=spec.tonic_k_decay[d],
            drift=spec.tonic_drift)
        x = tonic.baseline(T, inj)
        if spec.ou_sd > 0:
            warnings.warn(
                f"CohortSpec.ou_sd is deprecated: an OU sampled at dt is an AR(1). "
                f"Use sigma_y={spec.ou_sd:g} with "
                f"ar_rho={math.exp(-dt / spec.ou_tau):.5f} instead "
                f"(= exp(-dt/{spec.ou_tau:g})), which puts the same process in the "
                f"emission, where the model can score it.",
                DeprecationWarning, stacklevel=2)
            # Exact discrete OU, not an Euler step: with dt/tau around 0.06 the two
            # differ by a few percent in the stationary variance, and the amplitude is
            # the quantity being calibrated against real recordings.
            rho = math.exp(-dt / spec.ou_tau)
            innov = spec.ou_sd * math.sqrt(1 - rho**2)
            eps = torch.randn(T, generator=g).numpy()
            eps[0] = spec.ou_sd * eps[0] / innov          # stationary first sample
            # lfilter rather than a Python loop: the recursion is the same and 60000
            # steps per session times 24 sessions is a minute of interpreter time.
            ou = lfilter([innov], [1.0, -rho], eps)
            x = x + torch.as_tensor(ou, dtype=x.dtype)
        ex = PhasicEventProcess(kern, beta0=b0, beta_inj=bi, mu_a=spec.mu_a,
                                sigma_a=spec.sigma_a, dt=dt, history=None).sample(T, inj, 1, g)
        y = (AdditiveEmission(sigma_y=spec.sigma_y, ar_coeffs=spec.ar_coeffs)
             if spec.ar_coeffs else
             AdditiveEmission(sigma_y=spec.sigma_y, ar_rho=spec.ar_rho)).sample(
            x + ex["c"], generator=g)["y"][0]
        onsets = ex["s"][0].nonzero().reshape(-1)
        pre = int((onsets < inj).sum())
        post = int((onsets >= inj).sum())
        mle = (torch.log(torch.tensor(
            (post / ((T - inj) * dt)) / (pre / (inj * dt)))).item()
            if pre and post else float("nan"))
        ys.append(y)
        inj_ts.append(inj)
        beta_inj_true.append(bi)
        beta0_true.append(b0)
        mles.append(mle)
        n_pre.append(pre)
        n_post.append(post)

    return {
        "y": torch.stack(ys),
        "animal": animal,
        "drug": drug,
        "inj_t": inj_ts,
        "beta_inj_true": beta_inj_true,
        "beta0_true": beta0_true,
        "mle": mles,
        "n_pre": n_pre,
        "n_post": n_post,
        "seed": int(seed),
        "tonic_peak_s": [spec.tonic_peak_s(d) for d in range(spec.n_drugs)],
    }


def _flag(env: str, default) -> bool:
    """An environment override of a boolean config key; unset means the config's value."""
    v = os.environ.get(env)
    return bool(default) if v is None else v not in ("", "0", "false")


def _tonic_block_mode(default):
    """`fit.tonic_block` (False, True, "with_sessions", "with_sessions_level_drift"),
    overridable by TONIC_BLOCK=0/1/<mode>."""
    v = os.environ.get("TONIC_BLOCK")
    if v is None:
        return default
    return {"": False, "0": False, "false": False, "1": True, "true": True}.get(v, v)


def build_cohort_model(cfg: dict, spec: CohortSpec, coh: dict, guide: str | None = None,
                       couple_events: bool = False,
                       levels: tuple[str, ...] | None = None):
    """The fitting side of a cohort study: matched filters, tonic warm start, model.

    Extracted so that every arm fits the SAME model. The comparison the study makes is
    between analyses of identical data; an arm that quietly built a different model --
    a different warm start, a different prior scale -- would be measuring something
    else and nothing in the outputs would say so.

    `couple_events` lets the guide's per-bin event logits depend on the sampled
    coefficients rather than on a detached constant. That is a dependence between the
    globals and the ~2.9 M per-bin latents, and it is the one the interval work has
    not been able to reach: the globals are already jointly Gaussian, and putting the
    non-centred effects in the block displaces variance instead of adding it.

    `levels` forces the random-effect levels instead of letting the design choose
    them. Dropping a level does not delete the variation it carried -- that gets
    reassigned to whatever is left -- so this is a way of asking where a level's
    variance goes, not only whether the model is simpler without it.

    `guide` selects the variational family; `None` means the config's
    `fit.hier_beta_guide`, which is where the shipped choice and the measurement
    behind it live. There is deliberately no literal default here. The one this
    signature used to carry was `meanfield`, and a caller that simply forgot the
    argument got the study's worst-calibrated family without a line of output saying
    so -- which is how the two-stage arm ran its whole course.

    Returns `(model, Y, mf)`. The caller owns the optimiser and the stopping rule,
    which is where the arms legitimately differ.
    """
    from photosvi.benchmark import matched_filter_score

    Y, inj = coh["y"], coh["inj_t"]
    # The session count comes from the RECORDINGS, not from the spec. For a simulated
    # cohort the two agree by construction -- `spec.n_sessions` is n_animals times the
    # number of drugs -- and taking the spec's was harmless until a real cohort was
    # passed in: fifteen recordings against a spec that computed twenty-four, and the
    # first thing indexed past the end was the injection list.
    S, dt = Y.shape[0], spec.dt
    if not len(inj) == len(coh["animal"]) == len(coh["drug"]) == S:
        raise ValueError(
            f"cohort is inconsistent: {S} recordings, {len(inj)} injection bins, "
            f"{len(coh['animal'])} animal indices, {len(coh['drug'])} drug indices"
        )
    f, w = cfg["fit"], cfg["warm"]
    kern = build_kernel(dict(spec.kernel), spec.kernel_length, dt)
    # One matched filter per session: a cohort is S different recordings, and warm
    # starting every session at one animal's transients is the wrong init for all
    # the others.
    mf = torch.stack([
        torch.as_tensor(matched_filter_score(Y[s:s + 1], kern, inj[s], dt=dt),
                        dtype=torch.float32).reshape(-1) for s in range(S)])
    # The baseline log-rate anchor, from the recordings as in the single-trace model:
    # the mean over sessions of each one's matched-filter count of pre-injection peaks
    # over its pre-injection exposure. It used to be `spec.beta0_mean`, a config
    # constant, which for real recordings is a guess and for simulated ones is the
    # answer (changed 2026-10-07).
    from photosvi.experiments.sweep import mf_anchor
    d = cfg["detection"]
    beta0_anchor = float(torch.tensor([
        mf_anchor(mf[s].numpy(), int(inj[s]), dt,
                  thresh=d["mf_thresh"], refractory=d["refractory"])[0]
        for s in range(S)]).mean())
    tonic = stack_tonic_baselines(
        [fit_tonic_baseline(Y[s:s + 1], inj[s], dt=dt) for s in range(S)])
    # One sigma_y per session (opt-in, experiments.marks.per_session_sigma), each estimated
    # as the single-trace model does, from that session's pre-injection segment with the
    # AR correction, and fixed. Otherwise the cohort's single sigma_y, as always.
    from photosvi.experiments.marks import per_session_sigma
    from photosvi.models.points import estimate_log_noise_scale
    sigma_fit = spec.sigma_y
    if per_session_sigma(f):
        sigma_fit = [math.exp(estimate_log_noise_scale(
            Y[s:s + 1], int(inj[s]), ar_coeffs=list(spec.ar_coeffs) or None)) for s in range(S)]
    em = (AdditiveEmission(sigma_y=sigma_fit, ar_coeffs=spec.ar_coeffs)
          if spec.ar_coeffs else AdditiveEmission(sigma_y=sigma_fit, ar_rho=spec.ar_rho))
    em.log_sigma_y.requires_grad_(False)
    model = HierarchicalModel(
        PhasicEventProcess(kern, beta0=spec.beta0_mean, beta_inj=0.0, mu_a=spec.mu_a,
                           sigma_a=spec.sigma_a, dt=dt, history=None),
        tonic, em, animal=coh["animal"], drug=coh["drug"], inj_t=inj,
        beta0_anchor=beta0_anchor, tau=f["tau"],
        mu_b0_prior_scale=f["mu_b0_prior_scale"],
        mu_drug_prior_scale=f["mu_drug_prior_scale"],
        event_guide=WindowedCategoricalEventGuide(
            mf, window=f["window"], thresh=w["thresh"], gain=w["gain"]),
        couple_events=couple_events,
        # The floor on the gated mark divergence, from the config so a cohort can ask for
        # it without every caller growing an argument. Absent means zero, which is the
        # behaviour everything was fitted with before it existed.
        mark_gate_floor=f.get("mark_gate_floor", 0.0),
        block_init_at_prior=f.get("block_init_at_prior", False),
        # Opt-in joint block over the per-drug tonic means (2026-10-20); absent, off.
        tonic_block=_tonic_block_mode(f.get("tonic_block", False)),
        beta_guide=f["hier_beta_guide"] if guide is None else guide,
        levels=levels)
    # The mark distribution's prior and q(a)'s starting point, from the config's `fit`
    # block (2026-10-09): `mark_prior: {mu_sd, log_sigma_sd}` anchors mu_a and sigma_a on
    # the visible peaks of every session's pre-injection segment, pooled as for the real
    # recordings; `mark_init: prior` starts q(a) at mu_a. Absent, nothing changes.
    from photosvi.experiments.marks import fix_marks_at, noise_unit_marks, noise_unit_mode
    nu = noise_unit_mode(f)
    mp = None if nu is not None else f.get("mark_prior")
    if mp:
        from photosvi.experiments.marks import estimate_mark_distribution
        segs = [Y[s, :int(inj[s])].detach().cpu().numpy() for s in range(S)]
        model.phasic.set_mark_prior(*estimate_mark_distribution(segs, 1.0 / dt),
                                    mu_sd=float(mp.get("mu_sd", 0.2)),
                                    log_sigma_sd=float(mp.get("log_sigma_sd", 0.3)))
    if f.get("mark_init"):
        model.phasic.q_a_init = f["mark_init"]
    # `fix_marks: true` holds mu_a and sigma_a at the empirical estimate instead of fitting
    # them (2026-10-10). Pooling S recordings, the likelihood pulls the mark distribution
    # down past any reasonable prior width (sd 0.016 still let mu_a fall 0.84 below its
    # centre) and the contrast was compressed to ~0.4 of its value; fixed, the contrast is
    # recovered (0.46 and 0.41 against 0.5) with a residual at the noise level.
    if f.get("fix_marks"):
        model.phasic.mu_a.requires_grad_(False)
        model.phasic.log_sigma_a.requires_grad_(False)
    # Noise-unit marks: fixed at a_t = sigma_y r_t with the sigma_y this fit uses.
    if nu is not None and isinstance(sigma_fit, list):
        from photosvi.experiments.marks import fix_marks_per_session
        fix_marks_per_session(model.phasic, [noise_unit_marks(v, *nu)[0] for v in sigma_fit],
                              nu[1])
    elif nu is not None:
        fix_marks_at(model.phasic, *noise_unit_marks(spec.sigma_y, *nu))
    model.sigma_y_sessions = (list(sigma_fit) if isinstance(sigma_fit, list)
                              else [float(spec.sigma_y)] * S)
    model.train(False)
    return model, Y, mf
