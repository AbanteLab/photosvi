"""Smoke tests for the points tonic baseline.

Covers the deterministic tonic node only: the reparameterization (positivity +
k_rise > k_decay ordering), the baseline shape (injection gating, Bateman bump,
linear drift), the peak time, and the additive mean y = x_hat + c assembled
against the phasic PhasicEventProcess. The hierarchical drug->animal prior and
the emission noise are not built here.
"""

from __future__ import annotations

import math

import torch

from photosvi.models.points import (
    PhasicEventProcess,
    TonicBaseline,
    bi_exponential_kernel,
    fit_tonic_baseline,
)


def test_transform_positivity_and_ordering():
    """peak=exp(alpha)>0 and k_rise=exp(rho)+exp(gamma)>k_decay by construction."""
    tb = TonicBaseline(b=0.1, peak=2.0, k_rise=0.01, k_decay=0.002, drift=-1e-4)
    assert tb.peak.item() > 0
    assert tb.A.item() > 0          # the derived textbook amplitude, also positive
    assert tb.k_decay.item() > 0
    assert tb.k_rise.item() > tb.k_decay.item()
    # round-trips the constrained values it was built from
    assert abs(tb.peak.item() - 2.0) < 1e-5
    assert abs(tb.k_decay.item() - 0.002) < 1e-7
    assert abs(tb.k_rise.item() - 0.01) < 1e-7


def test_constructor_rejects_bad_ordering():
    for kw in (dict(peak=-1.0), dict(k_rise=0.002, k_decay=0.01), dict(k_decay=-0.001)):
        try:
            TonicBaseline(**kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kw}")


def test_baseline_is_flat_b_before_injection():
    """1[t>=t_inj] gates the response: pre-injection the baseline is exactly b."""
    tb = TonicBaseline(b=0.5, peak=1.0, k_rise=0.01, k_decay=0.002)
    x = tb.baseline(T=2000, inj_t=1000)
    assert x.shape == (2000,)
    assert torch.allclose(x[:1000], torch.full((1000,), 0.5), atol=1e-6)
    assert x[1000].item() == 0.5  # tau=0 at injection: bump and drift both vanish


def test_bump_peaks_near_eq2_peak_time():
    """Drift off, the post-injection argmax sits at the Bateman peak time t_max."""
    dt = 1 / 25
    tb = TonicBaseline(b=0.0, peak=1.0, k_rise=0.01, k_decay=0.002, drift=0.0, dt=dt)
    inj_t = 500
    T = inj_t + 20000
    x = tb.baseline(T=T, inj_t=inj_t)
    post = x[inj_t:]
    peak_bin = post.argmax().item()
    t_max_hand = math.log(0.01 / 0.002) / (0.01 - 0.002)   # Bateman peak time, seconds
    # the curve is flat near its max, so the discrete argmax can sit a bin or
    # two from the continuous peak; a couple of bins is the right tolerance
    assert abs(peak_bin * dt - t_max_hand) < 2 * dt
    assert abs(tb.peak_time.item() - t_max_hand) < 1e-3    # property matches it
    assert post.max().item() > 0                           # A>0 -> positive bump


def test_drift_adds_linear_ramp_post_injection():
    """The drift term is a pure linear ramp; isolate it with the peak -> 0.

    The ramp is d*tau/3600 -- the drift is a rate PER HOUR. Per second it took a
    gradient step at the end of a recording 1800 times the size of its step at the
    start, which is what actually destabilised the fitted tonic.
    """
    dt = 1 / 25
    d = -2e-4
    tb = TonicBaseline(b=0.0, peak=1e-8, k_rise=0.01, k_decay=0.002, drift=d, dt=dt)
    inj_t = 100
    T = inj_t + 5000
    x = tb.baseline(T=T, inj_t=inj_t)
    # at k bins past injection, tau = k*dt, ramp = d*tau/3600 (bump ~ 0)
    for k in (1000, 3000, 4999 - inj_t):
        assert abs(x[inj_t + k].item() - d * k * dt / 3600.0) < 1e-5


def test_additive_mean_y_equals_xhat_plus_c():
    """y = x_hat + c: the tonic baseline broadcasts onto the phasic c [B, T]."""
    T, inj_t, batch = 1500, 750, 3
    tb = TonicBaseline(b=0.0, peak=1.0, k_rise=0.01, k_decay=0.002)
    h = bi_exponential_kernel(200, k_rise=2.0, k_decay=0.5)
    proc = PhasicEventProcess(h, beta0=-2.0, beta_inj=1.0)
    g = torch.Generator().manual_seed(0)
    out = proc.sample(T=T, inj_t=inj_t, batch=batch, generator=g)

    x_hat = tb.baseline(T=T, inj_t=inj_t)        # [T]
    y = x_hat + out["c"]                         # broadcast -> [batch, T]
    assert y.shape == (batch, T)
    assert torch.isfinite(y).all()
    # each row is exactly the shared baseline plus that row's phasic trace
    for b in range(batch):
        assert torch.allclose(y[b], x_hat + out["c"][b], atol=1e-6)


def test_baseline_is_differentiable_in_unconstrained_space():
    """Gradients flow to psi -- the node is a deterministic function of params."""
    tb = TonicBaseline(b=0.0, peak=1.0, k_rise=0.01, k_decay=0.002, drift=0.0)
    x = tb.baseline(T=1000, inj_t=200)
    x.sum().backward()
    for name in ("b", "alpha", "rho", "gamma", "drift"):
        g = getattr(tb, name).grad
        assert g is not None and torch.isfinite(g).all()


# --------------------------------------------------------- fit_tonic_baseline

def test_fit_tonic_baseline_recovers_known_params():
    """Pre-fit a noiseless Bateman trace; recovers the truth and a valid module."""
    dt = 1 / 25
    truth = TonicBaseline(b=0.2, peak=4.0, k_rise=0.05, k_decay=0.02, drift=-1e-4, dt=dt)
    T, inj_t = 60000, 30000
    y = truth.baseline(T, inj_t).detach()           # [T], no noise, no phasic

    fit = fit_tonic_baseline(y, inj_t, dt=dt, smooth_s=10.0)
    assert isinstance(fit, TonicBaseline)
    # constraints hold by construction
    assert fit.A.item() > 0
    assert fit.k_rise.item() > fit.k_decay.item() > 0
    # the fitted baseline tracks the truth tightly (relative L2 over the trace)
    x_fit = fit.baseline(T, inj_t).detach()
    rel = (x_fit - y).pow(2).sum().sqrt() / y.pow(2).sum().sqrt()
    assert rel.item() < 0.05


def test_fit_tonic_baseline_flat_trace_yields_modest_bump():
    """A near-flat (saline) trace must not manufacture a tall bump."""
    dt = 1 / 25
    T, inj_t = 20000, 10000
    y = torch.full((T,), 0.3)                        # perfectly flat at b=0.3
    fit = fit_tonic_baseline(y, inj_t, dt=dt, smooth_s=10.0)
    assert abs(fit.b.item() - 0.3) < 0.05
    # amplitude collapses to the lower bound; the baseline stays ~flat
    x_fit = fit.baseline(T, inj_t).detach()
    assert (x_fit.max() - x_fit.min()).item() < 0.05


def test_fit_tonic_baseline_respects_amplitude_cap():
    """A trace wanting a tall bump is held at peak_max, which now caps the curve.

    Under the old parameterization this capped the raw amplitude, which is not the
    height of anything: one real recording sat exactly on the cap while producing an
    entirely ordinary tonic. Capping the peak caps what the cap was always meant to.
    """
    dt = 1 / 25
    truth = TonicBaseline(b=0.0, peak=50.0, k_rise=0.05, k_decay=0.049, dt=dt)  # near-equal rates
    T, inj_t = 60000, 30000
    y = truth.baseline(T, inj_t).detach()
    fit = fit_tonic_baseline(y, inj_t, dt=dt, smooth_s=10.0, peak_max=10.0)
    assert fit.peak.item() <= 10.0 + 1e-3
    # The BUMP honours it, which is the property the cap exists for and the one the
    # old cap did not deliver. Isolated from the level and the drift, which are free
    # and absorb what the capped bump cannot reach -- the whole curve's swing is
    # larger than the cap for exactly that reason, and that is not the cap failing.
    t = torch.arange(T, dtype=torch.float32)
    tau = (t - inj_t).clamp_min(0) * dt
    post = (t >= inj_t).to(tau.dtype)
    bump = (fit.baseline(T, inj_t).detach() - fit.b.detach()
            - post * fit.drift.detach() * tau / 3600.0)
    assert bump.max().item() <= 10.0 + 1e-2
    # The derived amplitude may exceed the cap freely -- it is not a height.
    assert fit.A.item() > 10.0


def test_fit_tonic_baseline_accepts_batched_input():
    """[B, T] input is averaged over the batch (shared tonic) before fitting."""
    dt = 1 / 25
    truth = TonicBaseline(b=0.1, peak=3.0, k_rise=0.05, k_decay=0.02, dt=dt)
    T, inj_t = 60000, 30000
    x = truth.baseline(T, inj_t).detach()
    y = x.unsqueeze(0).repeat(4, 1)                  # [4, T], identical rows
    fit = fit_tonic_baseline(y, inj_t, dt=dt, smooth_s=10.0)
    x_fit = fit.baseline(T, inj_t).detach()
    rel = (x_fit - x).pow(2).sum().sqrt() / x.pow(2).sum().sqrt()
    assert rel.item() < 0.05


# ------------------------------------------------- numerical bounds on the curve
def test_a_runaway_amplitude_cannot_produce_nan():
    """exp(alpha) overflowing turns the injection bin into inf * 0 = NaN.

    At the injection bin tau is exactly zero, so the difference of exponentials is
    zero; if the amplitude has overflowed to inf, the product is NaN and the whole
    fit dies at the emission. Measured 2026-09-09: this killed 2 of 8 concurrent
    cohort fits before the bounds were added.
    """
    from photosvi.models.points import BatchedTonicBaseline

    b = BatchedTonicBaseline(n_seq=2, dt=1 / 25, peak=1.0, k_rise=0.01, k_decay=0.002)
    with torch.no_grad():
        b.alpha[0] = 500.0
    x = b.baseline(200, [50, 50])
    assert torch.isfinite(x).all()


def test_runaway_rate_constants_cannot_produce_nan():
    from photosvi.models.points import BatchedTonicBaseline

    b = BatchedTonicBaseline(n_seq=2, dt=1 / 25, peak=1.0, k_rise=0.01, k_decay=0.002)
    with torch.no_grad():
        b.rho[0], b.gamma[0] = 400.0, 400.0
        b.rho[1], b.gamma[1] = -400.0, -400.0
    x = b.baseline(200, [50, 50])
    assert torch.isfinite(x).all()


def test_the_bounds_are_the_identity_on_a_real_curve():
    """Clamping changes nothing a real fit does, so no existing result moves."""
    from photosvi.models.points import BatchedTonicBaseline

    b = BatchedTonicBaseline(n_seq=1, dt=1 / 25, b=-0.34, peak=7.0,
                             k_rise=0.005, k_decay=0.0009, drift=-0.0022)
    ref = TonicBaseline(dt=1 / 25, b=-0.34, peak=7.0, k_rise=0.005, k_decay=0.0009,
                        drift=-0.0022)
    torch.testing.assert_close(b.baseline(2000, [500])[0], ref.baseline(2000, 500),
                               rtol=1e-5, atol=1e-7)


def test_bateman_shape_is_normalised_and_stable():
    """Peak exactly 1, zero at the injection, and no precision loss as the rates close.

    The last is the whole reason for the factored `expm1` form. With a gap of 1e-6 the
    two exponentials agree to six digits, so the textbook difference is rounding error
    in float32 and the normalisation would divide noise by noise. Checked against the
    same curve evaluated in float64, where the naive form still has digits to spare.
    """
    import numpy as np
    import torch

    from photosvi.models.points.tonic import bateman_shape

    for k_decay, gap in ((0.002, 0.004), (0.0009, 0.0041), (0.05, 1e-6), (1e-5, 1e-6)):
        kd = torch.tensor(k_decay)
        kr = torch.tensor(k_decay + gap)
        t_peak = float(torch.log(kr / kd) / gap)
        tau = torch.linspace(0.0, max(4 * t_peak, 2400.0), 4000)
        sh = bateman_shape(tau, kd, kr)

        assert abs(float(bateman_shape(torch.tensor(t_peak), kd, kr)) - 1.0) < 1e-5
        assert float(sh[0]) == 0.0                       # nothing before the clock starts
        assert float(sh.max()) <= 1.0 + 1e-5             # the peak is the maximum
        assert torch.isfinite(sh).all()
        assert (sh >= -1e-6).all()                       # k_rise > k_decay: never negative

        # float64 reference, computed the naive way, where it is still accurate.
        t = tau.double().numpy()
        f = np.exp(-k_decay * t) - np.exp(-(k_decay + gap) * t)
        ref = f / f.max()
        assert np.abs(sh.double().numpy() - ref).max() < 2e-4, (k_decay, gap)


def test_peak_parameterisation_reproduces_the_old_curves():
    """The stored simulator peaks generate exactly the curves the amplitudes did.

    This is what licenses keeping every cohort already fitted. `CohortSpec.tonic_peak`
    was obtained by multiplying the amplitudes it replaced -- (1.2, 7.0, 6.0) -- by
    each drug's own Bateman peak factor, so the generated recordings are unchanged.
    If someone edits those constants without redoing that arithmetic, this fails.
    """
    import numpy as np

    from photosvi.experiments.cohort import CohortSpec
    from photosvi.models.points import TonicBaseline

    spec = CohortSpec()
    OLD_AMPLITUDES = (1.2, 7.0, 6.0)          # what tonic_peak replaced, 2026-09-17
    # The reference below divides the drift by 3600 as well: `tonic_drift` is now per
    # hour, and its stored value was multiplied by 3600 in the same commit, so the two
    # changes cancel and the curve is the one the old parameters produced.
    T, INJ, dt = 60000, 15000, 1 / 25

    for d, A in enumerate(OLD_AMPLITUDES):
        kd, kr = spec.tonic_k_decay[d], spec.tonic_k_rise[d]
        new = TonicBaseline(b=spec.tonic_b, peak=spec.tonic_peak[d], k_rise=kr,
                            k_decay=kd, drift=spec.tonic_drift, dt=dt)
        got = new.baseline(T, INJ).detach().double().numpy()

        # The old curve, written out longhand in float64 so it is a reference and not
        # a second call to the code under test.
        t = np.arange(T, dtype=float)
        tau = np.clip(t - INJ, 0.0, None) * dt
        post = (t >= INJ).astype(float)
        old = spec.tonic_b + post * (
            A * (np.exp(-kd * tau) - np.exp(-kr * tau))
            + spec.tonic_drift * tau / 3600.0)

        assert np.abs(got - old).max() < 1e-5, f"drug {d}: curves diverged"
        # And the peak really is the peak height, which is the point of the change.
        bump = got - spec.tonic_b - post * spec.tonic_drift * tau / 3600.0
        assert abs(bump.max() - spec.tonic_peak[d]) < 1e-4
