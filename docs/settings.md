# Choosing your settings

## The kernel

The phasic basis is a fixed impulse response, not a fitted one, and its **taps sum to
zero**. That is a modelling device rather than a claim about the sensor: with a
non-negative kernel the phasic component contributes to the signal mean, so it
competes with the tonic carrier for it, and the carrier is pushed onto the floor of
the trace. The rate coefficient absorbs the difference.

Set the shape from your sensor's transient, not from ours. What matters:

- **Time to peak and width** should match a single transient in your recordings. The
  default is K10, dLight1.3b as fitted to the paper's recordings (peak about 0.5 s, FWHM
  about 0.5 s), with 8 s of support at any sampling rate.
- **Effective support** should be well inside your typical inter-event interval. If
  events are closer together than the kernel is long, their lobes merge.
- The **window** of the event guide should contain the correlation lobe: wide enough
  that timing alternatives compete, narrow enough that two real events rarely share
  one. The default, 10 bins, is the window the paper uses with the default kernel at 48 Hz.

### Calibrating it from your recordings

`calibrate_kernel` does this from the data, before any fit. It takes the transients that
stand alone in your recordings, averages them aligned on their peaks, fits each kernel
family to that average through the same band-pass, and scores every candidate (the
fitted ones and the two shipped ones) on transients it was not fitted to:

```python
from photosvi.api import FitSettings, calibrate_kernel

cal = calibrate_kernel(sessions, sensor="GCaMP6f")   # sensor sets the starting values
cal.plot("kernel.pdf")                               # every candidate over the data
settings = FitSettings(**cal.settings())             # the suggested kernel
settings = FitSettings(**cal.settings("dLight1.3b-K10"))   # or any other, by key
```

or from the command line, which writes the `kernel:` block of a config:

```bash
photosvi kernel calibrate rec1.csv rec2.csv --fs 48 --sensor GCaMP6f \
    --out kernel.yaml --plot kernel.pdf
```

The suggestion is the lowest held-out error, and it is only a suggestion: a residual
measures how well a shape draws the average transient, not how well a fit with it
recovers the rate. Look at the figure. With fewer than twenty isolated transients
nothing is fitted and the shipped kernel is suggested; the band-pass at 10 Hz also
means a rise faster than about 0.1 s cannot be resolved.

`scripts/kernel_candidates.py` plots candidate bases with peak time, width, undershoot
and support, for choosing among hand-written ones.

## What an unconfigured fit uses

`FitSettings()` with nothing changed is the paper's single-trace model: the settings of
`configs/benchmark_guppy_48.yaml` (kernel K10, tau 0.2, window 10, AR(2) emission), with
the kernel support and the refractory window rescaled from 48 Hz to your sampling rate.
Four quantities are read off each recording, before the injection: the AR(2)
coefficients (Yule-Walker, the procedure behind the paper's coefficients; the paper's
analyses then used the cohort's average pair, which a cohort config can still set), the
noise scale (successive differences,
corrected for that correlation), and the centre of the mark prior (the median and spread
of the visible peaks). The mark prior has width 0.05 and the gated mark divergence a floor
of 0.1, both from `configs/cohort_power_48.yaml`. Any of these can be set explicitly;
`ar_coeffs=[]` fits white noise.

## When the tonic pre-fit fails

`fit_tonic_baseline` caps the amplitude, because when `k_rise` (the release rate constant)
approaches `k_decay` (the clearance rate constant) the difference of exponentials
collapses and only the product is identified, and an uncapped fit sends the amplitude to enormous values against a vanishing gap.

Symptoms of a bad tonic and what they mean:

- **A peak time near zero or near the end of the recording.** The curve has degenerated
  into a step or a ramp. Usually means the bump is small relative to the noise; a
  vehicle session legitimately looks like this.
- **The fit tracks the transients rather than the trend.** Increase `smooth_s`: the
  pre-fit targets a median-smoothed trace precisely so the slow carrier is not chasing
  fast structure.

## How far to trust the interval

The single-trace interval was checked against Hamiltonian Monte Carlo on the same model
and recordings: its width is 0.97 times that of the exact posterior. This needs the joint
Gaussian over the two rate coefficients, whose posterior correlation is about −0.9; a
factorized guide reported intervals about twice too narrow.

For the cohort model, fitted as in the paper (`configs/final_cohort_48.yaml`: one joint
Gaussian over the population terms, the non-centred effects and the tonic terms), the
width of the interval on a contrast matched the spread of the estimate across simulated
cohorts at medium SNR. The estimate is shrunk toward zero, by about a quarter at a
log-rate effect of 0.5, so the 90% interval contained the simulated effect in 80% of
cohorts, and zero in 88% of null cohorts.

Both measurements are on simulations from the model's own generator. When an interval
decides a result, simulate cohorts matched to your recordings and check the coverage
there, as the paper does.

## CPU or GPU

The per-step cost on a GPU is nearly flat in the number of sessions, because a step is
a long chain of small kernels and the card spends its time launching them rather than
computing. Measured at `T = 60000`, ms per step:

| sessions | 1 | 8 | 16 | 32 |
|---|---|---|---|---|
| GPU | 35 | 56 | 55 | 46 |
| CPU (1 thread) | 29 | ~210 | ~370 | ~890 |

So a **single-trace** fit is about as fast either way, and many of them run best as a
CPU fan-out: 24 at once beats one on a card. A **cohort** fit is batched, and there
the GPU is the only sensible choice: sixteen sessions cost what one trace costs.

`resolve_device("auto", batch=S)` applies exactly that rule. Asking for CUDA when
there is none raises rather than falling back, because a run launched for a GPU that
quietly took sixty times longer on CPU is worse than one that refuses to start.
