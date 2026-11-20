# Fit one recording

The whole example, runnable and executed in CI:

```python title="docs/examples/single_trace.py"
--8<-- "docs/examples/single_trace.py"
```

## What each step is doing

**The two anchors (step 2)** give the fit a sensible starting point. `mf_anchor` runs a matched filter and turns the pre-injection peak count
into a baseline log-rate; `fit_tonic_baseline` least-squares-fits the Bateman curve to
a median-smoothed version of the trace. Neither is the final estimate, since SVI refines
both, but a hard-coded starting value would sooner or later fall outside the range of a
real recording.

**The emission scale is frozen (step 3).** `sigma_y` is estimated from the trace's
high-frequency content and then held fixed. Left free it inflates to absorb tonic
misfit, which makes the fit look better and the events disappear.

**The event guide** is the variational family over the events. The windowed
categorical block makes timing alternatives inside a short window *compete* rather
than accumulate, so the guide's total mass reads as an event count. With an
independent per-bin factor it does not: one transient produces a correlation lobe
several bins wide and the guide answers with fractional mass at every bin in it.

## Reading the output

`beta_posterior()` returns `(value, scale)` for each coefficient on the log-rate axis.
Report $e^{\beta_{\mathrm{inj}}}$ as the rate ratio.

`event_posterior(y)` returns per-bin $P(\text{event})$, length `T`. To turn it into
discrete calls, apply non-maximum suppression (`photosvi.benchmark.nms`) rather than
thresholding directly, which counts every bin of a transient's lobe.

!!! note "How far to trust the interval"
    The `scale` comes from a joint Gaussian over the two coefficients. On simulated
    recordings its width is 0.97 times that of the exact posterior, computed by
    Hamiltonian Monte Carlo. At low SNR the estimate is biased toward zero. See
    [Choosing your settings](../settings.md#how-far-to-trust-the-interval).
