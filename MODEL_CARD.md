# Frozen models

What each released version fits, and why each lever is set where it is. Every entry
is a measurement, not a preference; the run that produced it is named so it can be
re-checked.

## v1.0.0, the code of the paper

What changed from v0.2.0 below, lever by lever.

| lever | v0.2.0 said | what the code does now | evidence |
|---|---|---|---|
| kernel | K5 biphasic, "chosen by eye — the one lever not swept" | two gamma lobes, least-squares fit to the peak-triggered average of 298 isolated transients over the five animals of the third batch | `scripts/paper_calibration_numbers.py`; refitting on the current pipeline moves the half-width 0.500 s → 0.479 s, which is why the shipped parameters stand |
| emission | white | AR(2), φ = [1.432, −0.4464], σ_y the **marginal** sd | fitted per session on the pre-injection window; per-session φ₁ runs 0.71 to 1.66, so the shipped pair is the cohort's average colour and not any one animal's |
| marks | tied per window, gated divergence | unchanged, plus `mark_gate_floor` (API default 0.1 since 2026-10-15; 0.0 in the benchmark) and `q_a_scale` bounded at 5.0 | the gate leaves q(a)'s width unconstrained where the gate is near zero; on real recordings it ran to 30 and overflowed single precision |
| benchmark | `runs_benchmark_v3`, 50 cells, transient-to-noise 59 → 2.9 | `runs_benchmark_48`, 30 cells, 2.9 → 1.5 at 48 Hz | the old grid ran in a regime the recordings do not occupy; these animals sit at 4.4 |
| cohort study | `cohort_comparison.yaml`, three drugs, 8 animals, 25 Hz | `cohort_power_48.yaml`, two conditions, 48 Hz, scales measured on the recordings | and its slow response was wrong until 2026-10-05: `tonic_peak` was inherited from the old config at 3.9394 while `mu_a` and `sigma_y` were recalibrated, leaving the drug's slow component at 303 noise sds and 121× the transient amplitude where the recordings carry 16 and 9.5. Now 0.1962. `tests/test_config_tonic_scale.py` guards the proportion |

**Two statements under "Not defended" are now false and are corrected here rather than
edited above, because that entry is a record of what v0.2.0 was.** "The kernel was chosen by
eye; no sweep exists" — it is now fitted to data, though still not swept. "No result yet
touches a real recording" — the opioid cohort is fitted, both designs, and reported in the
paper.

**Still not defended, and new:** a cohort simulation is only as honest as its slow response.
The tonic and the injection-driven rate change both begin at the injection and both raise the
signal afterwards, so they are nearly confounded, and nothing in the pipeline checked their
relative scale until the proportion was found to be off by two orders of magnitude. The cell
simulated that way reported the estimator recovering 0.23 of a true 0.40 with none of twelve
cohorts covering it, which reads as a verdict on the method. What the corrected proportion
gives is being measured.

**Still not defended, and new:** the marks are identified only through their prior. The mean
depends on the product of an event indicator and a mark and a relaxed fit leaves both free,
so nothing in the likelihood separates them. This is invisible when the prior is the truth,
which is every simulation here, and it is not invisible on real recordings.

## v0.2.0 — 2026-09-10

### Single-trace (`SingleTraceModel`) — settled

| lever | value | evidence |
|---|---|---|
| kernel | K5 biphasic, peak 0.080 s, FWHM 0.080 s, undershoot −0.25 at 0.44 s | shortlist, chosen by eye — **the one lever not swept** |
| Σh | 0 by construction | with a non-negative kernel the phasic contributes to the signal mean and the carrier is pushed to the floor; F1@0.5 went 0.000 → 0.935 |
| event factor | analytic Bernoulli divergence | error across τ spans 0.343 vs 2.743 for relaxed densities (`runs_tau`) |
| marks | tied per window, gated divergence | per-bin marks dragged μ_a 1.767 → 0.368 and σ_a 0.212 → 0.877 |
| guide window | 10 bins (≈0.21 s at 48 Hz) | chosen 2026-10-07 for every analysis; 5 bins at 25 Hz was the original 0.2 s, which contains the 3-bin lobe with only 1.6% of true ISIs below 0.2 s. User-selectable (advanced) |
| **τ** | **0.2** | flat over [0.05, 0.5], degrades above, one divergence at 1.0 (`runs_tau`) |
| **prior** | **Normal, σ = 3 (β₀), 2 (β_inj)** | shrinkage 0.77%, matching the analytic 0.74%; and it removes a delta-method readback that reported 0.1525 where the truth was 0.1711 |
| **guide over β** | **full-rank 2×2** | 2.2× → **0.97×** against NUTS on the same recordings; corr(β₀, β_inj) = −0.88 (`runs_hmc_chains`) |
| lr | 0.01 | raised from 1e-3 once the mark drift was fixed |
| stopping | estimand plateau, tol 0.02, probe 3000 | the ELBO is unusable — it swings across orders of magnitude on relaxed samples |
| σ_y | estimated from the trace, then frozen | left free it inflates to absorb tonic misfit |
| log λ | clamped to [−20, 5] | the Normal prior removed the support constraint that did this implicitly |

**Validated against exact inference.** NUTS on the same model and recordings, nuisances
frozen in both arms: posterior width ratio 0.97×, R-hat < 1.05 on all five, n_eff
48–155. Point estimate +0.016 against the count MLE.

**Benchmark** (`runs_benchmark_v3`, 50 cells): F1 0.997 → 0.937 across
transient-to-noise 59 → 2.9, against GuPPy 0.989 → 0.739. Rate bias −0.012 → −0.163
over the same sweep.

### Cohort (`HierarchicalModel`)

**As fitted in the paper** (`configs/final_cohort_48.yaml`): one joint Gaussian over the
population terms, the non-centred effects, the per-condition tonic means and the
per-session tonic deviations. On simulated cohorts at medium SNR the width of the
interval on the contrast matches the spread of the estimate across cohorts (ratio 1.00).
The 90% interval contained the simulated effect in 80% of cohorts at a log-rate effect of
0.5, and zero in 88% of null cohorts; the shortfall comes from the downward bias of the
estimate (76% of the effect recovered), not from the width. The rest of this section and
the next record how the guide got there.

Same phasic configuration. Additionally: crossed animal × drug hierarchy,
non-centred, with the levels a design can identify switched on by inspection of the
index vectors; drug-structured tonic; per-session baseline term.

| | ratio | 90% coverage |
|---|---|---|
| mean-field | 3.33 | 50% |
| full-rank 4×4 over (μ_b0, μ_drug) | 1.88 | 55% |
| **full-rank 8×8, adding the variance components** | **1.67** | **70%** |

**Open at the time, since closed by the joint block above.** A joint block over the globals alone does not close it. What remains outside the block is 64 `z`
latents plus 1.4 M per-bin ones, all correlated with μ_drug, and a block over those
would scale with the cohort rather than being a fix. Until this is resolved, report
**calibrated thresholds**, not the posterior's own interval.

### Not defended

- The kernel is fitted to the peak-triggered average of isolated transients of the
  experimental recordings (`photosvi kernel calibrate`); no sweep over kernel shapes exists.
- Every accuracy figure is measured on simulations from this model's own generator. The
  real opioid cohort has no ground truth.

## The variational family over the cohort's globals

*History: superseded by the joint block described under Cohort above.*

Five were measured on the same 23 simulated null cohorts, identical data and seeds.
*Width* is the median reported posterior SD of the drug contrast, *spread* the SD of
the estimate across cohorts (averaged over the last six probes of each fit, not read
at the stopping step), and their ratio the factor by which the interval is too narrow.

| family | width | spread | ratio | coverage | RMSE | sigma_animal |
|---|---|---|---|---|---|---|
| factorised | 0.0271 | 0.0843 | 3.11 | 50% | 0.0843 | 0.229 |
| **block over the 8 globals** | 0.0546 | 0.0837 | 1.53 | 70% | 0.0838 | **0.265** |
| + animal effects (24) | 0.0489 | 0.0836 | 1.71 | 70% | 0.0837 | 0.163 |
| + all effects (72) | 0.1047 | 0.1018 | 0.97 | 83% | 0.1019 | 0.149 |
| **block + linear response** | 0.0706 | 0.0837 | **1.19** | **81%** | 0.0838 | **0.265** |

Simulated `sigma_animal` is 0.25; nominal coverage is 90%.

**Shipped: the block over the globals, with linear response when an interval is
quoted.** The last two columns are why the table carries more than the width. The
72-dimensional block reaches nominal coverage, and it does so by displacing variance
out of the hierarchy: the between-animal component falls to 0.149 and the estimator's
RMSE grows 18%. A wider interval around a worse estimate is not a better interval.

Linear response is the only variant that improves the interval while leaving the
estimator and the variance components untouched. It costs four warm-started refits
per estimand, corrects the one direction it is tilted along, and returns a
non-positive variance on about 1 cell in 10 at the shallow refit budget and 1 in 4 at
the deep one -- so the shallow budget is what should be used.

**Still short of nominal.** 81% against 90%. Not closed.
