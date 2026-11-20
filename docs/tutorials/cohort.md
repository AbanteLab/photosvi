# Fit a cohort

```python title="docs/examples/cohort.py"
--8<-- "docs/examples/cohort.py"
```

## The design is read from the index vectors

You pass `animal` and `drug` as one entry per recording. Any assignment is allowed
(one drug per animal, every drug in every animal, or anything between), and the model
works out which variance components that design can identify:

| component | requires |
|---|---|
| `sigma_animal` | more than one animal |
| `sigma_cell` (animal × drug) | some animal that received **more than one drug** |
| `sigma_session` | a repeated (animal, drug) cell |
| `sigma_b0_session` | some animal recorded more than once |

`design_summary()` reports what it found, including `within_animal`: whether the drug
contrast is a within-animal comparison or a between-animal one. It determines
how precisely the cohort can estimate a contrast, so report it with the result.

!!! danger "Variance components the design cannot identify"
    A variance component the design cannot identify does not fail loudly: it
    **absorbs the component below it**, and the one it usually takes from is
    `sigma_animal`, which sets the width of the drug contrast. Requesting a level the
    design cannot support raises rather than warns, because the failure is otherwise
    silent.

## Report contrasts, not levels

`mu_drug[d]` on its own mixes the drug with the handling response: `beta_inj` is a
within-session pre/post contrast, and being picked up and injected raises the
transient rate on its own. Report `drug_contrast(opioid, saline)`, and for the
comparison between drugs `drug_contrast(opioid2, opioid1)`.

The levels are also the slowest thing to converge. If you write your own fitting loop,
watch the **opioid levels** for convergence rather than the contrast: two levels
climbing together leave the contrast looking settled while both are still far from
their final values, which biases every non-null effect toward zero.

## The tonic response comes out too

The tonic carrier is pooled by drug, so `tonic_posterior()` returns each drug's
amplitude, rate constants and peak time. Neither a per-session tonic fit nor a
two-stage analysis of per-session fits can produce these, since neither has a drug
level to pool over.

The tonic response to vehicle is small, so it is weakly identified and its peak time is
unreliable even when those of the opioids are recovered well.
