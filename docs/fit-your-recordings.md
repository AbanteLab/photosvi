# Fit your own recordings

Three things can be done from a file, either from the command line or from Python:

- fit **one recording** with the single-trace model, which gives the factor by which the
  injection changed the transient rate in that session;
- fit **a cohort** jointly with the hierarchical model, which gives the difference
  between conditions with between-animal variability in its interval;
- **calibrate the kernel** to your sensor, before either fit.

Each recording is a Doric `.doric` file, or a trace that is already corrected dF/F
(`.csv`, `.txt`, `.tsv`, `.npy` or `.npz`). A Doric file is corrected to dF/F with its
isosbestic channel, and can take its injection time from its events. For a trace file
you give the sampling rate and the injection time in seconds.

Handling the animal at the injection moves the fibre and leaves artefacts at the moment
the rate is meant to change, so a handling window after each injection is left out of
the likelihood: from the injection to `inj_end`, and at least `--mask-min-s` seconds. The
paper used 120 s.

## Try it on toy data first

These two scripts simulate recordings, write them to `photosvi_toy/` as your own files
would be, and fit them with the functions below. Each runs in under a minute.

```python title="docs/examples/toy_fit_session.py"
--8<-- "docs/examples/toy_fit_session.py"
```

```python title="docs/examples/toy_fit_cohort.py"
--8<-- "docs/examples/toy_fit_cohort.py"
```

The same files can then be fitted from the command line, with as few steps:

```bash
photosvi fit photosvi_toy/session.csv --fs 48 --inj-start 120 --steps 800 --out run/
photosvi fit-cohort photosvi_toy/sessions.csv --reference vehicle \
    --min-steps 200 --max-steps 300 --out run_cohort/
```

## Calibrating the kernel

The kernel is the shape of one transient. The shipped one is dLight1.3b in the nucleus
accumbens; with another sensor or region, calibrate it from your recordings first. The
procedure is described in [Choosing your settings](settings.md#calibrating-it-from-your-recordings).

```bash
photosvi kernel calibrate rec1.doric rec2.doric rec3.doric --mask-min-s 120 \
    --out kernel.yaml --plot kernel.pdf
photosvi kernel calibrate rec1.csv rec2.csv --fs 48 --sensor GCaMP6f --out kernel.yaml
photosvi kernel calibrate --sessions sessions.csv --mask-min-s 120 --out kernel.yaml
```

The command prints every candidate kernel with its error on transients it was not fitted
to, writes the suggested one to `kernel.yaml` and draws all of them over the average
transient in `kernel.pdf`. `--choose` writes another candidate instead. `--sessions`
takes a table of sessions (see below), which gives each file its own injection times;
use it for Doric files that do not record the injection as an event.

Pass the file to either fit with `--kernel kernel.yaml`. From Python:

```python
from photosvi.api import FitSettings, calibrate_kernel

cal = calibrate_kernel(sessions, sensor="GCaMP6f")
cal.plot("kernel.pdf")
settings = FitSettings(**cal.settings())          # the suggested kernel
```

## One recording

```bash
# a Doric file, injection read from its events
photosvi fit session.doric --mask-min-s 120 --out run/

# a Doric file, injection times given
photosvi fit session.doric --inj-start 1502.6 --inj-end 1520.1 --out run/

# a dF/F trace in a text file, sampled at 48 Hz, with a calibrated kernel
photosvi fit trace.csv --fs 48 --inj-start 600 --inj-end 640 --mask-min-s 120 \
    --kernel kernel.yaml --out run/
```

`--column` says which column holds the trace, if the file has several. The command prints
the rate ratio with its 90% interval and writes four files to `--out`:

| file | contents |
|---|---|
| `result.json` | the rate ratio, the posterior of the rate coefficients and every setting used |
| `events.csv` | one row per detected transient: bin, time in seconds, posterior probability |
| `curves.npz` | the trace, the per-bin event probability and the fitted tonic carrier |
| `injection.json` | the injection and masked window used, and where the times came from |

`photosvi fit --help` lists the other options (device, threads, number of steps).

From Python:

```python
from photosvi.api import FitSettings, fit_session, load_doric_session, load_session

# a Doric file; extra keyword arguments are stored as the session's metadata
session, info = load_doric_session("session.doric", mask_min_s=120,
                                   animal="m1", condition="drug")
# or a dF/F trace
session = load_session("trace.csv", fs=48, inj_s=600, animal="m1", condition="drug")

result = fit_session(session, FitSettings(), on_progress=print)

result.rate_ratio        # {'estimate', 'lo', 'hi', 'level'}: e^beta_inj, 90% interval
result.events()          # {'bins', 'times_s', 'prob'} of the detected transients
result.rate_per_min()    # detections per minute before and after the injection
result.save("run/")      # the same files as the command line
```

`FitSettings()` with no arguments is the paper's model; every field left at `None` takes
the shipped value. A forty-minute recording takes about ten minutes on four CPU threads.

`result.rate_ratio` is the estimate to report. It is a parameter of the model, so it
does not depend on the detection threshold. `events()` and `rate_per_min()` are readouts
of the per-bin posterior and do: `result.events(threshold=0.7)` re-thresholds without
refitting.

## A cohort

A cohort is described by a table of sessions, one row per recording:

```text
path,animal,condition,inj_start,inj_end
m1_vehicle.doric,m1,vehicle,,
m1_drug.doric,m1,drug,,
m2_vehicle.doric,m2,vehicle,1502.6,1520.1
m2_drug.csv,m2,drug,600,640
```

`path` is read relative to the table. `animal` and `condition` give the design: the model
reads from them which animals were recorded under which conditions, and so which
variance components it can estimate. `inj_start` and `inj_end` are in seconds; an empty
cell on a Doric file means the file's injection event. A trace file also needs `fs`, in
the table or with `--fs`. Other columns are kept as metadata.

```bash
photosvi fit-cohort sessions.csv --reference vehicle --mask-min-s 120 --out run/
```

The command prints, for each condition, the factor by which its transient rate after the
injection differs from that under `--reference`, with its 90% interval, and writes:

| file | contents |
|---|---|
| `result.json` | the contrasts, the tonic response of each condition, the variance components, the design and every setting used |
| `curves.npz` | the fitted traces, the per-bin event probability, each session's tonic curve and each condition's pooled one |
| `sessions.json` | the injection and masked window of each session, and where the times came from |

All sessions must have the same sampling rate. Each is cut to the same window: 10 minutes
before its injection (`--pre-s`) and as much after it as every session has. The window
should not reach back to an earlier injection. The paper's cohort, 13 forty-minute
recordings, takes about half an hour on a GPU.

From Python:

```python
from photosvi.api import CohortSettings, Study, fit_cohort, read_manifest

sessions, _ = read_manifest("sessions.csv", mask_min_s=120)
# or build them one by one with load_doric_session / load_session, giving each
# animal= and condition= as metadata

result = fit_cohort(Study(sessions), CohortSettings(), reference="vehicle")

result.contrast("drug")              # drug - vehicle: estimate, lo, hi and as a ratio
result.contrast("drug_b", "drug")    # any two conditions
result.tonic_ratio("drug")           # ratio of the tonic peak heights, 90% interval
result.tonic                         # each condition's pooled tonic curve parameters
result.variance_components           # what a power calculation for the next cohort needs
result.design                        # within- or between-animal, levels estimated
result.save("run/")
```

`CohortSettings()` with no arguments is the paper's real-cohort fit. On the paper's 13
recordings it reproduces the reported fit exactly: a rate ratio of 1.57 [1.13, 2.17] for
RO64-6198 against vehicle, and a tonic height ratio of 0.41 [0.40, 0.43].

To fit each recording on its own instead, `fit_study` returns one single-trace result per
session; that is not the hierarchical model and its rate ratios are per session.
