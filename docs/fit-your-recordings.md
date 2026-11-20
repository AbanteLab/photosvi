# Fit your own recordings

One recording with one injection, fitted with the single-trace model of the paper. You
need three things:

- **The recording**: a Doric `.doric` file, or a trace that is already corrected dF/F
  (`.csv`, `.txt`, `.tsv`, `.npy` or `.npz`).
- **The injection time**, in seconds. A Doric file that records the injection as an event
  does not need it.
- **The end of the handling window**, in seconds. Handling moves the fibre and leaves
  artefacts at the moment the rate is meant to change, so this stretch is left out of the
  likelihood. The paper masked at least 120 s after each injection.

If your sensor is not dLight1.3b, calibrate the kernel first; see
[Choosing your settings](settings.md#calibrating-it-from-your-recordings).

## From the command line

```bash
# a Doric file, injection read from its events
photosvi fit session.doric --mask-min-s 120 --out run/

# a Doric file, injection times given
photosvi fit session.doric --inj-start 1502.6 --inj-end 1520.1 --out run/

# a dF/F trace in a text file, sampled at 48 Hz
photosvi fit trace.csv --fs 48 --inj-start 600 --inj-end 640 --mask-min-s 120 --out run/
```

A Doric file is corrected to dF/F with its isosbestic channel. Any other file must already
hold dF/F; `--column` says which column, if there are several. The command prints the rate
ratio with its 90% interval and writes four files to `--out`:

| file | contents |
|---|---|
| `result.json` | the rate ratio, the posterior of the rate coefficients and every setting used |
| `events.csv` | one row per detected transient: bin, time in seconds, posterior probability |
| `curves.npz` | the trace, the per-bin event probability and the fitted tonic carrier |
| `injection.json` | the injection and masked window used, and where the times came from |

`photosvi fit --help` lists the other options (device, threads, number of steps).

## From Python

```python
from photosvi.api import FitSettings, fit_session, load_doric_session, load_session

# a Doric file; extra keyword arguments are stored as the session's metadata
session, info = load_doric_session("session.doric", mask_min_s=120,
                                   animal="m1", drug="morphine")
# or a dF/F trace
session = load_session("trace.csv", fs=48, inj_s=600, animal="m1", drug="morphine")

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

## Several recordings

`fit_study` fits each session of a `Study` on its own, which gives one rate ratio per
recording. To compare conditions across animals, fit them jointly with the hierarchical
model instead, as in [Fit a cohort](tutorials/cohort.md). Fill in each session's
metadata (animal, drug) from the start: the cohort model reads its design from it.
