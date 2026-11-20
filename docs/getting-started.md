# Getting started

## Install

```bash
pip install photosvi
```

Python 3.10 or newer. The heavy dependencies are PyTorch and Pyro; everything else is
the usual scientific stack. For a CUDA build of PyTorch, install it first as described
at [pytorch.org](https://pytorch.org/get-started/locally/); `pip` then keeps it. The
interactive benchmark views (`BenchmarkResult.pr_figure()`, `.dashboard()`) need
plotly:

```bash
pip install "photosvi[plots]"
```

To reproduce the paper, run the examples below as they are written, or run the test
suite, install from a clone:

```bash
git clone https://github.com/AbanteLab/photosvi.git
cd photosvi
pip install -e ".[dev]"
```

!!! note "GPU"
    Optional, and it only pays for the cohort model. A single-trace fit is about as
    fast on one CPU core as on a GPU, so many of them run best as a CPU fan-out; a
    cohort fit is *batched*, and there the per-step cost is nearly flat in the number
    of sessions. `photosvi.inference.points.resolve_device("auto", batch=S)` applies
    that rule. See [Choosing your settings](settings.md#cpu-or-gpu).

## Check the install

```bash
photosvi --help
```

From a clone, the test suite checks everything:

```bash
pytest tests/ -q
```

## A first fit

The example below is run by the test suite on every push. It
uses the toy configuration shipped with the package, `config_path("toy.yaml")`: two
minutes of recording instead of forty, so it finishes in seconds. Without a clone,
download [`single_trace.py`](https://github.com/AbanteLab/photosvi/blob/main/docs/examples/single_trace.py)
and run it with `python single_trace.py`.

```bash
python docs/examples/single_trace.py
```

```
rate ratio      4.31x   (simulated 4.00x)
beta_inj        +1.460 +- 0.134
expected events 42   (simulated 31)
```

!!! warning "These numbers are not the paper's"
    At two minutes a recording carries a few dozen transients, and the fit runs for
    600 steps rather than tens of thousands. The estimate is dominated by counting
    noise and the fit has not converged. Use the toy configuration to learn the API;
    the settings of the paper are in `configs/final_detection_48.yaml` (one recording)
    and `configs/final_cohort_48.yaml` (a cohort), also shipped with the package and
    found with `config_path`.

## What you need from your own data

You need three things:

1. **A trace**, as a `[1, T]` tensor of corrected dF/F, one recording.
2. **The injection bin**, an integer index into that trace.
3. **The sampling rate**, to set `dt`.

For a cohort, additionally the `animal` and `drug` index vectors, one entry per
recording. Preprocessing (isosbestic correction, bleaching, dF/F) happens upstream
of this package.
