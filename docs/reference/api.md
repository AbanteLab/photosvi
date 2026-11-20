# Python API

`photosvi.api` is the part of the package meant to be used from outside it: loading a
recording, fitting it and reading the result. Its interfaces stay stable within a major
version. The pages under Internals document the model and inference code behind it,
which can change between versions.

## Recordings

::: photosvi.api.Session

::: photosvi.api.Study

::: photosvi.api.load_doric_session

::: photosvi.api.load_session

## Fitting

::: photosvi.api.FitSettings

::: photosvi.api.fit_session

::: photosvi.api.fit_study

## Results

::: photosvi.api.Result

::: photosvi.api.events_from_posterior

## Before a fit

::: photosvi.api.calibrate_kernel

::: photosvi.api.KernelCalibration

::: photosvi.api.spectrum

::: photosvi.api.config_path
