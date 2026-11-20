# Models

::: photosvi.models.points.SingleTraceModel

::: photosvi.models.points.HierarchicalModel

::: photosvi.models.points.PhasicEventProcess

::: photosvi.models.points.TonicBaseline

::: photosvi.models.points.BatchedTonicBaseline

::: photosvi.models.points.AdditiveEmission

## Event guides

The variational family over the events is pluggable: each block returns per-bin
logits and swapping one for another changes the family without touching the model.

::: photosvi.models.points.WindowedCategoricalEventGuide

::: photosvi.models.points.FreeEventGuide

::: photosvi.models.points.WarmFreeEventGuide

::: photosvi.models.points.MatchedFilterEventGuide

## Kernels and anchors

::: photosvi.models.points.build_kernel

::: photosvi.models.points.biphasic_kernel

::: photosvi.models.points.bi_exponential_kernel

::: photosvi.models.points.fit_tonic_baseline

::: photosvi.models.points.stack_tonic_baselines

::: photosvi.models.points.estimate_log_baseline_rate

::: photosvi.models.points.estimate_log_noise_scale

## Calibrating the kernel

::: photosvi.kernel_calibration.calibrate_kernel

::: photosvi.kernel_calibration.KernelCalibration

::: photosvi.kernel_calibration.extract_isolated_transients
