"""photosvi -- variational inference for a marked point process on photometry traces.

A fiber-photometry trace is modelled as two carriers on separated timescales under
one noise model: a sparse marked point process (the phasic dopamine transients,
each carrying a log-normal amplitude) convolved with a fixed sensor kernel, riding
on a slow pharmacokinetic baseline. Inference recovers where the events are and how
a drug shifts their rate.
"""

__version__ = "1.0.0"

from photosvi.models.points import SingleTraceModel

__all__ = ["SingleTraceModel", "__version__"]
