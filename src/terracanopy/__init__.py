"""terracanopy: band-aware, streaming analysis of aerial/phone/satellite imagery.

The package is a small pipeline: sources produce `Frame`s, analyzers turn each
Frame into an `AnalysisResult`, and sinks write the results out (see pipeline.py
for how they connect). This file only re-exports the names most users need, so
`from terracanopy import Frame, Pipeline` works without knowing which module each
lives in. The analyzers, sources and sinks are imported from their own modules.
"""

from terracanopy.frame import Frame
from terracanopy.indices import INDICES, available_indices, compute_index
from terracanopy.pipeline import Pipeline

__all__ = ["Frame", "INDICES", "available_indices", "compute_index", "Pipeline"]
__version__ = "0.1.0"
