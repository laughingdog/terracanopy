"""terracanopy: band-aware, streaming analysis of aerial/phone/satellite imagery."""

from terracanopy.frame import Frame
from terracanopy.indices import INDICES, available_indices, compute_index
from terracanopy.pipeline import Pipeline

__all__ = ["Frame", "INDICES", "available_indices", "compute_index", "Pipeline"]
__version__ = "0.1.0"
