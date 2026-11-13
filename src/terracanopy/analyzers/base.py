"""Analyzer plugin interface.

A domain (forest health, invasive species, water quality, wildlife) is just an
Analyzer: Frame in, AnalysisResult out. Analyzers must be stateless per frame
(or keep only small, explicit state) so the same code runs on one photo, a
tile grid, or a live stream.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from terracanopy.frame import Frame


@dataclass
class AnalysisResult:
    analyzer: str
    frame: Frame
    # Named per-pixel layers (H, W): e.g. "index", "mask". Used by sinks to render.
    layers: dict[str, np.ndarray] = field(default_factory=dict)
    # Scalar summary: JSON-serializable numbers.
    metrics: dict[str, float] = field(default_factory=dict)
    info: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class Analyzer(Protocol):
    name: str

    def analyze(self, frame: Frame) -> AnalysisResult: ...
