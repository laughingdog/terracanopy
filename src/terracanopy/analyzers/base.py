"""Analyzer plugin interface.

A domain (forest health, invasive species, water quality, wildlife) is just an
Analyzer: Frame in, AnalysisResult out. Analyzers must be stateless per frame
(or keep only small, explicit state) so the same code runs on one photo, a
tile grid, or a live stream.

This module defines the two things every analyzer shares:

- `Analyzer`, the interface: a `name` and an `analyze(frame)` method. It is a
  Protocol, so an analyzer does not inherit from anything; any class with those two
  members qualifies. To add a new kind of analysis, write such a class and hand it to
  the Pipeline.
- `AnalysisResult`, the output: per-pixel `layers` (for visualisation), scalar
  `metrics` (for logging and trends), and free-text `info` (labels such as which
  index was used).

Keeping the output split this way is what lets sinks stay generic: the JSONL sink
only reads metrics/info, the overlay sink only reads layers.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from terracanopy.frame import Frame


@dataclass
class AnalysisResult:
    """What one analyzer produced for one frame."""

    analyzer: str
    frame: Frame
    # Named per-pixel layers (H, W): e.g. "index", "mask". Used by sinks to render.
    layers: dict[str, np.ndarray] = field(default_factory=dict)
    # Scalar summary: JSON-serializable numbers. They must be plain Python floats, not numpy scalars:
    # the JSONL sink uses json.dumps, which raises TypeError on numpy types. (Hence the float(...)
    # wrappers in analyzers.) The Pipeline also adds "latency_ms" here after the analyzer returns,
    # so analyzers should not rely on this dict being theirs alone.
    metrics: dict[str, float] = field(default_factory=dict)
    info: dict[str, str] = field(default_factory=dict)


# runtime_checkable allows isinstance(x, Analyzer), but that only checks that the attributes
# `name` and `analyze` exist. It does not check the method's signature or return type.
@runtime_checkable
class Analyzer(Protocol):
    name: str

    def analyze(self, frame: Frame) -> AnalysisResult: ...
