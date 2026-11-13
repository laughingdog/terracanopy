"""Result sinks: where analysis output goes."""

import json
from pathlib import Path
from typing import IO, Protocol

import cv2
import numpy as np

from terracanopy.analyzers.base import AnalysisResult
from terracanopy.indices import INDICES


class Sink(Protocol):
    def write(self, result: AnalysisResult) -> None: ...
    def close(self) -> None: ...


class JsonlSink:
    """One JSON line per frame/analyzer. Easy to tail, load in pandas, or POST."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = self.path.open("w")

    def write(self, result: AnalysisResult) -> None:
        f = result.frame
        record = {
            "analyzer": result.analyzer,
            "source": f.source,
            "frame": f.index,
            "timestamp": f.timestamp,
            "info": result.info,
            "metrics": result.metrics,
        }
        assert self._fh is not None
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def render_overlay(result: AnalysisResult) -> np.ndarray:
    """BGR uint8 image: index heatmap restricted to the mask, over RGB if available."""
    idx = result.layers["index"]
    mask = result.layers.get("mask")
    lo, hi = INDICES[result.info["index"]].value_range
    scaled = (np.clip((idx - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)

    rgb = result.frame.rgb()
    base = (
        cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if rgb is not None
        else np.zeros_like(heat)
    )
    out = base.copy()
    if mask is not None:
        blended = cv2.addWeighted(base, 0.4, heat, 0.6, 0)
        out[mask] = blended[mask]
    else:
        out = cv2.addWeighted(base, 0.4, heat, 0.6, 0)
    return out


class OverlaySink:
    """Writes PNG overlays (one per frame)."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, result: AnalysisResult) -> None:
        stem = Path(result.frame.source).stem or "frame"
        name = f"{stem}_{result.frame.index:06d}_{result.analyzer}.png"
        cv2.imwrite(str(self.dir / name), render_overlay(result))

    def close(self) -> None:
        pass
