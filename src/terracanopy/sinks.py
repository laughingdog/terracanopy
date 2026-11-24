"""Result sinks: where analysis output goes.

A sink receives each `AnalysisResult` as the pipeline produces it (`write`) and is
told when the run ends (`close`). Two are provided:

- `JsonlSink`: one JSON object per line in a `.jsonl` file, holding the numbers
  (source, frame number, timestamp, metrics, info). It never contains pixels. This
  is the file to load into pandas/spreadsheets or to tail while a stream runs.
- `OverlaySink`: one PNG per frame showing where the canopy mask is, coloured by the
  index value, drawn over the original photo. This is the file to look at to check
  whether the analysis is sensible.

`render_overlay()` is the drawing function behind the second one, kept separate so
it can be used on its own (e.g. in a notebook) without writing files.

Sinks are independent of analyzers: they only read the generic fields of
`AnalysisResult`, so a new analyzer works with the existing sinks as long as it
fills in `metrics` and (for overlays) `layers["index"]` and `info["index"]`.
"""

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
        # Mode "w" truncates: the file is emptied as soon as the sink is constructed, before any
        # frame is processed. A failed run therefore still destroys the previous results.jsonl.
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
        # The assert only narrows the type for the checker; it fails if write() is called after close().
        assert self._fh is not None
        # json.dumps fails on numpy numbers, so metrics must already be plain floats. A NaN is written
        # as the bare token NaN, which Python reads back fine but strict JSON parsers reject.
        self._fh.write(json.dumps(record) + "\n")
        # Flush after every line so the file is complete up to the last frame at any moment: you can
        # `tail -f` it during a live stream, and a crash loses nothing already processed. It costs a
        # little speed per frame.
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def render_overlay(result: AnalysisResult) -> np.ndarray:
    """BGR uint8 image: index heatmap restricted to the mask, over RGB if available."""
    idx = result.layers["index"]
    mask = result.layers.get("mask")
    # The index is stretched to 0-255 using the same declared value_range as the analyzer's Otsu step,
    # so the colours are comparable between frames using the same index (blue = low, red = high), not
    # re-normalised per frame. That is why the index *name* must be in info["index"].
    lo, hi = INDICES[result.info["index"]].value_range
    scaled = (np.clip((idx - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)

    # If the frame has no RGB bands (e.g. a pure NIR/thermal sensor), there is no photo to draw over,
    # so the base is black instead.
    rgb = result.frame.rgb()
    base = (
        cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if rgb is not None
        else np.zeros_like(heat)
    )
    out = base.copy()
    if mask is not None:
        # Blend the whole image 40% photo / 60% heatmap, then copy blended pixels only where the mask
        # is True. Non-canopy pixels remain the plain photo, so the coloured areas show exactly what
        # the analyzer counted as canopy. All images here are BGR because cv2.imwrite expects BGR.
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
        # Filename = source name + zero-padded frame number + analyzer, e.g. photo_000000_vegetation.png.
        # The frame number keeps video frames from overwriting each other; zero-padding keeps them in
        # order when sorted. Re-running with the same output folder overwrites files of the same name.
        stem = Path(result.frame.source).stem or "frame"
        name = f"{stem}_{result.frame.index:06d}_{result.analyzer}.png"
        # cv2.imwrite returns False on failure (bad path, full disk) instead of raising, and the
        # return value is not checked here, so a failed overlay write is silent.
        cv2.imwrite(str(self.dir / name), render_overlay(result))

    def close(self) -> None:
        pass
