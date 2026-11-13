"""Spectral indices. Each declares the bands it needs, so the pipeline can pick
what's computable for whatever sensor produced the frame (phone RGB through
multispectral drone)."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from terracanopy.frame import BLUE, GREEN, NIR, RED, RED_EDGE, Frame

EPS = 1e-6


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, returning 0 where |den| is ~0 instead of inf/nan."""
    out = np.zeros_like(num, dtype=np.float32)
    np.divide(num, den, out=out, where=np.abs(den) > EPS)
    return out


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return safe_divide(a - b, a + b)


@dataclass(frozen=True)
class Index:
    name: str
    requires: tuple[str, ...]
    fn: Callable[[Frame], np.ndarray]
    value_range: tuple[float, float]  # expected range, used for scaling/thresholding
    description: str = ""


def _exg(f: Frame) -> np.ndarray:
    # Excess green on chromatic coordinates: 2g - r - b, with r+g+b == 1.
    r, g, b = f.band(RED), f.band(GREEN), f.band(BLUE)
    total = r + g + b
    return 2 * safe_divide(g, total) - safe_divide(r, total) - safe_divide(b, total)


def _vari(f: Frame) -> np.ndarray:
    r, g, b = f.band(RED), f.band(GREEN), f.band(BLUE)
    return np.clip(safe_divide(g - r, g + r - b), -1.0, 1.0)


INDICES: dict[str, Index] = {
    i.name: i
    for i in [
        Index("exg", (RED, GREEN, BLUE), _exg, (-1.0, 2.0), "Excess green (RGB only)"),
        Index("vari", (RED, GREEN, BLUE), _vari, (-1.0, 1.0), "Visible atmospherically resistant index (RGB only)"),
        Index(
            "ndvi",
            (NIR, RED),
            lambda f: normalized_difference(f.band(NIR), f.band(RED)),
            (-1.0, 1.0),
            "Normalized difference vegetation index",
        ),
        Index(
            "ndre",
            (NIR, RED_EDGE),
            lambda f: normalized_difference(f.band(NIR), f.band(RED_EDGE)),
            (-1.0, 1.0),
            "Normalized difference red edge (chlorophyll / early stress)",
        ),
        Index(
            "gndvi",
            (NIR, GREEN),
            lambda f: normalized_difference(f.band(NIR), f.band(GREEN)),
            (-1.0, 1.0),
            "Green NDVI (more sensitive to chlorophyll content)",
        ),
        Index(
            "ndwi",
            (GREEN, NIR),
            lambda f: normalized_difference(f.band(GREEN), f.band(NIR)),
            (-1.0, 1.0),
            "Normalized difference water index (open water)",
        ),
    ]
}

# When no index is requested, prefer the most informative one the sensor supports.
VEGETATION_PREFERENCE = ("ndvi", "ndre", "gndvi", "exg", "vari")


def available_indices(frame: Frame) -> list[str]:
    return [name for name, idx in INDICES.items() if frame.has(*idx.requires)]


def compute_index(frame: Frame, name: str) -> np.ndarray:
    try:
        idx = INDICES[name]
    except KeyError:
        raise KeyError(f"unknown index {name!r}; known: {sorted(INDICES)}") from None
    missing = [b for b in idx.requires if not frame.has(b)]
    if missing:
        raise ValueError(f"index {name!r} needs bands {missing} not present in frame {frame.band_names}")
    return idx.fn(frame).astype(np.float32)


def best_vegetation_index(frame: Frame, preference: tuple[str, ...] | None = None) -> str:
    for name in preference or VEGETATION_PREFERENCE:
        if name in INDICES and frame.has(*INDICES[name].requires):
            return name
    raise ValueError(f"no vegetation index computable from bands {frame.band_names}")
