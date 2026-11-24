"""Spectral indices. Each declares the bands it needs, so the pipeline can pick
what's computable for whatever sensor produced the frame (phone RGB through
multispectral drone).

A spectral index turns a few bands into one number per pixel that tracks something
physical, e.g. how green or vigorous the vegetation is. Each one here is an `Index`
record: a name, the band names it `requires`, a function from Frame to an (H, W)
array, the `value_range` it normally falls in, and a description. All of them are
collected in the `INDICES` registry, which is what the CLI, the analyzer and the
overlay renderer look things up in.

The key idea is that availability depends on the Frame's bands, not on the index
code: `available_indices()` lists what a given frame supports, and
`best_vegetation_index()` picks the first supported index in `VEGETATION_PREFERENCE`
(NIR-based indices first, RGB-only ones as the fallback). To add an index, write a
function over a Frame and add one `Index(...)` entry to the list below.

Every index is safe on dark/empty pixels: divisions go through `safe_divide`, which
returns 0 instead of inf/NaN.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from terracanopy.frame import BLUE, GREEN, NIR, RED, RED_EDGE, Frame

EPS = 1e-6


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, returning 0 where |den| is ~0 instead of inf/nan."""
    # `out` starts as all zeros, and np.divide with `where=` only writes the pixels where the
    # condition is true. The skipped pixels therefore keep their initial 0. Note the consequence:
    # "undefined" (e.g. a pure black pixel, where red + nir == 0) is reported as exactly 0, which is
    # indistinguishable from a real index value of 0. Without `out=`, np.divide would leave those
    # pixels uninitialised (random garbage), so do not drop it.
    out = np.zeros_like(num, dtype=np.float32)
    np.divide(num, den, out=out, where=np.abs(den) > EPS)
    return out


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # (a - b) / (a + b): the shape shared by NDVI, NDRE, GNDVI and NDWI. Output is in [-1, 1] for
    # non-negative inputs. Argument order matters: swapping a and b flips the sign of the result.
    return safe_divide(a - b, a + b)


@dataclass(frozen=True)
class Index:
    """A named spectral index: which bands it needs and how to compute it from a Frame."""

    name: str
    requires: tuple[str, ...]
    fn: Callable[[Frame], np.ndarray]
    # Expected range, used for scaling/thresholding. This is not just documentation: Otsu thresholding
    # (analyzers/vegetation.py) and the heatmap overlay (sinks.py) both stretch this range onto 0-255 and
    # clip anything outside it. A range that is too wide wastes resolution (all values bunch into a few
    # levels); too narrow clips real data to the ends.
    value_range: tuple[float, float]
    description: str = ""


def _exg(f: Frame) -> np.ndarray:
    # Excess green on chromatic coordinates: 2g - r - b, with r+g+b == 1.
    # Each channel is first divided by r+g+b, which removes overall brightness so a shaded leaf and a
    # sunlit leaf of the same colour score alike. Result range is (-1, 2): pure green is 2, pure red or
    # blue is -1. Pixels with r+g+b ~ 0 (black) come out as 0 via safe_divide.
    r, g, b = f.band(RED), f.band(GREEN), f.band(BLUE)
    total = r + g + b
    return 2 * safe_divide(g, total) - safe_divide(r, total) - safe_divide(b, total)


def _vari(f: Frame) -> np.ndarray:
    # (g - r) / (g + r - b). The denominator can approach zero for non-vegetation pixels (e.g. g + r ~ b),
    # which makes the ratio blow up to huge values. safe_divide only guards |den| < EPS, so values are
    # still clipped to [-1, 1] here to keep them in the index's declared range.
    r, g, b = f.band(RED), f.band(GREEN), f.band(BLUE)
    return np.clip(safe_divide(g - r, g + r - b), -1.0, 1.0)


# The registry: name -> Index. Other modules look indices up here by name (CLI --index choices,
# analyzer validation, overlay scaling). Which index is "best" is not decided here but in
# VEGETATION_PREFERENCE below. Note ndwi is a water index, not a vegetation one, and is deliberately
# absent from that preference list.
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
# Order matters: the first entry whose bands are all present wins. NIR-based indices come first
# because near-infrared separates healthy vegetation from everything else far better than visible light;
# RGB-only ones are the fallback for ordinary photos. This is why the same command gives ExG on a phone
# photo but NDVI on a multispectral frame, and results from the two are not directly comparable.
VEGETATION_PREFERENCE = ("ndvi", "ndre", "gndvi", "exg", "vari")


def available_indices(frame: Frame) -> list[str]:
    return [name for name, idx in INDICES.items() if frame.has_bands(*idx.requires)]


def compute_index(frame: Frame, name: str) -> np.ndarray:
    try:
        idx = INDICES[name]
    except KeyError:
        raise KeyError(f"unknown index {name!r}; known: {sorted(INDICES)}") from None
    missing = [b for b in idx.requires if not frame.has_bands(b)]
    if missing:
        raise ValueError(f"index {name!r} needs bands {missing} not present in frame {frame.band_names}")
    # Force float32: a custom index function might return float64, which would double memory use and
    # make downstream dtype assumptions (OpenCV scaling, JSON output) inconsistent.
    return idx.fn(frame).astype(np.float32)


def best_vegetation_index(frame: Frame, preference: tuple[str, ...] | None = None) -> str:
    # `preference or ...` means an empty tuple also falls back to the default order, not "nothing".
    for name in preference or VEGETATION_PREFERENCE:
        if name in INDICES and frame.has_bands(*INDICES[name].requires):
            return name
    raise ValueError(f"no vegetation index computable from bands {frame.band_names}")
