"""The one data type every source produces and every analyzer consumes.

Everything upstream (phone photos, video frames, satellite tiles, multispectral
drone imagery) is converted into a `Frame`: a stack of named spectral bands as a
single float32 array of shape (bands, H, W), plus bookkeeping (where it came from,
frame number, timestamp, optional georeferencing). Everything downstream
(indices, analyzers, sinks) only ever sees `Frame`, so none of it needs to know
whether the pixels came from a JPEG or a 5-band sensor.

The band names are the contract between sources and indices. A source maps its
native channels onto the canonical names below, and each spectral index declares
which of those names it needs (see indices.py). That is how the same code can run
ExG on a phone photo and NDVI on a multispectral frame.

This module deliberately depends only on numpy (no OpenCV, no rasterio) so the core
data type stays importable anywhere.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Canonical band names. Sources map their native bands onto these.
RED = "red"
GREEN = "green"
BLUE = "blue"
RED_EDGE = "red_edge"
NIR = "nir"
SWIR = "swir"
THERMAL = "thermal"

RGB_BANDS = (RED, GREEN, BLUE)


@dataclass
class Frame:
    """A (bands, H, W) float32 array plus metadata.

    Values are expected in [0, 1] (reflectance, or 8-bit RGB / 255). Thermal
    bands may hold other units; analyzers that use them document their own
    assumptions.

    `data[i]` is the band named `band_names[i]`; the two are validated to be the
    same length when the Frame is created. The Frame is a plain mutable
    dataclass: it does not copy `data`, so two Frames built from the same array
    share pixels, and in-place edits to `data` are visible everywhere.

    `index` is the frame's position in its source (file number in a folder, raw
    frame number in a video), and `timestamp` is epoch seconds when known. Both
    exist so results can be matched back to the input they came from.
    """

    data: np.ndarray
    band_names: tuple[str, ...]
    source: str = ""
    index: int = 0
    timestamp: float | None = None
    # Optional georeferencing: affine transform + CRS as plain data, so the
    # core does not depend on rasterio.
    geo: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.band_names = tuple(self.band_names) # validate input is actually a tuple; noop if it already is
        if self.data.ndim != 3:
            raise ValueError(f"data must be (bands, H, W); got shape {self.data.shape}")

        if self.data.shape[0] != len(self.band_names):
            raise ValueError(
                f"{self.data.shape[0]} bands in data but {len(self.band_names)} band names"
            )

        # Converts the dtype only; it does NOT rescale. A uint8 array (0-255) passed in directly
        # becomes float32 values of 0-255, silently breaking the [0, 1] assumption that
        # thresholds and index ranges rely on. Use from_rgb_uint8() for 8-bit images.
        # astype() also makes a copy, so after this the Frame no longer shares memory with the caller's array.
        if self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)


    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[1], self.data.shape[2]


    def has_bands(self, *bands: str) -> bool:
        """True if every named band is present. With no arguments it is vacuously True."""
        return all(b in self.band_names for b in bands)


    def band(self, name: str) -> np.ndarray:
        """The (H, W) array for one band. Raises KeyError if the band is not in this frame."""
        try:
            # Indexing the first axis returns a view, not a copy: writing into the result changes the
            # Frame's own data (and any other Frame sharing the same array). Copy it first if you need
            # to modify it.
            return self.data[self.band_names.index(name)]
        # tuple.index raises ValueError when the name is missing; it is translated to KeyError (the
        # error callers expect from a lookup by name), and `from None` hides the confusing ValueError.
        except ValueError:
            raise KeyError(f"band {name!r} not in frame (have {self.band_names})") from None


    def rgb(self) -> np.ndarray | None:
        """(H, W, 3) float32 RGB view for display, or None if bands are missing."""
        if not self.has_bands(*RGB_BANDS):
            return None

        # Unlike a single-band lookup, np.stack builds a new array, so this is a copy.
        # Editing the result does not change the Frame.
        return np.stack([self.band(b) for b in RGB_BANDS], axis=-1)


    @classmethod
    def from_rgb_uint8(cls, rgb: np.ndarray, **kwargs: Any) -> Frame:
        """Build a frame from an (H, W, 3) uint8 RGB image."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) image; got {rgb.shape}")

        # Image libraries store (H, W, channels); Frame wants (bands, H, W), so move the channel axis
        # to the front. moveaxis returns a view with scrambled strides, not a rearranged copy, so
        # ascontiguousarray makes the memory layout match the new shape. Without it, per-band slices
        # are strided views and OpenCV functions that need contiguous input can fail or copy.
        # The channel order is assumed to be R, G, B here; callers holding OpenCV's BGR must convert first.
        data = np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0)
        return cls(data=np.ascontiguousarray(data), band_names=RGB_BANDS, **kwargs)


    @classmethod
    def from_bands(cls, bands: dict[str, np.ndarray], **kwargs: Any) -> Frame:
        # Band order in the Frame is the dict's insertion order. All arrays must have the same
        # (H, W); np.stack raises if they don't. Values are not rescaled (see __post_init__).
        names = tuple(bands)
        data = np.stack([bands[n] for n in names], axis=0)
        return cls(data=data, band_names=names, **kwargs)
