"""The one data type every source produces and every analyzer consumes."""

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
        self.band_names = tuple(self.band_names)
        if self.data.ndim != 3:
            raise ValueError(f"data must be (bands, H, W); got shape {self.data.shape}")
        if self.data.shape[0] != len(self.band_names):
            raise ValueError(
                f"{self.data.shape[0]} bands in data but {len(self.band_names)} band names"
            )
        if self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[1], self.data.shape[2]

    def has(self, *bands: str) -> bool:
        return all(b in self.band_names for b in bands)

    def band(self, name: str) -> np.ndarray:
        try:
            return self.data[self.band_names.index(name)]
        except ValueError:
            raise KeyError(f"band {name!r} not in frame (have {self.band_names})") from None

    def rgb(self) -> np.ndarray | None:
        """(H, W, 3) float32 RGB view for display, or None if bands are missing."""
        if not self.has(*RGB_BANDS):
            return None
        return np.stack([self.band(b) for b in RGB_BANDS], axis=-1)

    @classmethod
    def from_rgb_uint8(cls, rgb: np.ndarray, **kwargs: Any) -> Frame:
        """Build a frame from an (H, W, 3) uint8 RGB image."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) image; got {rgb.shape}")
        data = np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0)
        return cls(data=np.ascontiguousarray(data), band_names=RGB_BANDS, **kwargs)

    @classmethod
    def from_bands(cls, bands: dict[str, np.ndarray], **kwargs: Any) -> Frame:
        names = tuple(bands)
        data = np.stack([bands[n] for n in names], axis=0)
        return cls(data=data, band_names=names, **kwargs)
