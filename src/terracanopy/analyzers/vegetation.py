"""Baseline vegetation analyzer: index + canopy mask + per-frame statistics.

This is deliberately a transparent, rule-based baseline. It reports what the
spectra say (greenness/vigor statistics over vegetated pixels) and does NOT
claim a diagnosis. Learned classifiers should be added as separate analyzers
once labeled data exists.

For each frame it does three things:

1. Computes a spectral index (see indices.py), either the one you name or the best
   one the frame's bands allow.
2. Splits pixels into "canopy" and "not canopy" by thresholding that index. The
   threshold is yours, or if omitted it is picked automatically for each frame with
   Otsu's method (the cut that best separates the histogram into two groups).
3. Summarises the canopy pixels: what fraction of the frame they cover, and the mean,
   median, 10th/90th percentile and spread of the index over them.

Two things to keep in mind when reading the numbers. The mask means "pixels greener
than the threshold", so yellow or brown (stressed or dead) crowns can fall outside
it and be undercounted as canopy. And because the automatic threshold is recomputed
per frame, `canopy_fraction` is only comparable across frames if the scenes are
similar, or if you pass a fixed `threshold`.
"""

import cv2
import numpy as np

from terracanopy.analyzers.base import AnalysisResult
from terracanopy.frame import Frame
from terracanopy.indices import INDICES, best_vegetation_index, compute_index


def _otsu_threshold(index: np.ndarray, value_range: tuple[float, float]) -> float:
    # OpenCV's Otsu only works on 8-bit images, so the float index is squeezed onto 0-255 using the
    # index's declared value_range (values outside it are clipped), thresholded, and the answer is
    # converted back to index units. Consequences: the threshold has only 256 possible levels across
    # the range, and a wrong value_range would skew it.
    lo, hi = value_range
    scaled = np.clip((index - lo) / (hi - lo), 0.0, 1.0)
    u8 = (scaled * 255).astype(np.uint8)
    # With THRESH_OTSU, OpenCV ignores the 0 we pass as the threshold and computes its own, returned as t.
    # Otsu always finds *some* split, even on a frame with no vegetation at all (it then splits noise)
    # or a frame that is all canopy. So an automatic canopy_fraction is never trustworthy on empty or
    # uniform scenes; pass an explicit threshold if that matters.
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(lo + (t / 255.0) * (hi - lo))


class VegetationAnalyzer:
    """Index -> canopy mask -> statistics, for one frame at a time (see the module docstring).

    Holds only its settings, no per-frame state, so one instance can safely be reused across
    a whole folder or an endless stream.
    """

    name = "vegetation"

    def __init__(
        self,
        index: str | None = None,
        threshold: float | None = None,
        min_blob_px: int = 0,
    ) -> None:
        """
        index: index name; defaults to the best one the frame's bands support.
        threshold: canopy cutoff on the index; defaults to a per-frame Otsu split.
        min_blob_px: drop connected canopy blobs smaller than this (noise removal).
        """
        if index is not None and index not in INDICES:
            raise KeyError(f"unknown index {index!r}; known: {sorted(INDICES)}")
        self.index = index
        self.threshold = threshold
        self.min_blob_px = min_blob_px

    def analyze(self, frame: Frame) -> AnalysisResult:
        # If no index was given, it is chosen per frame, so a folder mixing RGB and multispectral
        # images could be analysed with different indices. The index actually used is recorded
        # in the result's info["index"], and the overlay renderer relies on that to scale colours.
        idx_name = self.index or best_vegetation_index(frame)
        idx = compute_index(frame, idx_name)
        # `if thr is None`, not `if not thr`: an explicit threshold of 0.0 is valid and must be kept.
        thr = self.threshold
        if thr is None:
            thr = _otsu_threshold(idx, INDICES[idx_name].value_range)

        # Strictly greater than: pixels exactly at the threshold are not canopy.
        mask = idx > thr
        if self.min_blob_px > 0:
            mask = self._remove_small_blobs(mask)

        # Boolean indexing flattens the selected pixels into a 1-D array, so these statistics use
        # only canopy pixels and ignore all the rest. A high index_mean means the vegetation that
        # was found is vigorous, not that the scene is mostly vegetated (that is canopy_fraction).
        canopy = idx[mask]
        metrics = {
            "threshold": float(thr),
            "canopy_fraction": float(mask.mean()),
        }
        # If the mask is empty there are no pixels to summarise, so the index_* keys are omitted
        # entirely rather than set to NaN. Consumers must tolerate their absence (the CLI uses
        # m.get("index_mean", nan)); indexing m["index_mean"] directly will raise KeyError.
        if canopy.size:
            p10, p50, p90 = np.percentile(canopy, [10, 50, 90])
            metrics.update(
                index_mean=float(canopy.mean()),
                index_p10=float(p10),
                index_median=float(p50),
                index_p90=float(p90),
                index_std=float(canopy.std()),
            )
        return AnalysisResult(
            analyzer=self.name,
            frame=frame,
            layers={"index": idx, "mask": mask},
            metrics=metrics,
            info={"index": idx_name},
        )

    def _remove_small_blobs(self, mask: np.ndarray) -> np.ndarray:
        # Connected-component labelling: every group of touching canopy pixels gets an integer id in
        # `labels` (same shape as the mask), with `stats` holding each group's area. connectivity=8
        # means diagonal neighbours count as touching, so thin diagonal lines stay as one blob.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        # Id 0 is always the background (everything that was False), so it is skipped with [1:] and
        # stays False in `keep`. Then `keep[labels]` looks up, for every pixel, whether its blob
        # survived, producing a full-size boolean mask again without any Python loop over pixels.
        keep = np.zeros(n, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= self.min_blob_px
        return keep[labels]
