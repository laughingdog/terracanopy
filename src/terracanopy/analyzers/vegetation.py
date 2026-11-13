"""Baseline vegetation analyzer: index + canopy mask + per-frame statistics.

This is deliberately a transparent, rule-based baseline. It reports what the
spectra say (greenness/vigor statistics over vegetated pixels) and does NOT
claim a diagnosis. Learned classifiers should be added as separate analyzers
once labeled data exists.
"""

import cv2
import numpy as np

from terracanopy.analyzers.base import AnalysisResult
from terracanopy.frame import Frame
from terracanopy.indices import INDICES, best_vegetation_index, compute_index


def _otsu_threshold(index: np.ndarray, value_range: tuple[float, float]) -> float:
    lo, hi = value_range
    scaled = np.clip((index - lo) / (hi - lo), 0.0, 1.0)
    u8 = (scaled * 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(lo + (t / 255.0) * (hi - lo))


class VegetationAnalyzer:
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
        idx_name = self.index or best_vegetation_index(frame)
        idx = compute_index(frame, idx_name)
        thr = self.threshold
        if thr is None:
            thr = _otsu_threshold(idx, INDICES[idx_name].value_range)

        mask = idx > thr
        if self.min_blob_px > 0:
            mask = self._remove_small_blobs(mask)

        canopy = idx[mask]
        metrics = {
            "threshold": float(thr),
            "canopy_fraction": float(mask.mean()),
        }
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
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= self.min_blob_px
        return keep[labels]
