"""Frame sources. All yield `Frame`s lazily, so a folder of photos and a live
RTSP stream look identical to the pipeline."""

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import cv2

from terracanopy.frame import Frame

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


class FrameSource(Protocol):
    def __iter__(self) -> Iterator[Frame]: ...


def _bgr_to_frame(bgr, source: str, index: int, timestamp: float | None = None) -> Frame:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Frame.from_rgb_uint8(rgb, source=source, index=index, timestamp=timestamp)


class ImageSource:
    """One image file or a directory of them (RGB; phone photos, drone stills)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_dir():
            self.files = sorted(p for p in self.path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        elif self.path.is_file():
            self.files = [self.path]
        else:
            raise FileNotFoundError(self.path)
        if not self.files:
            raise FileNotFoundError(f"no images found in {self.path}")

    def __iter__(self) -> Iterator[Frame]:
        for i, p in enumerate(self.files):
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(f"could not decode image {p}")
            yield _bgr_to_frame(bgr, source=str(p), index=i, timestamp=p.stat().st_mtime)


class VideoSource:
    """Video file, camera index, or stream URL (rtsp://, http://...).

    stride: analyze every Nth frame (cheap way to hit a latency budget).
    """

    def __init__(self, target: str | int | Path, stride: int = 1, max_frames: int = 0) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.target = target if isinstance(target, int) else str(target)
        self.stride = stride
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.target)
        if not cap.isOpened():
            raise OSError(f"could not open video source {self.target!r}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n_read = n_yield = 0
            t0 = time.time()
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if n_read % self.stride == 0:
                    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    ts = t0 + pos_ms / 1000.0 if pos_ms and pos_ms > 0 else time.time()
                    frame = _bgr_to_frame(bgr, source=str(self.target), index=n_read, timestamp=ts)
                    frame.meta["fps"] = fps
                    yield frame
                    n_yield += 1
                    if self.max_frames and n_yield >= self.max_frames:
                        break
                n_read += 1
        finally:
            cap.release()


def open_source(target: str, stride: int = 1, max_frames: int = 0) -> FrameSource:
    """Pick a source from a path/URL string."""
    if "://" in target or target.isdigit():
        return VideoSource(int(target) if target.isdigit() else target, stride, max_frames)
    p = Path(target)
    if p.is_dir() or p.suffix.lower() in IMAGE_EXTS:
        return ImageSource(p)
    if p.suffix.lower() in VIDEO_EXTS:
        return VideoSource(p, stride, max_frames)
    raise ValueError(f"cannot tell how to open {target!r} (not an image, directory, video, or URL)")
