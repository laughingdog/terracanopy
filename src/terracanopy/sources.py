"""Frame sources. All yield `Frame`s lazily, so a folder of photos and a live
RTSP stream look identical to the pipeline.

A source is anything you can iterate to get `Frame`s one at a time. Two are provided:

- `ImageSource`: a single image file, or every image in a folder (sorted by name).
- `VideoSource`: a video file, a webcam by number, or a stream URL such as rtsp://.
  `stride` skips frames and `max_frames` stops early.

`open_source()` is the front door used by the CLI: it looks at a string and decides
which of the two to build. Sources are generators, so nothing is read from disk or
the network until the pipeline starts iterating, and only one frame is held in
memory at a time. That matters for long videos and endless streams.

Every source currently produces 3-band 8-bit RGB frames, because the images are
decoded with OpenCV. Multispectral or georeferenced input (GeoTIFF etc.) would need
a new source that fills in the other canonical bands and `Frame.geo`; none exists yet.
"""

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
    # OpenCV decodes images and video as B, G, R, the reverse of what Frame expects. Skipping this swap
    # would not crash: red and blue would silently trade places, so every index that uses them
    # (ExG, VARI) would be computed on the wrong channels and return plausible-looking but wrong values.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Frame.from_rgb_uint8(rgb, source=source, index=index, timestamp=timestamp)


class ImageSource:
    """Reads in one image file or a directory of them (RGB; phone photos, drone stills)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # The file list is built here, once, but images are only decoded later during iteration.
        # sorted() makes the order (and so each frame's `index`) deterministic. Not recursive:
        # images in subfolders are ignored. Files are matched by extension only, not by content.
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
            # IMREAD_COLOR forces a 3-channel 8-bit image whatever the file holds: grayscale is
            # replicated into 3 channels, any alpha channel is dropped, and 16-bit images (common for
            # TIFFs) are reduced to 8-bit, losing precision. imread signals failure by returning
            # None rather than raising, hence the explicit check.
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(f"could not decode image {p}")
            # The timestamp is the file's modification time, not when the photo was taken, and copying
            # or re-saving a file changes it.
            yield _bgr_to_frame(bgr, source=str(p), index=i, timestamp=p.stat().st_mtime)


class VideoSource:
    """Video file, camera index, or stream URL (rtsp://, http://...).

    stride: analyze every Nth frame (cheap way to hit a latency budget).
    max_frames: stop after this many frames have been yielded; 0 means no limit.

    Every frame is still decoded even when skipped by `stride`, so stride saves analysis time,
    not decoding time. On a live stream, if analysis is slower than the frame rate, frames
    queue up inside OpenCV's buffer and the results fall behind real time.
    """

    def __init__(self, target: str | int | Path, stride: int = 1, max_frames: int = 0) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        # An int means "camera number" (0 = first webcam) to OpenCV; a str means a file path or URL.
        # The two are not interchangeable, so the type is preserved rather than stringified.
        self.target = target if isinstance(target, int) else str(target)
        self.stride = stride
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.target)
        # VideoCapture does not raise on a bad path or unreachable stream; it just reports not-open.
        if not cap.isOpened():
            raise OSError(f"could not open video source {self.target!r}")
        # try/finally guarantees the capture handle is released, including when the consumer stops
        # iterating early (e.g. breaks out of a loop), which closes the generator.
        try:
            # Some streams report an fps of 0 (unknown); `or 0.0` keeps that as a number, not None.
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            # n_read counts every frame decoded (including skipped ones); n_yield counts only those
            # handed on. Frame.index uses n_read, so it is the frame's position in the video
            # (0, 5, 10, ... with stride 5), not a count of analyzed frames. max_frames uses n_yield.
            n_read = n_yield = 0
            t0 = time.time()
            while True:
                ok, bgr = cap.read()
                # read() returns ok=False at end of file, and also if a live stream drops.
                if not ok:
                    break
                if n_read % self.stride == 0:
                    # For files, position within the video is added to the wall-clock start time, giving
                    # each frame a timestamp as if the video were played back starting now. Live streams
                    # often report a position of 0 or less, in which case the current time is used.
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
    # Order matters. Anything containing "://" is treated as a stream URL, and an all-digits string
    # is treated as a camera number, both before the filesystem is consulted. So a file or folder
    # literally named "0" or "2024" would be taken for a webcam; give such paths as "./0".
    # After that, the decision is by file extension only, not by looking inside the file.
    if "://" in target or target.isdigit():
        return VideoSource(int(target) if target.isdigit() else target, stride, max_frames)
    p = Path(target)
    if p.is_dir() or p.suffix.lower() in IMAGE_EXTS:
        return ImageSource(p)
    if p.suffix.lower() in VIDEO_EXTS:
        return VideoSource(p, stride, max_frames)
    raise ValueError(f"cannot tell how to open {target!r} (not an image, directory, video, or URL)")
