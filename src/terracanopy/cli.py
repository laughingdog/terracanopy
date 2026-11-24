"""Command line: `terracanopy analyze <photo|dir|video|rtsp://...>`.

This is the entry point that turns the library into a tool. The installed
`terracanopy` command is registered in pyproject.toml (`[project.scripts]`) and calls
`main()` below. It has two subcommands:

- `terracanopy indices` prints the available spectral indices and the bands each needs.
- `terracanopy analyze INPUT [options]` builds a pipeline out of the other modules and
  runs it: a source chosen from INPUT (`open_source`), one `VegetationAnalyzer` set up
  from the flags, and two sinks (a JSONL file and PNG overlays, in `--out`). It prints a
  one-line summary per frame while it works.

The module only wires things together and does argument parsing and printing. All the
actual analysis logic lives in the library modules, which is what makes it testable
without the command line.
"""

import argparse
import sys
from pathlib import Path

from terracanopy.analyzers import VegetationAnalyzer
from terracanopy.indices import INDICES
from terracanopy.pipeline import Pipeline
from terracanopy.sinks import JsonlSink, OverlaySink, Sink
from terracanopy.sources import open_source


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="terracanopy")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="run the vegetation analyzer on photos or video")
    a.add_argument("input", help="image file, directory of images, video file, camera index, or stream URL")
    a.add_argument("--index", choices=sorted(INDICES), help="index to use (default: best available)")
    a.add_argument("--threshold", type=float, help="canopy threshold on the index (default: per-frame Otsu)")
    a.add_argument("--min-blob", type=int, default=0, help="drop canopy blobs smaller than this many pixels")
    a.add_argument("--stride", type=int, default=1, help="video: analyze every Nth frame")
    a.add_argument("--max-frames", type=int, default=0, help="video: stop after N analyzed frames")
    a.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    a.add_argument("--no-overlay", action="store_true", help="skip writing overlay PNGs")

    sub.add_parser("indices", help="list available spectral indices")
    return p


def main(argv: list[str] | None = None) -> int:
    # argv=None makes argparse read the real command line. Tests pass a list instead (see
    # tests/test_core.py) so main() can be exercised without a subprocess. The return value is the
    # process exit code (0 = success), passed to sys.exit() at the bottom of the file.
    args = _build_parser().parse_args(argv)

    if args.command == "indices":
        for name, idx in INDICES.items():
            print(f"{name:6s} needs {', '.join(idx.requires):18s} {idx.description}")
        return 0

    # The sinks are created before the input is opened. JsonlSink empties results.jsonl on creation, so
    # a mistyped input path still wipes the previous run's results before the error is reported.
    sinks: list[Sink] = [JsonlSink(args.out / "results.jsonl")]
    if not args.no_overlay:
        sinks.append(OverlaySink(args.out / "overlays"))

    # --stride and --max-frames only affect video and streams. For images and folders they are
    # accepted but have no effect. --min-blob, --index and --threshold are passed straight to the
    # analyzer; leaving them out means "choose automatically", not zero.
    pipeline = Pipeline(
        source=open_source(args.input, stride=args.stride, max_frames=args.max_frames),
        analyzers=[VegetationAnalyzer(args.index, args.threshold, args.min_blob)],
        sinks=sinks,
    )
    n = 0
    # Iterating stream() is what actually drives the run: frames are read, analyzed and written to
    # the sinks one at a time as this loop pulls them. Nothing happens before this line.
    for results in pipeline.stream():
        for r in results:
            n += 1
            m = r.metrics
            print(
                f"[{r.frame.index}] {Path(r.frame.source).name} index={r.info['index']} "
                f"canopy={m['canopy_fraction']:.1%} "
                # .get with a NaN default: index_mean is absent when no pixel was classed as canopy.
                f"mean={m.get('index_mean', float('nan')):.3f} "
                f"({m['latency_ms']:.1f} ms)"
            )
    print(f"{n} frame(s) analyzed; results in {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
