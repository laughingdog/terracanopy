"""Pipeline: source -> analyzers -> sinks, with per-frame latency tracking.

Latency is measured from the start because "real time" is a requirement of
this project, not an afterthought.

The `Pipeline` is the glue between the three kinds of component. You give it a
source (anything that yields Frames), one or more analyzers, and zero or more
sinks. For each frame it runs every analyzer, stamps the result with how long that
took, and passes each result to every sink. It knows nothing about vegetation or
files; swapping the source, adding an analyzer, or adding a sink needs no change
here.

There are two ways to run it. `stream()` is a generator that yields each frame's
results as soon as they are ready, which is the right choice for videos and live
streams because memory stays flat. `run()` collects everything into a list, which
is convenient for small jobs and tests but keeps every result (including full-size
pixel layers) in memory at once.
"""

import time
from collections.abc import Iterable, Iterator, Sequence

from terracanopy.analyzers.base import AnalysisResult, Analyzer
from terracanopy.frame import Frame
from terracanopy.sinks import Sink


class Pipeline:
    def __init__(
        self,
        source: Iterable[Frame],
        analyzers: Sequence[Analyzer],
        sinks: Sequence[Sink] = (),
    ) -> None:
        if not analyzers:
            raise ValueError("pipeline needs at least one analyzer")
        self.source = source
        self.analyzers = list(analyzers)
        self.sinks = list(sinks)

    def stream(self) -> Iterator[list[AnalysisResult]]:
        """Yield the results for each frame as soon as it's processed."""
        # This is a generator, so nothing runs (no frame is read, no file written) until the caller
        # starts iterating over it. Just calling stream() does no work.
        try:
            for frame in self.source:
                # perf_counter is a monotonic high-resolution clock, correct for measuring durations
                # (unlike time.time(), which can jump if the system clock is adjusted).
                t0 = time.perf_counter()
                results = [a.analyze(frame) for a in self.analyzers]
                latency_ms = (time.perf_counter() - t0) * 1000.0
                # The timer covers analysis only. It excludes decoding the frame (the source) and
                # writing outputs (the sinks), so end-to-end delay on a stream is longer than this.
                # With several analyzers, every result gets the same combined time, not its own.
                for r in results:
                    # The metric is added before the sinks write, which is why it appears in the
                    # JSONL output. Moving this after the sink loop would drop it from the files.
                    r.metrics["latency_ms"] = latency_ms
                    for s in self.sinks:
                        s.write(r)
                yield results
        # `finally` closes the sinks when iteration ends normally, on an error, or when the caller
        # stops early. Closing happens when the generator is closed or garbage-collected, so after
        # an early `break` the files may not be closed until then.
        finally:
            for s in self.sinks:
                s.close()

    def run(self) -> list[list[AnalysisResult]]:
        return list(self.stream())
