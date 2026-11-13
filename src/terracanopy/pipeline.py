"""Pipeline: source -> analyzers -> sinks, with per-frame latency tracking.

Latency is measured from the start because "real time" is a requirement of
this project, not an afterthought.
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
        try:
            for frame in self.source:
                t0 = time.perf_counter()
                results = [a.analyze(frame) for a in self.analyzers]
                latency_ms = (time.perf_counter() - t0) * 1000.0
                for r in results:
                    r.metrics["latency_ms"] = latency_ms
                    for s in self.sinks:
                        s.write(r)
                yield results
        finally:
            for s in self.sinks:
                s.close()

    def run(self) -> list[list[AnalysisResult]]:
        return list(self.stream())
