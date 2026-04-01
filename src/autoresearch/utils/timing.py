"""Performance profiling and timing utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Generator


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000


@contextmanager
def timing(name: str = "operation") -> Generator[Timer, None, None]:
    """Context manager for timing operations."""
    timer = Timer(name)
    with timer:
        yield timer


class Profiler:
    """Simple profiler for tracking multiple operations."""

    def __init__(self) -> None:
        self._timings: Dict[str, float] = {}

    def start(self, name: str) -> None:
        self._timings[name] = time.time()

    def stop(self, name: str) -> float:
        if name not in self._timings:
            return 0.0
        elapsed = time.time() - self._timings[name]
        self._timings[name] = elapsed
        return elapsed

    def get_timing(self, name: str) -> float:
        return self._timings.get(name, 0.0)

    def get_all_timings(self) -> Dict[str, float]:
        return dict(self._timings)

    def summary(self) -> str:
        lines = ["Performance Summary:"]
        total = sum(self._timings.values())
        for name, elapsed in self._timings.items():
            pct = (elapsed / max(total, 0.001)) * 100
            lines.append(f"  {name}: {elapsed:.3f}s ({pct:.1f}%)")
        lines.append(f"  Total: {total:.3f}s")
        return "\n".join(lines)
