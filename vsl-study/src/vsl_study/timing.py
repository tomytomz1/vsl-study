"""Monotonic stage timing for jobs. Cache hits are recorded separately from work."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


class StageClock:
    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.stages: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}

    def note(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    @contextmanager
    def span(self, name: str, *, cache_hit: bool = False) -> Iterator[dict[str, Any]]:
        started = time.monotonic()
        row: dict[str, Any] = {
            "name": name,
            "cache_hit": bool(cache_hit),
            "elapsed_s": 0.0,
        }
        try:
            yield row
        finally:
            row["elapsed_s"] = round(time.monotonic() - started, 3)
            self.stages.append(row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_s": round(time.monotonic() - self.started_at, 3),
            "stages": list(self.stages),
            "metrics": dict(self.metrics),
        }
