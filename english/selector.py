"""Selector -- choose which Discoverer-found processes actually run this
cycle. Purely structural deduplication, with no judgment call to make, so
it stays plain Python rather than reaching for an LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .process import Process


@dataclass
class Selector:
    """A Selector selects from discovered candidates, deduplicating by
    process name while preserving discovery order."""

    def select(self, runtime: Any, candidates: list[Process]) -> list[Process]:
        seen: set[str] = set()
        selected: list[Process] = []
        for process in candidates:
            if process.name not in seen:
                seen.add(process.name)
                selected.append(process)
        return selected
