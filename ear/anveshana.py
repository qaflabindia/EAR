"""Anveshana -- discover: find which of the runtime's Karma processes are
relevant to a Sankalpa, before anything is selected or composed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .karma import Karma
from .sankalpa import Sankalpa


@dataclass
class Anveshana:
    """Anveshana searches the runtime's registered Karma processes for ones
    whose name overlaps the Sankalpa's words; keyword overlap is enough for
    discovery, swap in embeddings if you need more for your runtime."""

    def discover(self, runtime: Any, sankalpa: Sankalpa) -> list[Karma]:
        words = {word.lower() for word in sankalpa.text.split() if len(word) > 3}
        if not words:
            return list(runtime.processes)
        matches = [process for process in runtime.processes if any(word in process.name.lower() for word in words)]
        return matches or list(runtime.processes)
