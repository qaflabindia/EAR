"""Varana -- select: choose which Anveshana-discovered Karma processes
actually run this cycle. Named Varana (not the similarly-spelled Varna,
EAR's workflow class) to keep the two visually and structurally distinct."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .karma import Karma


@dataclass
class Varana:
    """Varana selects from discovered candidates, deduplicating by process
    name while preserving discovery order."""

    def select(self, runtime: Any, candidates: list[Karma]) -> list[Karma]:
        seen: set[str] = set()
        selected: list[Karma] = []
        for process in candidates:
            if process.name not in seen:
                seen.add(process.name)
                selected.append(process)
        return selected
