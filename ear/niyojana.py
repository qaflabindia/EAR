"""Niyojana -- schedule: order Samyojana's composed plan before execution.
Varna carries no priority field today, so this is the seam future ordering
logic (priority, dependency, cost) hooks into without touching callers."""

from __future__ import annotations

from dataclasses import dataclass

from .varna import Varna


@dataclass
class Niyojana:
    """Niyojana orders a composed plan. With no priority concept on Varna
    yet, it returns a defensive copy in discovery order."""

    def schedule(self, plan: list[Varna]) -> list[Varna]:
        return list(plan)
