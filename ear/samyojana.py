"""Samyojana -- compose: assemble selected Karma processes' Varna workflows
into one ordered execution plan, ready for Niyojana to schedule."""

from __future__ import annotations

from dataclasses import dataclass

from .karma import Karma
from .varna import Varna


@dataclass
class Samyojana:
    """Samyojana flattens the Varna workflows of every selected Karma
    process into one composed plan."""

    def compose(self, selected: list[Karma]) -> list[Varna]:
        plan: list[Varna] = []
        for process in selected:
            plan.extend(process.workflows)
        return plan
