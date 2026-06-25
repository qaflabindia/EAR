"""Pariksha -- validate: the checker layer every "maker" stage's output
passes through before the next stage trusts it. Anveshana, Varana,
Samyojana, Niyojana and Nirnaya each produce something; Pariksha is the one
place that output is checked, instead of each maker re-implementing its
own validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .karma import Karma
from .varna import Varna


@dataclass
class Pariksha:
    """Pariksha checks the output of every maker stage in the pipeline:
    Anveshana's candidates, Varana's selection, Samyojana's plan,
    Niyojana's schedule, and Nirnaya's decision."""

    def validate_candidates(self, candidates: list[Karma]) -> list[Karma]:
        return self._validate_list(candidates, Karma, "Anveshana candidates")

    def validate_selection(self, selected: list[Karma]) -> list[Karma]:
        return self._validate_list(selected, Karma, "Varana selection")

    def validate_plan(self, plan: list[Varna]) -> list[Varna]:
        return self._validate_list(plan, Varna, "Samyojana plan")

    def validate_schedule(self, scheduled: list[Varna]) -> list[Varna]:
        return self._validate_list(scheduled, Varna, "Niyojana schedule")

    def validate(self, decision: Any) -> Any:
        if isinstance(decision, str) and not decision.strip():
            raise ValueError("Pariksha rejected an empty decision")
        return decision

    @staticmethod
    def _validate_list(items: Any, item_type: type, label: str) -> list:
        if not isinstance(items, list):
            raise TypeError(f"{label} must be a list, got {type(items).__name__}")
        for item in items:
            if not isinstance(item, item_type):
                raise TypeError(f"{label} must contain only {item_type.__name__} instances")
        return items
