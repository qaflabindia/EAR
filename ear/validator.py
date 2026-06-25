"""Validator -- the checker layer every "maker" stage's output passes
through before the next stage trusts it. Discoverer, Selector, Composer,
Scheduler and Decider each produce something; Validator is the one place
that output is checked, instead of each maker re-implementing its own
validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .process import Process
from .workflow import Workflow


@dataclass
class Validator:
    """A Validator checks the output of every maker stage in the pipeline:
    the Discoverer's candidates, the Selector's selection, the Composer's
    plan, the Scheduler's schedule, and the Decider's decision."""

    def validate_candidates(self, candidates: list[Process]) -> list[Process]:
        return self._validate_list(candidates, Process, "Discoverer candidates")

    def validate_selection(self, selected: list[Process]) -> list[Process]:
        return self._validate_list(selected, Process, "Selector selection")

    def validate_plan(self, plan: list[Workflow]) -> list[Workflow]:
        return self._validate_list(plan, Workflow, "Composer plan")

    def validate_schedule(self, scheduled: list[Workflow]) -> list[Workflow]:
        return self._validate_list(scheduled, Workflow, "Scheduler schedule")

    def validate(self, decision: Any) -> Any:
        if isinstance(decision, str) and not decision.strip():
            raise ValueError("Validator rejected an empty decision")
        return decision

    @staticmethod
    def _validate_list(items: Any, item_type: type, label: str) -> list:
        if not isinstance(items, list):
            raise TypeError(f"{label} must be a list, got {type(items).__name__}")
        for item in items:
            if not isinstance(item, item_type):
                raise TypeError(f"{label} must contain only {item_type.__name__} instances")
        return items
