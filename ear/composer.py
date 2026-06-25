"""Composer -- assemble selected processes' workflows into one ordered
execution plan, ready for the Scheduler to order."""

from __future__ import annotations

from dataclasses import dataclass

from .process import Process
from .workflow import Workflow


@dataclass
class Composer:
    """A Composer flattens the workflows of every selected process into one
    composed plan."""

    def compose(self, selected: list[Process]) -> list[Workflow]:
        plan: list[Workflow] = []
        for process in selected:
            plan.extend(process.workflows)
        return plan
