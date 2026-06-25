"""Scheduler -- order the Composer's composed plan before execution.
Workflow carries no priority field today, so this is the seam future
ordering logic (priority, dependency, cost) hooks into without touching
callers."""

from __future__ import annotations

from dataclasses import dataclass

from .workflow import Workflow


@dataclass
class Scheduler:
    """A Scheduler orders a composed plan. With no priority concept on
    Workflow yet, it returns a defensive copy in discovery order."""

    def schedule(self, plan: list[Workflow]) -> list[Workflow]:
        return list(plan)
