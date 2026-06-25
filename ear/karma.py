"""Karma -- the process: workflows stacked into an executable action."""

from __future__ import annotations

from dataclasses import dataclass, field

from .varna import Varna


@dataclass
class Karma:
    """A Karma is a process: a stack of Varna workflows that performs an action."""

    name: str
    workflows: list[Varna] = field(default_factory=list)

    def add_workflow(self, workflow: Varna) -> "Karma":
        self.workflows.append(workflow)
        return self
