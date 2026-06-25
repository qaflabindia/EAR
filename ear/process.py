"""Process -- workflows stacked into an executable action.

Carries a `description` (alongside its `name`) so the Discoverer can reason
in natural language about which processes are relevant to an Intent, rather
than only matching on keywords in the name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .workflow import Workflow


@dataclass
class Process:
    """A Process is a stack of Workflows that performs an action."""

    name: str
    description: str = ""
    workflows: list[Workflow] = field(default_factory=list)

    def add_workflow(self, workflow: Workflow) -> "Process":
        self.workflows.append(workflow)
        return self
