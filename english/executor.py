"""Executor -- run the cycle's Performer action. The seam between
Orchestrator's coordination and Performer's actual performed action."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .intent import Intent
from .performer import Performer


@dataclass
class Executor:
    """An Executor executes a cycle by handing it to a Performer."""

    performer: Performer = field(default_factory=Performer)

    def execute(self, runtime: Any, intent: Intent) -> Any:
        return self.performer.perform(runtime, intent)
