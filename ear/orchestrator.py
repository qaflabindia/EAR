"""Orchestrator -- coordinate a cycle's execution end to end. The top of
the Orchestrator -> Executor -> Performer -> {Deliberator, Decider,
Validator} chain that Runtime delegates the back half of a cycle to."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .executor import Executor
from .intent import Intent


@dataclass
class Orchestrator:
    """An Orchestrator orchestrates a cycle by handing it to an Executor."""

    executor: Executor = field(default_factory=Executor)

    def orchestrate(self, runtime: Any, intent: Intent, plan: Any = None) -> Any:
        return self.executor.execute(runtime, intent, plan=plan)
