"""Deliberator -- deliberate via the runtime's Reasoner given its current
state. Kept distinct from Decider (committing to a decision) and Validator
(validating it) so deliberation, decision and validation are three
separate, inspectable steps rather than one opaque call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .intent import Intent


@dataclass
class Deliberator:
    """A Deliberator deliberates by handing the intent to the runtime's
    Reasoner."""

    def deliberate(self, runtime: Any, intent: Intent, plan: Any = None) -> Any:
        return runtime.reasoner.reason(intent, runtime=runtime, plan=plan)
