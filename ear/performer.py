"""Performer -- the lowest-level action of a cycle. Chains Deliberator
(deliberate), Decider (decide) and Validator (validate) into one performed
action, so each sub-step stays inspectable and swappable on its own."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .decider import Decider
from .deliberator import Deliberator
from .intent import Intent
from .validator import Validator


@dataclass
class Performer:
    """A Performer performs one action: deliberate, decide, then validate."""

    deliberator: Deliberator = field(default_factory=Deliberator)
    decider: Decider = field(default_factory=Decider)
    validator: Validator = field(default_factory=Validator)

    def perform(self, runtime: Any, intent: Intent, plan: Any = None) -> Any:
        deliberation = self.deliberator.deliberate(runtime, intent, plan=plan)
        decision = self.decider.decide(deliberation)
        return self.validator.validate(decision)
