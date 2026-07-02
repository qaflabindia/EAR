"""Deliberator -- deliberate via the runtime's Reasoner given its current
state. Kept distinct from Decider (committing to a decision) and Validator
(validating it) so deliberation, decision and validation are three
separate, inspectable steps rather than one opaque call.

The deliberation engine is a seam: attach a `backend` (anything with
`deliberate(runtime, intent, plan=..., research=...)` -- e.g. a typed-agent
adapter built on PydanticAI) and it replaces the Reasoner for this step
only. Everything around it stays EAR's: the Governor has already gated the
cycle, the Decider/Validator still check the result, Contracts still judge
the deliverable, and the deliberation still lands on the trail with the
backend named -- a backend never gets to reason off the record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .intent import Intent


@dataclass
class Deliberator:
    """A Deliberator deliberates by handing the intent to the runtime's
    Reasoner, or to an attached backend."""

    backend: Optional[Any] = None

    def deliberate(self, runtime: Any, intent: Intent, plan: Any = None, research: Any = None) -> Any:
        if self.backend is not None:
            decision = self.backend.deliberate(runtime, intent, plan=plan, research=research)
            log = getattr(runtime, "reasoning_log", None)
            if log is not None:
                log.record(
                    stage="deliberation",
                    inputs={"intent": intent.text, "context": dict(intent.context)},
                    output=str(decision),
                    model=f"backend:{type(self.backend).__name__}",
                )
            return decision
        return runtime.reasoner.reason(intent, runtime=runtime, plan=plan, research=research)
