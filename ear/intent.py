"""Intent -- the prompt: a resolved request that starts a reasoning cycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Intent:
    """An Intent is a prompt: the request handed to the runtime."""

    text: str
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    def continued_with(self, decision: Any) -> "Intent":
        """Return the follow-on Intent for the next cycle of a goal-driven
        loop: the same request, with the prior decision threaded into
        context so the next cycle builds on it rather than starting cold.

        Memory already threads prior cycles back into reasoning; this also
        exposes the latest decision (as `decision`) and the running list
        (`_prior_decisions`) to a Goal's deterministic fallback."""
        prior = list(self.context.get("_prior_decisions", []))
        prior.append(decision)
        new_context = {
            **self.context,
            "_prior_decision": decision,
            "_prior_decisions": prior,
        }
        return Intent(text=self.text, context=new_context)
