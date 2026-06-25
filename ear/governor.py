"""Governor -- the regulation gate a cycle must clear before anything else
runs. Kept as its own step so policy enforcement is a named operation
rather than logic buried inside the runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .intent import Intent
from .policy import Policy


@dataclass
class Governor:
    """A Governor governs a cycle: it checks the runtime's policies against
    an intent's context and reports which ones are violated."""

    def govern(self, runtime: Any, intent: Intent) -> list[Policy]:
        model_binding = getattr(runtime, "model_binding", None)
        return [
            policy
            for policy in runtime.policies
            if not policy.evaluate(model_binding=model_binding, **intent.context)
        ]
