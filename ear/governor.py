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
        if model_binding is not None:
            # Activation just builds (once) and configures the DSPy LM --
            # idempotent and cheap -- so policies are judged against a real
            # model here rather than silently falling back because the
            # Initializer step hasn't run yet.
            model_binding.activate()
        return self._violations(runtime.policies, model_binding, intent)

    def govern_workflows(self, runtime: Any, intent: Intent, workflows: Any) -> list[Policy]:
        """Enforce the policies attached to each workflow in the composed
        plan. Runtime-wide policies are checked by `govern` up front; a
        workflow's own policies can only be checked once the plan is known,
        so this runs after composition and before the workflow's steps are
        reasoned."""
        model_binding = getattr(runtime, "model_binding", None)
        policies = [policy for workflow in workflows for policy in getattr(workflow, "policies", [])]
        return self._violations(policies, model_binding, intent)

    @staticmethod
    def _violations(policies: Any, model_binding: Any, intent: Intent) -> list[Policy]:
        return [
            policy
            for policy in policies
            if not policy.evaluate(model_binding=model_binding, **intent.context)
        ]
