"""Governor -- the regulation gate a cycle must clear before anything else
runs. Kept as its own step so policy enforcement is a named operation
rather than logic buried inside the runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .intent import Intent
from .policy import Policy
from .reasoning_log import model_name


@dataclass
class Governor:
    """A Governor governs a cycle: it checks the runtime's policies against
    an intent's context and reports which ones are violated. Every judgment
    -- pass or block, with its rationale -- is written to the runtime's
    ReasoningLog, so governance leaves an audit trail rather than a bare
    boolean."""

    def govern(self, runtime: Any, intent: Intent) -> list[Policy]:
        model_binding = getattr(runtime, "model_binding", None)
        if model_binding is not None:
            # Activation just builds (once) and configures the DSPy LM --
            # idempotent and cheap -- so policies are judged against a real
            # model here rather than silently falling back because the
            # Initializer step hasn't run yet.
            model_binding.activate()
        return self._violations(runtime.policies, model_binding, intent, runtime=runtime)

    def govern_workflows(self, runtime: Any, intent: Intent, workflows: Any) -> list[Policy]:
        """Enforce the policies attached to each workflow in the composed
        plan. Runtime-wide policies are checked by `govern` up front; a
        workflow's own policies can only be checked once the plan is known,
        so this runs after composition and before the workflow's steps are
        reasoned."""
        model_binding = getattr(runtime, "model_binding", None)
        policies = [policy for workflow in workflows for policy in getattr(workflow, "policies", [])]
        return self._violations(policies, model_binding, intent, runtime=runtime)

    @staticmethod
    def _violations(policies: Any, model_binding: Any, intent: Intent, runtime: Any = None) -> list[Policy]:
        log = getattr(runtime, "reasoning_log", None)
        violations: list[Policy] = []
        for policy in policies:
            complies, rationale = policy.judge(model_binding=model_binding, **intent.context)
            if log is not None:
                log.record(
                    stage="policy",
                    inputs={
                        "policy": policy.name,
                        "statement": policy.statement,
                        "context": dict(intent.context),
                    },
                    output="complies" if complies else "VIOLATED",
                    rationale=rationale,
                    model=model_name(model_binding),
                )
            if not complies:
                violations.append(policy)
        return violations
