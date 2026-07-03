"""Policy -- governance mapped onto one or more processes.

A Policy is written as a plain-English `statement` (e.g. "the purchase
amount must not exceed the approver's approval limit") and is judged by an
LLM against the intent's context when a ModelBinding is active -- that is
the primary path, and it is genuinely natural-language reasoning, not a
hardcoded rule. An optional `fallback_expression` (a short boolean/
arithmetic expression over the same context, safely evaluated -- never
`eval`/`exec`) lets the same policy still be enforced deterministically
when no LLM is configured, so governance never silently passes through
just because a provider wasn't wired up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .safe_evaluator import MissingVariableError, safe_eval


@dataclass
class Policy:
    """A Policy is a governance rule that a Runtime enforces before it lets
    a Process run. With `approval_required` (authored as `Approval:
    required` in policy.md) a violation parks the cycle for a human verdict
    instead of blocking it outright -- the judgment stays the model's, the
    waiver belongs only to a human, and code enforces both."""

    name: str
    statement: str = ""
    fallback_expression: str = ""
    approval_required: bool = False
    # An approval gate may declare when a parked journey escalates
    # (`Escalate: after 3 days` in policy.md); the Journeys runner marks a
    # journey ESCALATED once the declared period passes unapproved.
    escalation: str = ""
    escalation_days: Optional[float] = None

    def evaluate(self, model_binding: Optional[Any] = None, **context: Any) -> bool:
        """Return True when the policy is satisfied (or not applicable)."""
        complies, _rationale = self.judge(model_binding=model_binding, **context)
        return complies

    def judge(self, model_binding: Optional[Any] = None, **context: Any) -> tuple[bool, str]:
        """Judge the policy and return (complies, rationale). The rationale
        is what the Governor writes to the reasoning audit trail, so *why*
        a policy passed or blocked is reviewable, not just that it did."""
        if model_binding is not None and getattr(model_binding, "lm", None) is not None:
            return self._judge_with_llm(model_binding.lm, context)
        if not self.fallback_expression:
            return True, "no model active and no fallback expression -- policy treated as not applicable"
        try:
            complies = bool(safe_eval(self.fallback_expression, context))
            return complies, f"fallback expression '{self.fallback_expression}' evaluated to {complies}"
        except MissingVariableError as missing:
            # The expression references a variable this intent's context
            # doesn't carry, so the policy doesn't apply to this intent.
            return True, f"not applicable to this intent: {missing}"

    def _judge_with_llm(self, lm: Any, context: dict[str, Any]) -> tuple[bool, str]:
        if not self.statement:
            return True, "policy has no statement to judge"
        from .signatures import JudgePolicyCompliance

        result = JudgePolicyCompliance.run(lm, policy_statement=self.statement, context=context)
        return bool(result.complies), str(result.rationale)
