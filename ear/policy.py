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
    a Process run."""

    name: str
    statement: str = ""
    fallback_expression: str = ""

    def evaluate(self, model_binding: Optional[Any] = None, **context: Any) -> bool:
        """Return True when the policy is satisfied (or not applicable)."""
        if model_binding is not None and getattr(model_binding, "lm", None) is not None:
            return self._judge_with_llm(model_binding.lm, context)
        if not self.fallback_expression:
            return True
        try:
            return bool(safe_eval(self.fallback_expression, context))
        except MissingVariableError:
            # The expression references a variable this intent's context
            # doesn't carry, so the policy doesn't apply to this intent.
            return True

    def _judge_with_llm(self, lm: Any, context: dict[str, Any]) -> bool:
        if not self.statement:
            return True
        import dspy

        from .signatures import JudgePolicyCompliance

        judge = dspy.Predict(JudgePolicyCompliance)
        with dspy.context(lm=lm):
            result = judge(policy_statement=self.statement, context=context)
        return bool(result.complies)
