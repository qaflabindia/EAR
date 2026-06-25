"""Explainer -- render a human-readable explanation of why a decision was
reached, from its Evidence.

When a ModelBinding is active, the explanation is written by an LLM from
the evidence and decision in natural language. Without one, a deterministic
one-line rendering is used so explanation never requires an LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .evidence import Evidence


@dataclass
class Explainer:
    """An Explainer explains a decision by pairing its Evidence basis with
    the decision itself."""

    def explain(self, evidence: Evidence, decision: Any, model_binding: Optional[Any] = None) -> str:
        if model_binding is not None and getattr(model_binding, "lm", None) is not None:
            return self._explain_with_llm(evidence, decision, model_binding.lm)
        return f"{evidence.basis} -> {decision}"

    @staticmethod
    def _explain_with_llm(evidence: Evidence, decision: Any, lm: Any) -> str:
        import dspy

        from .signatures import ExplainDecision

        explainer = dspy.Predict(ExplainDecision)
        with dspy.context(lm=lm):
            result = explainer(basis=evidence.basis, decision=str(decision))
        return result.explanation
