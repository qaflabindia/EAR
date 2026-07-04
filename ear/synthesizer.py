"""Synthesizer -- combine several sub-agents' decisions into one coherent
decision. The join Delegator's fan-out needs.

DeerFlow calls this "structured result aggregation": sub-agents run in
parallel, report back, and the lead agent synthesizes everything into a
coherent output. Judged in natural language by the active model when one
is configured (agreement reconciled, conflicts noted, one clear outcome
stated); without one, a deterministic join keeps synthesis usable and
testable offline, exactly like every other judgment-laden stage here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .intent import Intent


@dataclass
class Synthesizer:
    """A Synthesizer folds a list of `(label, decision)` sub-agent results
    into one final decision."""

    def synthesize(self, runtime: Any, intent: Intent, sub_decisions: list[tuple[str, Any]]) -> Any:
        if len(sub_decisions) == 1:
            # Nothing to reconcile -- the one sub-agent's decision stands.
            return sub_decisions[0][1]
        model_binding = getattr(runtime, "model_binding", None)
        if model_binding is not None and getattr(model_binding, "lm", None) is not None:
            return self._synthesize_with_llm(intent, sub_decisions, model_binding.lm)
        return self._default_synthesis(sub_decisions)

    @staticmethod
    def _default_synthesis(sub_decisions: list[tuple[str, Any]]) -> str:
        lines = "\n".join(f"- {label}: {decision}" for label, decision in sub_decisions)
        return f"Synthesized from {len(sub_decisions)} sub-agents:\n{lines}"

    @staticmethod
    def _synthesize_with_llm(intent: Intent, sub_decisions: list[tuple[str, Any]], lm: Any) -> str:
        import dspy

        from .signatures import SynthesizeSubAgentResults

        rendered = "\n".join(f"{label}: {decision}" for label, decision in sub_decisions)
        synthesizer = dspy.Predict(SynthesizeSubAgentResults)
        with dspy.context(lm=lm):
            result = synthesizer(intent_text=intent.text, sub_agent_results=rendered)
        return result.synthesis
