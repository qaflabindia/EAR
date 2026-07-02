"""Recaller -- recall the Memory context relevant to a cycle and snapshot
it, so what was actually remembered when a decision was made is itself
part of that decision's evidence trail.

Recall is a runtime judgment: when a ModelBinding is active, the LLM reads
the full remembered context window against the intent and recalls only
what is genuinely relevant -- prior decisions, amounts and outcomes that
should inform this cycle -- instead of dragging the whole history into
evidence. With no model it falls back to the full context window, so
nothing is ever lost offline. The recalled context (and which mind
recalled it) is written to the ReasoningLog; the underlying Memory is
never altered by recall -- selection happens on the way out, the record
stays intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .intent import Intent
from .memory import Memory
from .reasoning_log import model_name


@dataclass
class Recaller:
    """A Recaller recalls from Memory: LLM-judged relevance when a model
    is active, the full context window otherwise."""

    def recall(self, memory: Memory, intent: Intent, runtime: Any = None) -> str:
        window = memory.context_window()
        model_binding = getattr(runtime, "model_binding", None)
        if not window or model_binding is None or model_binding.lm is None:
            return window
        recalled = self._recall_with_llm(intent, window, model_binding.lm)
        log = getattr(runtime, "reasoning_log", None)
        if log is not None:
            log.record(
                stage="recall",
                inputs={"intent": intent.text, "history": window},
                output=recalled or window,
                model=model_name(model_binding),
            )
        # An empty recall falls back to the full window: forgetting
        # everything is never the right reading of "nothing was relevant".
        return recalled or window

    @staticmethod
    def _recall_with_llm(intent: Intent, window: str, lm: Any) -> str:
        from .signatures import RecallRelevantMemory

        result = RecallRelevantMemory.run(lm, intent_text=intent.text, history=window)
        return str(result.relevant_context).strip()