"""Selector -- choose which Discoverer-found processes actually run this
cycle.

Selection is a runtime judgment, not a build-time rule: when a ModelBinding
is active and there is more than one candidate, the LLM reads each
candidate's name and description against the intent and chooses which
processes this cycle genuinely needs, most relevant first. With no model
(or a single candidate) it falls back to deduplication in discovery order,
so the package stays fully usable offline. Either way the choice -- and
which mind made it -- is written to the runtime's ReasoningLog.

The LLM chooses among discovered candidates only: it can narrow and rank,
it cannot invent a process that was never discovered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .intent import Intent
from .process import Process
from .reasoning_log import model_name


@dataclass
class Selector:
    """A Selector selects from discovered candidates: an LLM judgment when
    a model is active, deduplication in discovery order otherwise."""

    def select(self, runtime: Any, candidates: list[Process], intent: Optional[Intent] = None) -> list[Process]:
        deduped = self._dedupe(candidates)
        model_binding = getattr(runtime, "model_binding", None)
        selected = deduped
        if len(deduped) > 1 and intent is not None and model_binding is not None and model_binding.lm is not None:
            selected = self._select_with_llm(deduped, intent, model_binding.lm)
        log = getattr(runtime, "reasoning_log", None)
        if log is not None and len(deduped) > 1:
            # Only a real choice is worth an audit record; a single
            # candidate leaves nothing to select.
            log.record(
                stage="selection",
                inputs={
                    "intent": intent.text if intent is not None else "",
                    "candidates": [process.name for process in deduped],
                },
                output=", ".join(process.name for process in selected),
                model=model_name(model_binding),
            )
        return selected

    @staticmethod
    def _dedupe(candidates: list[Process]) -> list[Process]:
        seen: set[str] = set()
        deduped: list[Process] = []
        for process in candidates:
            if process.name not in seen:
                seen.add(process.name)
                deduped.append(process)
        return deduped

    @staticmethod
    def _select_with_llm(candidates: list[Process], intent: Intent, lm: Any) -> list[Process]:
        from .discoverer import _match_by_name
        from .signatures import SelectProcesses

        catalogue = "\n".join(f"{process.name}: {process.description or 'no description'}" for process in candidates)
        result = SelectProcesses.run(lm, intent_text=intent.text, candidate_processes=catalogue)
        chosen = _match_by_name(candidates, result.selected_process_names)
        # An empty or unusable answer falls back to every candidate rather
        # than silently running none.
        return chosen or candidates