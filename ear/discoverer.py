"""Discoverer -- find which of the runtime's processes are relevant to an
intent, before anything is selected or composed.

When a ModelBinding is active, relevance is judged by an LLM reading each
process's name and natural-language description against the intent -- not
a keyword match. Without an LLM, it falls back to keyword overlap so the
package stays usable offline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .intent import Intent
from .process import Process


@dataclass
class Discoverer:
    """A Discoverer searches the runtime's registered processes for ones
    relevant to an intent."""

    def discover(self, runtime: Any, intent: Intent) -> list[Process]:
        model_binding = getattr(runtime, "model_binding", None)
        if model_binding is not None and getattr(model_binding, "lm", None) is not None:
            return self._discover_with_llm(runtime.processes, intent, model_binding.lm)
        return self._discover_by_keyword(runtime.processes, intent)

    @staticmethod
    def _discover_by_keyword(processes: list[Process], intent: Intent) -> list[Process]:
        words = {word.lower() for word in intent.text.split() if len(word) > 3}
        if not words:
            return list(processes)
        matches = [process for process in processes if any(word in process.name.lower() for word in words)]
        return matches or list(processes)

    @staticmethod
    def _discover_with_llm(processes: list[Process], intent: Intent, lm: Any) -> list[Process]:
        if not processes:
            return []
        import dspy

        from .signatures import DiscoverRelevantProcesses

        catalogue = "\n".join(f"{process.name}: {process.description or 'no description'}" for process in processes)
        finder = dspy.Predict(DiscoverRelevantProcesses)
        with dspy.context(lm=lm):
            result = finder(intent_text=intent.text, available_processes=catalogue)
        by_name = {process.name: process for process in processes}
        found = [by_name[name] for name in result.relevant_process_names if name in by_name]
        return found or list(processes)
