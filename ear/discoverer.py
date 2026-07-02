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
from .reasoning_log import calls_so_far, model_name, usage_since
from .section import normalize


def _match_by_name(processes: list[Process], names: list[str]) -> list[Process]:
    """Resolve model-returned names back to processes, case- and
    punctuation-insensitively, preserving the model's order and dropping
    anything it named that no process matches."""
    by_key = {normalize(process.name): process for process in processes}
    found: list[Process] = []
    for name in names:
        process = by_key.get(normalize(str(name)))
        if process is not None and process not in found:
            found.append(process)
    return found


@dataclass
class Discoverer:
    """A Discoverer searches the runtime's registered processes for ones
    relevant to an intent."""

    def discover(self, runtime: Any, intent: Intent) -> list[Process]:
        model_binding = getattr(runtime, "model_binding", None)
        start = calls_so_far(getattr(model_binding, "lm", None))
        if model_binding is not None and getattr(model_binding, "lm", None) is not None:
            found = self._discover_with_llm(runtime.processes, intent, model_binding.lm, self._guidance(runtime))
        else:
            found = self._discover_by_keyword(runtime.processes, intent)
        log = getattr(runtime, "reasoning_log", None)
        if log is not None:
            log.record(
                stage="discovery",
                inputs={
                    "intent": intent.text,
                    "available_processes": [process.name for process in runtime.processes],
                    "guidance": self._guidance(runtime),
                },
                output=", ".join(process.name for process in found) or "none",
                model=model_name(model_binding),
                usage=usage_since(getattr(model_binding, "lm", None), start),
            )
        return found

    @staticmethod
    def _guidance(runtime: Any) -> str:
        """The Skills Discovery guidance stacked in memory.md, if any --
        plain-English direction for how relevance should be judged."""
        strategy = getattr(runtime, "strategy", None)
        return getattr(strategy, "skills_discovery", "") if strategy is not None else ""

    @staticmethod
    def _discover_by_keyword(processes: list[Process], intent: Intent) -> list[Process]:
        words = {word.lower() for word in intent.text.split() if len(word) > 3}
        if not words:
            return list(processes)
        matches = [process for process in processes if any(word in process.name.lower() for word in words)]
        return matches or list(processes)

    @staticmethod
    def _discover_with_llm(processes: list[Process], intent: Intent, lm: Any, guidance: str = "") -> list[Process]:
        if not processes:
            return []
        from .signatures import DiscoverRelevantProcesses

        catalogue = "\n".join(f"{process.name}: {process.description or 'no description'}" for process in processes)
        if guidance:
            catalogue += f"\n\nGuidance for judging relevance: {guidance}"
        result = DiscoverRelevantProcesses.run(lm, intent_text=intent.text, available_processes=catalogue)
        found = _match_by_name(processes, result.relevant_process_names)
        return found or list(processes)
