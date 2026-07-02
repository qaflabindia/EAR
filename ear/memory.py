"""Memory -- persistent memory: what happened. Held in layers so the
runtime's context stays bounded as it grows. The evidentiary "why" behind
each entry lives in its `evidence` (Evidence), not folded into the memory
itself -- a distinction AI systems often blur.

Compression of overflowed history is written by an LLM (natural-language
summarization, via the SummarizeHistory signature) when a `summarizer` LM
is supplied; otherwise a deterministic digest is used, so memory
compression never requires an LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .evidence import Evidence


@dataclass
class MemoryEntry:
    """One remembered cycle: an intent, what the Reasoner decided, the
    intent's own input context, and -- separately -- the Evidence for why
    that decision was made."""

    intent_text: str
    decision: Any
    context: dict[str, Any] = field(default_factory=dict)
    evidence: Optional[Evidence] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def render(self) -> str:
        line = f"- ({self.timestamp:%Y-%m-%d %H:%M}) '{self.intent_text}' -> {self.decision}"
        if self.evidence is not None:
            line += f" [{self.evidence.basis}]"
        return line


@dataclass
class Memory:
    """The runtime's persistent memory. Recent cycles are kept verbatim in
    the `working` layer; once that layer grows past `capacity`, the oldest
    entries are rolled into the `compressed` layer as a single summary
    string. This is the memory compression step: it keeps the context the
    Reasoner sees bounded instead of growing forever with raw history."""

    capacity: int = 20
    working: list[MemoryEntry] = field(default_factory=list)
    compressed: list[str] = field(default_factory=list)

    def record(
        self,
        intent_text: str,
        decision: Any,
        context: Optional[dict[str, Any]] = None,
        evidence: Optional[Evidence] = None,
        summarizer: Optional[Any] = None,
    ) -> MemoryEntry:
        """Record one cycle. Pass the active LM as `summarizer` so any
        overflow compression this record triggers is written by the model
        rather than the deterministic digest."""
        entry = MemoryEntry(
            intent_text=intent_text,
            decision=decision,
            context=context or {},
            evidence=evidence,
        )
        self.working.append(entry)
        if len(self.working) > self.capacity:
            self.compress(summarizer=summarizer)
        return entry

    def compress(self, summarizer: Optional[Any] = None, keep: Optional[int] = None) -> Optional[str]:
        """Roll the oldest entries past `keep` (default `capacity`) out of
        `working` into one new summary string in `compressed`.

        Pass an activated `LM` (e.g. `runtime.model_binding.lm`) as
        `summarizer` for an LLM-written summary; without one, a
        deterministic digest is used so compression never requires an LLM.
        """
        keep = self.capacity if keep is None else keep
        if len(self.working) <= keep:
            return None
        if keep <= 0:
            overflow, self.working = self.working, []
        else:
            overflow, self.working = self.working[:-keep], self.working[-keep:]
        history = "\n".join(entry.render() for entry in overflow)
        if summarizer is not None:
            summary = self._summarize_with_llm(history, summarizer)
        else:
            decisions = ", ".join(str(entry.decision)[:60] for entry in overflow)
            summary = f"{len(overflow)} earlier cycles (e.g. {decisions})"
        self.compressed.append(summary)
        return summary

    @staticmethod
    def _summarize_with_llm(history: str, lm: Any) -> str:
        from .signatures import SummarizeHistory

        result = SummarizeHistory.run(lm, history=history)
        return result.summary

    def context_window(self, max_working: Optional[int] = None) -> str:
        """Render compressed history plus recent working entries as one
        string, ready to drop into a reasoning prompt."""
        working = self.working if max_working is None else self.working[-max_working:]
        parts = []
        if self.compressed:
            parts.append("Earlier history (compressed):\n" + "\n".join(self.compressed))
        if working:
            parts.append("Recent history:\n" + "\n".join(entry.render() for entry in working))
        return "\n\n".join(parts)

    def __len__(self) -> int:
        return len(self.working) + len(self.compressed)
