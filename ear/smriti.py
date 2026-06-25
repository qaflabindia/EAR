"""Smṛti -- persistent memory: execution history, evidence and decisions,
held in layers so the runtime's context stays bounded as it grows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class SmritiEntry:
    """One remembered cycle: a Sankalpa, what Bhuddi decided, and the
    evidence (context) behind it."""

    sankalpa_text: str
    decision: Any
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def render(self) -> str:
        return f"- ({self.timestamp:%Y-%m-%d %H:%M}) '{self.sankalpa_text}' -> {self.decision}"


@dataclass
class Smriti:
    """Smṛti is the runtime's persistent memory. Recent cycles are kept
    verbatim in the `working` layer; once that layer grows past `capacity`,
    the oldest entries are rolled into the `compressed` layer as a single
    summary string. This is the memory compression step: it keeps the
    context Bhuddi sees bounded instead of growing forever with raw
    history."""

    capacity: int = 20
    working: list[SmritiEntry] = field(default_factory=list)
    compressed: list[str] = field(default_factory=list)

    def record(
        self,
        sankalpa_text: str,
        decision: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> SmritiEntry:
        entry = SmritiEntry(sankalpa_text=sankalpa_text, decision=decision, context=context or {})
        self.working.append(entry)
        if len(self.working) > self.capacity:
            self.compress()
        return entry

    def compress(self, summarizer: Optional[Any] = None, keep: Optional[int] = None) -> Optional[str]:
        """Roll the oldest entries past `keep` (default `capacity`) out of
        `working` into one new summary string in `compressed`.

        Pass an activated `dspy.LM` (e.g. `runtime.manas.lm`) as
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
        if summarizer is not None:
            prompt = (
                "Summarize the following execution history into a short paragraph, "
                "preserving any decisions, amounts and outcomes that later reasoning "
                "might need:\n\n" + "\n".join(entry.render() for entry in overflow)
            )
            completions = summarizer(prompt=prompt)
            summary = completions[0] if completions else ""
        else:
            decisions = ", ".join(str(entry.decision)[:60] for entry in overflow)
            summary = f"{len(overflow)} earlier cycles (e.g. {decisions})"
        self.compressed.append(summary)
        return summary

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
