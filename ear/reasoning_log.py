"""ReasoningLog -- the audit trail of every reasoning step the runtime
takes, kept so the LLM's judgment is reviewable and the stacked prompts
can be optimised against what the model actually reasoned over.

Every judgment-laden stage writes one record per judgment:

    intent        the cycle opened, with the intent and its context
    policy        each Policy judgment, with the judge's rationale
    discovery     which processes were found relevant, and from what catalogue
    deliberation  the Reasoner's decision, with the full stacked capabilities
                  block and memory context it reasoned with -- the exact
                  prompt material an author reviews to optimise skills.md
    explanation   the Explainer's prose and the evidence it rested on

Records carry which model produced them ("deterministic-fallback" when no
ModelBinding was active), so offline and live cycles are distinguishable in
the same trail. Blocked cycles are logged too -- a Policy violation is
exactly what an auditor wants to see, not a gap in the record.

Declared in `memory.md` (a Reasoning Audit Trail section naming a `.jsonl`
path); the Runtime flushes new records to that file after every cycle,
append-only, so the trail also accumulates across sessions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def model_name(model_binding: Any) -> str:
    """The name a record attributes its judgment to: the bound model when
    one is active, the deterministic fallback otherwise."""
    if model_binding is not None and getattr(model_binding, "lm", None) is not None:
        return model_binding.model_id
    return "deterministic-fallback"


@dataclass
class ReasoningRecord:
    """One logged judgment: which cycle and stage, what the stage reasoned
    over (`inputs`), what it concluded (`output`), why (`rationale`), and
    which model concluded it."""

    cycle: int
    stage: str
    inputs: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    rationale: str = ""
    model: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def render(self, width: int = 160) -> str:
        line = f"[{self.stage}] -> {_clip(self.output, width)}"
        if self.model:
            line += f"  ({self.model})"
        if self.rationale:
            line += f"\n    why: {_clip(self.rationale, width)}"
        return line

    def to_json(self) -> str:
        return json.dumps(
            {
                "cycle": self.cycle,
                "stage": self.stage,
                "timestamp": self.timestamp.isoformat(),
                "model": self.model,
                "inputs": self.inputs,
                "output": self.output,
                "rationale": self.rationale,
            },
            default=str,
        )


@dataclass
class ReasoningLog:
    """The runtime's reasoning audit trail: an ordered list of
    ReasoningRecords, grouped by cycle, flushed append-only to a JSONL
    file when a `path` is set."""

    path: str = ""
    records: list[ReasoningRecord] = field(default_factory=list)
    cycle: int = 0
    flushed: int = 0

    def begin_cycle(self, intent: Any) -> int:
        """Open a new cycle in the trail, recording the intent that
        started it."""
        self.cycle += 1
        self.record(
            stage="intent",
            inputs={"context": dict(getattr(intent, "context", {}) or {})},
            output=str(intent),
        )
        return self.cycle

    def record(
        self,
        stage: str,
        inputs: Optional[dict[str, Any]] = None,
        output: Any = "",
        rationale: str = "",
        model: str = "",
    ) -> ReasoningRecord:
        entry = ReasoningRecord(
            cycle=self.cycle,
            stage=stage,
            inputs=dict(inputs or {}),
            output=str(output),
            rationale=str(rationale),
            model=model,
        )
        self.records.append(entry)
        return entry

    def for_stage(self, stage: str) -> list[ReasoningRecord]:
        return [record for record in self.records if record.stage == stage]

    def for_cycle(self, cycle: int) -> list[ReasoningRecord]:
        return [record for record in self.records if record.cycle == cycle]

    def render(self, cycle: Optional[int] = None, width: int = 160) -> str:
        """The trail as readable text, one cycle per block -- the skim
        view; full inputs (the stacked capabilities, the judged context)
        stay on the records and in the JSONL file."""
        records = self.records if cycle is None else self.for_cycle(cycle)
        lines: list[str] = []
        seen_cycle: Optional[int] = None
        for record in records:
            if record.cycle != seen_cycle:
                seen_cycle = record.cycle
                lines.append(f"=== Cycle {record.cycle} ({record.timestamp:%Y-%m-%d %H:%M:%S}) ===")
            lines.append(record.render(width=width))
        return "\n".join(lines) if lines else "No reasoning recorded yet."

    def flush(self) -> Optional[str]:
        """Append records not yet written to the JSONL file at `path`.
        With no path set the trail stays in memory only."""
        if not self.path or self.flushed >= len(self.records):
            return None
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            for record in self.records[self.flushed :]:
                handle.write(record.to_json() + "\n")
        self.flushed = len(self.records)
        return self.path

    def __len__(self) -> int:
        return len(self.records)


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."
