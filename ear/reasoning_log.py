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

Declared in `memory.md` (a Reasoning Audit Trail section naming a path);
the Runtime flushes new records to that file after every cycle,
append-only, so the trail also accumulates across sessions. The file's
extension picks the codec: `.md` (the system-native default) appends
readable markdown, one `## Cycle` section per cycle with every free-text
value blockquoted so it can never be mistaken for structure; any other
extension appends JSONL for machine pipelines.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .section import quote

_CYCLE_HEADING = re.compile(r"^## Cycle (\d+)", re.MULTILINE)


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

    def to_markdown(self) -> str:
        """This record as a markdown block: one-line inputs as bullets,
        multi-line values (the stacked capabilities, a long decision) as
        blockquotes under their own label."""
        header = f"### {self.stage}"
        if self.output:
            header += f" -- {_clip(self.output, 80)}"
        if self.model:
            header += f"  ({self.model})"
        lines = [header, ""]
        simple = {key: value for key, value in self.inputs.items() if "\n" not in str(value)}
        multiline = {key: value for key, value in self.inputs.items() if "\n" in str(value)}
        if simple:
            lines += [f"- {key}: {value}" for key, value in simple.items()] + [""]
        for key, value in multiline.items():
            if str(value).strip():
                lines += [f"{key.capitalize()}:", quote(value), ""]
        if self.rationale:
            lines += ["Why:", quote(self.rationale), ""]
        if "\n" in self.output or len(self.output) > 80:
            lines += ["Output:", quote(self.output), ""]
        return "\n".join(lines)


@dataclass
class ReasoningLog:
    """The runtime's reasoning audit trail: an ordered list of
    ReasoningRecords, grouped by cycle, flushed append-only to the trail
    file at `path` and fanned out to any attached `exporters`.

    An exporter is anything with `export(record)` (and optionally
    `flush()`) -- e.g. `ear.integrations.otel_backend.OpenTelemetryExporter`
    for Langfuse/Phoenix/any OTLP backend. The file on disk stays the
    canonical record: an exporter that raises never breaks a cycle, its
    failure is kept visible in `export_errors` instead."""

    path: str = ""
    records: list[ReasoningRecord] = field(default_factory=list)
    cycle: int = 0
    flushed: int = 0
    flushed_cycle: Optional[int] = None
    exporters: list[Any] = field(default_factory=list)
    export_errors: list[str] = field(default_factory=list)

    def resume(self) -> int:
        """Continue cycle numbering from an existing trail file, so a new
        session's cycles never repeat numbers inside the same audit trail.
        A missing or unreadable file leaves the counter untouched."""
        if not self.path or not os.path.exists(self.path):
            return self.cycle
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            return self.cycle
        numbers: list[int] = []
        if self.path.endswith(".md"):
            numbers = [int(number) for number in _CYCLE_HEADING.findall(text)]
        else:
            for line in text.splitlines():
                try:
                    numbers.append(int(json.loads(line).get("cycle", 0)))
                except (ValueError, AttributeError):
                    continue
        self.cycle = max(numbers, default=self.cycle)
        return self.cycle

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
        """Write records not yet flushed to the trail file at `path`
        (markdown when the path ends in `.md`, JSONL otherwise) and fan
        the same records out to every attached exporter. With no path and
        no exporters the trail stays in memory only."""
        pending = self.records[self.flushed :]
        if not pending or (not self.path and not self.exporters):
            return None
        if self.path:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            as_markdown = self.path.endswith(".md")
            with open(self.path, "a", encoding="utf-8") as handle:
                for record in pending:
                    if as_markdown:
                        if record.cycle != self.flushed_cycle:
                            self.flushed_cycle = record.cycle
                            handle.write(
                                f"\n## Cycle {record.cycle} -- {record.timestamp:%Y-%m-%d %H:%M:%S} UTC\n\n"
                            )
                        handle.write(record.to_markdown() + "\n")
                    else:
                        handle.write(record.to_json() + "\n")
        for exporter in self.exporters:
            try:
                for record in pending:
                    exporter.export(record)
                finish = getattr(exporter, "flush", None)
                if callable(finish):
                    finish()
            except Exception as error:  # noqa: BLE001 -- an exporter must never break a cycle
                self.export_errors.append(f"{type(exporter).__name__}: {error}")
                del self.export_errors[:-20]
        self.flushed = len(self.records)
        return self.path or None

    def __len__(self) -> int:
        return len(self.records)


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."
