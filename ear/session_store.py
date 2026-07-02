"""SessionStore -- cross-session data: the runtime's Memory, Experience and
Adaptations persisted to disk, so a new session picks up where the last one
left off instead of starting cold.

Declared in `memory.md` under the Cross-Session Data strategy section (the
loader creates the store and restores from it before the first cycle), and
written back automatically after every `Runtime.reason()` cycle.

The file's extension picks the codec. `.md` -- the system-native default --
writes the session as a readable markdown document and restores it through
the same Section parser the whole authoring stack uses: entries are
sections, facts are bullets, and every free-text value (a decision, an
insight) is blockquoted so it can never be mistaken for structure. A
`.json` path keeps the plain-JSON codec for machine pipelines. Neither
holds code, and Evidence's "why" travels as its basis sentence.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .adaptation import Adaptation
from .evidence import Evidence
from .memory import MemoryEntry
from .section import Section, coerce, normalize, parse_document, quote, unquote

_LABEL = re.compile(r"^([A-Za-z][\w ]*):\s*$")


@dataclass
class SessionStore:
    """Persists the runtime's memory layers to one file (markdown by
    default) and restores them into a fresh runtime at the start of the
    next session."""

    path: str

    def save(self, runtime: Any) -> str:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if self.path.endswith(".md"):
            text = self._render_markdown(runtime)
        else:
            text = json.dumps(self._payload(runtime), indent=2, default=str)
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return self.path

    def restore(self, runtime: Any) -> bool:
        """Load persisted layers back into the runtime. Returns False (and
        leaves the runtime untouched) when there is nothing usable to load,
        so a missing or corrupt store never blocks a session from starting."""
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                text = handle.read()
            if self.path.endswith(".md"):
                self._restore_markdown(runtime, text)
            else:
                self._restore_json(runtime, json.loads(text))
        except (OSError, ValueError):
            return False
        return True

    # -- the markdown codec (system-native) ---------------------------------

    def _render_markdown(self, runtime: Any) -> str:
        lines: list[str] = [f"# Session -- {runtime.name}", ""]
        lines += [f"Saved at: {datetime.now(timezone.utc).isoformat()}", ""]

        for number, summary in enumerate(runtime.memory.compressed, start=1):
            lines += [f"## Compressed {number}", "", quote(summary), ""]

        for number, entry in enumerate(runtime.memory.working, start=1):
            lines += [f"## Entry {number}", ""]
            lines += [f"Timestamp: {entry.timestamp.isoformat()}"]
            if entry.evidence is not None:
                lines += [
                    f"Evidence basis: {entry.evidence.basis}",
                    f"Evidence confidence: {entry.evidence.confidence}",
                ]
            lines += ["", "Intent:", quote(entry.intent_text), ""]
            lines += ["Decision:", quote(str(entry.decision)), ""]
            if entry.context:
                lines += ["Context:", ""]
                lines += [f"- {key}: {value}" for key, value in entry.context.items()]
                lines += [""]

        lines += ["## Experience", "", f"Observations: {runtime.experience.observations}", ""]
        for decision, count in runtime.experience.decision_counts.items():
            lines += ["### Observed decision", "", f"Count: {count}", "", quote(decision), ""]

        for adaptation in runtime.adaptations.impressions:
            lines += [
                f"## Adaptation -- {adaptation.name}",
                "",
                f"Confidence: {adaptation.confidence}",
                f"Evidence count: {adaptation.evidence_count}",
                "",
                quote(adaptation.insight),
                "",
            ]
        return "\n".join(lines)

    def _restore_markdown(self, runtime: Any, text: str) -> None:
        document = parse_document(text)
        working: list[MemoryEntry] = []
        compressed: list[str] = []
        observations = 0
        decision_counts: dict[str, int] = {}
        adaptations: list[Adaptation] = []

        for section in document.sections:
            key = normalize(section.name)
            if key.startswith("compressed"):
                compressed.append(unquote(section.lines))
            elif key.startswith("entry"):
                working.append(self._entry_from_section(section))
            elif key == "experience":
                observations = int(section.body(field_keys=("observations",)).field("observations") or 0)
            elif key.startswith("observed"):
                body = section.body(field_keys=("count",))
                decision_counts[unquote(section.lines)] = int(body.field("count") or 1)
            elif key.startswith("adaptation"):
                body = section.body(field_keys=("confidence", "evidence count"))
                _, _, name = section.name.partition("--")
                adaptations.append(
                    Adaptation(
                        name=name.strip() or section.name,
                        insight=unquote(section.lines),
                        confidence=float(body.field("confidence") or 1.0),
                        evidence_count=int(body.field("evidence count") or 0),
                    )
                )

        runtime.memory.compressed = compressed
        runtime.memory.working = working
        runtime.experience.observations = observations
        runtime.experience.decision_counts = decision_counts
        runtime.adaptations.impressions = adaptations

    @staticmethod
    def _entry_from_section(section: Section) -> MemoryEntry:
        body = section.body(field_keys=("timestamp", "evidence basis", "evidence confidence"))
        basis = body.field("evidence basis")
        evidence = Evidence(basis=basis, confidence=float(body.field("evidence confidence") or 1.0)) if basis else None
        try:
            timestamp = datetime.fromisoformat(body.field("timestamp"))
        except ValueError:
            timestamp = datetime.now(timezone.utc)
        blocks = _labelled_blocks(section.lines)
        context: dict[str, Any] = {}
        for bullet in body.bullets:
            key, separator, value = bullet.partition(": ")
            if not separator:
                key, separator, value = bullet.partition(":")
            if separator:
                context[key.strip()] = coerce(value)
        return MemoryEntry(
            intent_text=blocks.get("intent", ""),
            decision=blocks.get("decision", ""),
            context=context,
            evidence=evidence,
            timestamp=timestamp,
        )

    # -- the JSON codec (for machine pipelines) ------------------------------

    def _payload(self, runtime: Any) -> dict[str, Any]:
        return {
            "runtime": runtime.name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "memory": {
                "capacity": runtime.memory.capacity,
                "compressed": list(runtime.memory.compressed),
                "working": [
                    {
                        "intent_text": entry.intent_text,
                        "decision": str(entry.decision),
                        "context": entry.context,
                        "timestamp": entry.timestamp.isoformat(),
                        "evidence_basis": entry.evidence.basis if entry.evidence else "",
                        "evidence_confidence": entry.evidence.confidence if entry.evidence else 1.0,
                    }
                    for entry in runtime.memory.working
                ],
            },
            "experience": {
                "observations": runtime.experience.observations,
                "decision_counts": dict(runtime.experience.decision_counts),
            },
            "adaptations": [
                {
                    "name": adaptation.name,
                    "insight": adaptation.insight,
                    "confidence": adaptation.confidence,
                    "evidence_count": adaptation.evidence_count,
                }
                for adaptation in runtime.adaptations.impressions
            ],
        }

    def _restore_json(self, runtime: Any, payload: dict[str, Any]) -> None:
        remembered = payload.get("memory", {})
        runtime.memory.compressed = list(remembered.get("compressed", []))
        runtime.memory.working = [self._entry_from_json(record) for record in remembered.get("working", [])]

        experience = payload.get("experience", {})
        runtime.experience.observations = int(experience.get("observations", 0))
        runtime.experience.decision_counts = dict(experience.get("decision_counts", {}))

        runtime.adaptations.impressions = [
            Adaptation(
                name=record.get("name", ""),
                insight=record.get("insight", ""),
                confidence=float(record.get("confidence", 1.0)),
                evidence_count=int(record.get("evidence_count", 0)),
            )
            for record in payload.get("adaptations", [])
        ]

    @staticmethod
    def _entry_from_json(record: dict[str, Any]) -> MemoryEntry:
        basis = record.get("evidence_basis", "")
        evidence = Evidence(basis=basis, confidence=float(record.get("evidence_confidence", 1.0))) if basis else None
        try:
            timestamp = datetime.fromisoformat(record.get("timestamp", ""))
        except ValueError:
            timestamp = datetime.now(timezone.utc)
        return MemoryEntry(
            intent_text=record.get("intent_text", ""),
            decision=record.get("decision", ""),
            context=dict(record.get("context", {})),
            evidence=evidence,
            timestamp=timestamp,
        )


def _labelled_blocks(lines: list[str]) -> dict[str, str]:
    """Collect `Label:` lines followed by blockquotes into label -> text,
    e.g. the Intent and Decision blocks of a session entry."""
    blocks: dict[str, str] = {}
    label: Optional[str] = None
    pending: list[str] = []
    for line in lines + [""]:
        match = _LABEL.match(line.strip())
        stripped = line.strip()
        if match:
            if label and pending:
                blocks[normalize(label)] = unquote(pending)
            label, pending = match.group(1), []
        elif stripped.startswith(">"):
            pending.append(line)
        elif stripped and label and pending:
            blocks[normalize(label)] = unquote(pending)
            label, pending = None, []
    if label and pending:
        blocks[normalize(label)] = unquote(pending)
    return blocks