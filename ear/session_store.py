"""SessionStore -- cross-session data: the runtime's Memory, Experience and
Adaptations persisted to disk, so a new session picks up where the last one
left off instead of starting cold.

Declared in `memory.md` under the Cross-Session Data strategy section (the
loader creates the store and restores from it before the first cycle), and
written back automatically after every `Runtime.reason()` cycle. The file
is plain JSON -- inspectable, diffable, no pickled code -- and holds only
what the memory layers already record: what happened (Memory), the pattern
(Experience) and the distilled lessons (Adaptations). Evidence's "why"
travels as its basis sentence.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .adaptation import Adaptation
from .evidence import Evidence
from .memory import MemoryEntry


@dataclass
class SessionStore:
    """Persists the runtime's memory layers to one JSON file and restores
    them into a fresh runtime at the start of the next session."""

    path: str

    def save(self, runtime: Any) -> str:
        payload = {
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
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return self.path

    def restore(self, runtime: Any) -> bool:
        """Load persisted layers back into the runtime. Returns False (and
        leaves the runtime untouched) when there is nothing usable to load,
        so a missing or corrupt store never blocks a session from starting."""
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return False

        remembered = payload.get("memory", {})
        runtime.memory.compressed = list(remembered.get("compressed", []))
        runtime.memory.working = [self._entry(record) for record in remembered.get("working", [])]

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
        return True

    @staticmethod
    def _entry(record: dict[str, Any]) -> MemoryEntry:
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
