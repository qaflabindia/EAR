"""Anubhava -- experience: what repeated execution shows, aggregated from
Smṛti but not yet a behaviour change. AI systems often skip straight from
raw memory to an adaptation; Anubhava is the missing middle step -- the
pattern a Saṃskāra is then distilled from."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .pramana import Pramana
from .smriti import Smriti, SmritiEntry


@dataclass
class Anubhava:
    """Anubhava aggregates repeated Smriti entries into counts and the
    evidence seen along the way, without yet drawing a conclusion -- that's
    Samskara's job."""

    observations: int = 0
    decision_counts: dict[str, int] = field(default_factory=dict)
    evidence_seen: list[Pramana] = field(default_factory=list)

    def observe_entry(self, entry: SmritiEntry) -> "Anubhava":
        key = str(entry.decision)
        self.decision_counts[key] = self.decision_counts.get(key, 0) + 1
        self.observations += 1
        if entry.evidence is not None:
            self.evidence_seen.append(entry.evidence)
        return self

    def observe(self, smriti: Smriti) -> "Anubhava":
        """Re-aggregate from a Smriti's current `working` entries -- useful
        to rebuild experience from memory loaded after the fact, rather
        than accumulated incrementally via `observe_entry`."""
        for entry in smriti.working:
            self.observe_entry(entry)
        return self

    def most_common_decision(self) -> Optional[tuple[str, int]]:
        if not self.decision_counts:
            return None
        decision = max(self.decision_counts, key=self.decision_counts.get)
        return decision, self.decision_counts[decision]

    def summary(self) -> str:
        if not self.decision_counts:
            return "No observations yet."
        ranked = sorted(self.decision_counts.items(), key=lambda kv: -kv[1])
        lines = (f"- '{decision}': {count}/{self.observations} cycles" for decision, count in ranked)
        return "\n".join(lines)

    def __len__(self) -> int:
        return self.observations
