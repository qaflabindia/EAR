"""Saṃskāra -- learned adaptations distilled from Smṛti memory that bias
future Bhuddi reasoning and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .smriti import Smriti


@dataclass
class Samskara:
    """A Saṃskāra is one standing impression: an insight distilled from
    past cycles (e.g. "POs from Procurement over $400 get escalated --
    learned from 3 past approvals") that Ksetra surfaces back to Bhuddi."""

    name: str
    insight: str
    confidence: float = 1.0
    evidence_count: int = 0


@dataclass
class SamskaraBank:
    """The runtime's long-term, distilled memory: a small set of Samskaras
    learned from Smriti, plus the machinery to learn more."""

    impressions: list[Samskara] = field(default_factory=list)

    def add(self, samskara: Samskara) -> "SamskaraBank":
        self.impressions.append(samskara)
        return self

    def relevant_to(self, sankalpa_text: str) -> list[Samskara]:
        """Keyword-overlap retrieval; swap in embeddings if you need more
        than this for your runtime."""
        words = {word.lower() for word in sankalpa_text.split() if len(word) > 3}
        if not words:
            return list(self.impressions)
        return [s for s in self.impressions if any(word in s.insight.lower() for word in words)]

    def learn_from(self, smriti: Smriti, summarizer: Optional[Any] = None) -> Optional[Samskara]:
        """Distill the current Smriti memory into one new Samskara.

        Pass an activated `dspy.LM` (e.g. `runtime.manas.lm`) as
        `summarizer` for an LLM-written insight; without one, the most
        frequently repeated past decision is reported instead.
        """
        if not smriti.working and not smriti.compressed:
            return None
        if summarizer is not None:
            prompt = (
                "From this execution history, state one durable lesson that should "
                "bias future decisions, in one sentence:\n\n" + smriti.context_window()
            )
            completions = summarizer(prompt=prompt)
            insight = completions[0] if completions else ""
        else:
            decisions = [str(entry.decision) for entry in smriti.working]
            if not decisions:
                return None
            most_common = max(set(decisions), key=decisions.count)
            insight = (
                f"Most frequent recent outcome: '{most_common[:80]}' "
                f"({decisions.count(most_common)}/{len(decisions)} cycles)"
            )
        samskara = Samskara(
            name=f"Learned-{len(self.impressions) + 1}",
            insight=insight,
            evidence_count=len(smriti),
        )
        self.add(samskara)
        return samskara
