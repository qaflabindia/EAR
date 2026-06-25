"""Saṃskāra -- adaptation: how future behaviour should change, distilled
from Anubhava experience rather than raw Smṛti memory directly. Skipping
straight from memory to "lesson learned" is exactly the conflation EAR's
Pramana / Smriti / Anubhava / Samskara split is meant to avoid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .anubhava import Anubhava


@dataclass
class Samskara:
    """A Saṃskāra is one standing impression: an insight distilled from
    Anubhava experience (e.g. "POs from Procurement over $400 get
    escalated -- learned from 3 past approvals") that Ksetra surfaces back
    to Bhuddi."""

    name: str
    insight: str
    confidence: float = 1.0
    evidence_count: int = 0


@dataclass
class SamskaraBank:
    """The runtime's long-term, distilled memory: a small set of Samskaras
    learned from Anubhava experience, plus the machinery to learn more."""

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

    def learn_from(self, anubhava: Anubhava, summarizer: Optional[Any] = None) -> Optional[Samskara]:
        """Distill the current Anubhava experience into one new Samskara.

        Pass an activated `dspy.LM` (e.g. `runtime.manas.lm`) as
        `summarizer` for an LLM-written insight; without one, the most
        frequently repeated decision in the experience is reported instead.
        """
        if not anubhava.decision_counts:
            return None
        if summarizer is not None:
            prompt = (
                "From this aggregated execution experience, state one durable lesson "
                "that should bias future decisions, in one sentence:\n\n" + anubhava.summary()
            )
            completions = summarizer(prompt=prompt)
            insight = completions[0] if completions else ""
        else:
            common = anubhava.most_common_decision()
            if common is None:
                return None
            decision, count = common
            insight = (
                f"Most frequent outcome: '{decision[:80]}' ({count}/{anubhava.observations} cycles)"
            )
        samskara = Samskara(
            name=f"Learned-{len(self.impressions) + 1}",
            insight=insight,
            evidence_count=anubhava.observations,
        )
        self.add(samskara)
        return samskara
