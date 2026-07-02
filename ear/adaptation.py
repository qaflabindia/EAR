"""Adaptation -- how future behaviour should change, distilled from
Experience rather than raw Memory directly. Skipping straight from memory
to "lesson learned" is exactly the conflation this package's
Evidence / Memory / Experience / Adaptation split is meant to avoid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .experience import Experience


@dataclass
class Adaptation:
    """One standing impression: an insight distilled from Experience (e.g.
    "purchases from Procurement over $400 get escalated -- learned from 3
    past approvals") that the Runtime surfaces back to the Reasoner."""

    name: str
    insight: str
    confidence: float = 1.0
    evidence_count: int = 0


@dataclass
class AdaptationBank:
    """The runtime's long-term, distilled memory: a small set of
    Adaptations learned from Experience, plus the machinery to learn more."""

    impressions: list[Adaptation] = field(default_factory=list)

    def add(self, adaptation: Adaptation) -> "AdaptationBank":
        self.impressions.append(adaptation)
        return self

    def relevant_to(self, intent_text: str) -> list[Adaptation]:
        """Keyword-overlap retrieval; swap in embeddings if you need more
        than this for your runtime."""
        words = {word.lower() for word in intent_text.split() if len(word) > 3}
        if not words:
            return list(self.impressions)
        return [a for a in self.impressions if any(word in a.insight.lower() for word in words)]

    def learn_from(self, experience: Experience, summarizer: Optional[Any] = None) -> Optional[Adaptation]:
        """Distill the current Experience into one new Adaptation.

        Pass an activated `LM` (e.g. `runtime.model_binding.lm`) as
        `summarizer` for an LLM-distilled insight; without one, the most
        frequently repeated decision in the experience is reported instead.
        """
        if not experience.decision_counts:
            return None
        if summarizer is not None:
            insight = self._distill_with_llm(experience.summary(), summarizer)
        else:
            common = experience.most_common_decision()
            if common is None:
                return None
            decision, count = common
            insight = f"Most frequent outcome: '{decision[:80]}' ({count}/{experience.observations} cycles)"
        adaptation = Adaptation(
            name=f"Learned-{len(self.impressions) + 1}",
            insight=insight,
            evidence_count=experience.observations,
        )
        self.add(adaptation)
        return adaptation

    @staticmethod
    def _distill_with_llm(experience_summary: str, lm: Any) -> str:
        from .signatures import DistillInsight

        result = DistillInsight.run(lm, experience_summary=experience_summary)
        return result.insight
