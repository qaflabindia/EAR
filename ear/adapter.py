"""Adapter -- periodically distill Experience into a new Adaptation.
Throttled by `adapt_every` rather than firing on every cycle, so adaptation
doesn't spam one fresh impression per call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .adaptation import Adaptation, AdaptationBank
from .experience import Experience


@dataclass
class Adapter:
    """An Adapter adapts the runtime's behaviour by calling AdaptationBank's
    distillation every `adapt_every` observed cycles."""

    adapt_every: int = 5

    def adapt(
        self,
        adaptations: AdaptationBank,
        experience: Experience,
        summarizer: Optional[Any] = None,
    ) -> Optional[Adaptation]:
        if self.adapt_every <= 0 or experience.observations == 0:
            return None
        if experience.observations % self.adapt_every != 0:
            return None
        return adaptations.learn_from(experience, summarizer=summarizer)
