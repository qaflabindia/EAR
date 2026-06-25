"""Anukulana -- adapt: periodically distill Anubhava experience into a new
Samskara. Throttled by `adapt_every` rather than firing on every cycle, so
adaptation doesn't spam one fresh impression per call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .anubhava import Anubhava
from .samskara import Samskara, SamskaraBank


@dataclass
class Anukulana:
    """Anukulana adapts the runtime's behaviour by calling SamskaraBank's
    distillation every `adapt_every` observed cycles."""

    adapt_every: int = 5

    def adapt(
        self,
        samskara: SamskaraBank,
        anubhava: Anubhava,
        summarizer: Optional[Any] = None,
    ) -> Optional[Samskara]:
        if self.adapt_every <= 0 or anubhava.observations == 0:
            return None
        if anubhava.observations % self.adapt_every != 0:
            return None
        return samskara.learn_from(anubhava, summarizer=summarizer)
