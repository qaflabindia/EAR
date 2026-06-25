"""Kriya -- perform: the lowest-level action of a cycle. Chains Vicara
(deliberate), Nirnaya (decide) and Pariksha (validate) into one performed
action, so each sub-step stays inspectable and swappable on its own."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .nirnaya import Nirnaya
from .pariksha import Pariksha
from .sankalpa import Sankalpa
from .vicara import Vicara


@dataclass
class Kriya:
    """Kriya performs one action: deliberate, decide, then validate."""

    vicara: Vicara = field(default_factory=Vicara)
    nirnaya: Nirnaya = field(default_factory=Nirnaya)
    pariksha: Pariksha = field(default_factory=Pariksha)

    def perform(self, runtime: Any, sankalpa: Sankalpa) -> Any:
        deliberation = self.vicara.deliberate(runtime, sankalpa)
        decision = self.nirnaya.decide(deliberation)
        return self.pariksha.validate(decision)
