"""Vicara -- reason: deliberate via Bhuddi given the runtime's current
state. Kept distinct from Nirnaya (committing to a decision) and Pariksha
(validating it) so deliberation, decision and validation are three
separate, inspectable steps rather than one opaque call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .sankalpa import Sankalpa


@dataclass
class Vicara:
    """Vicara deliberates by handing the Sankalpa to the runtime's Bhuddi
    reasoner."""

    def deliberate(self, runtime: Any, sankalpa: Sankalpa) -> Any:
        return runtime.reasoner.reason(sankalpa, runtime=runtime)
