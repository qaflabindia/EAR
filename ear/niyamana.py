"""Niyamana -- govern: the regulation gate a cycle must clear before anything
else runs. Kept as its own step so policy enforcement is a named operation
rather than logic buried inside the runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dharma import Dharma
from .sankalpa import Sankalpa


@dataclass
class Niyamana:
    """Niyamana governs a cycle: it checks the runtime's Dharma policies
    against a Sankalpa's context and reports which ones are violated."""

    def govern(self, runtime: Any, sankalpa: Sankalpa) -> list[Dharma]:
        return [policy for policy in runtime.policies if not policy.evaluate(**sankalpa.context)]
