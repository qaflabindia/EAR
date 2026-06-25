"""Arambha -- initialize: the begin step of a cycle, where Manas (the LLM
provider) is activated before reasoning starts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Arambha:
    """Arambha begins a cycle by activating the runtime's Manas, if it has
    one, so the rest of the cycle runs against a configured LLM."""

    def initialize(self, runtime: Any) -> Optional[Any]:
        if runtime.manas is not None:
            return runtime.manas.activate()
        return None
