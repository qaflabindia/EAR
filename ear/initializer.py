"""Initializer -- the begin step of a cycle, where the ModelBinding (LLM
provider) is activated before reasoning starts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Initializer:
    """An Initializer begins a cycle by activating the runtime's
    ModelBinding, if it has one, so the rest of the cycle runs against a
    configured LLM."""

    def initialize(self, runtime: Any) -> Optional[Any]:
        if runtime.model_binding is not None:
            return runtime.model_binding.activate()
        return None
