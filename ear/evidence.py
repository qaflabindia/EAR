"""Evidence -- the justification for one decision, distinct from the
decision itself (Memory) or any pattern drawn from repeating it
(Experience). AI systems routinely conflate these three; this package
keeps them separate so "why" survives independently of "what happened"."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    """The evidentiary basis for a single decision: which policies it
    cleared, which reasoning path produced it, and whatever else justifies
    "why", as opposed to merely recording "what"."""

    basis: str
    sources: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def __str__(self) -> str:
        return self.basis
