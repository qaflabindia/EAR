"""Pramāṇa -- evidence: the justification for one decision, distinct from
the decision itself (Smṛti) or any pattern drawn from repeating it
(Anubhava). AI systems routinely conflate these three; EAR keeps them
separate so "why" survives independently of "what happened"."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Pramana:
    """A Pramana is the evidentiary basis for a single Bhuddi decision:
    which policies it cleared, which reasoning path produced it, and
    whatever else justifies "why", as opposed to merely recording "what"."""

    basis: str
    sources: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def __str__(self) -> str:
        return self.basis
