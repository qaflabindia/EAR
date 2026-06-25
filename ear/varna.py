"""Varna -- the workflow: personas stacked into an ordered role sequence."""

from __future__ import annotations

from dataclasses import dataclass, field

from .guna import Guna


@dataclass
class Varna:
    """A Varna is a workflow: an ordered stack of Guna personas."""

    name: str
    personas: list[Guna] = field(default_factory=list)

    def add_persona(self, persona: Guna) -> "Varna":
        self.personas.append(persona)
        return self
