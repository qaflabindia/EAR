"""Workflow -- personas stacked into an ordered role sequence."""

from __future__ import annotations

from dataclasses import dataclass, field

from .persona import Persona


@dataclass
class Workflow:
    """A Workflow is an ordered stack of Personas."""

    name: str
    personas: list[Persona] = field(default_factory=list)

    def add_persona(self, persona: Persona) -> "Workflow":
        self.personas.append(persona)
        return self
