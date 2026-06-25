"""Step -- one ordered step in a Workflow, delegated to a Persona.

The author narrates *what* the step should accomplish in plain English
(`instruction`) and *who* carries it out (`persona`). No code: the runtime
reasons the step out via the delegated persona's stacked skill prompts.
A Step with no persona is still valid -- the runtime reasons it with the
workflow's other capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .persona import Persona


@dataclass
class Step:
    """A Step is one narrated instruction in a Workflow, delegated to a
    Persona that the runtime reasons through."""

    instruction: str
    persona: Optional[Persona] = None
    name: str = ""

    def delegate_to(self, persona: Persona) -> "Step":
        self.persona = persona
        return self
