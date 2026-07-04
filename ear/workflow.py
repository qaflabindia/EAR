"""Workflow -- an ordered list of Steps (each delegated to a Persona),
governed by its own Policies.

The author adds steps in plain English and delegates each to a Persona
(`add_step`), and attaches the Policies that govern this workflow
(`add_policy`). The runtime reasons the ordered steps out through their
delegated personas and enforces the workflow's policies before it runs.

`personas` and `add_persona` remain for the simpler case of stacking
personas directly with no per-step narration.

Set `parallel=True` to declare this workflow's fan-out as independent
sub-agents rather than one shared reasoning call: each delegated Persona
is dispatched in isolation (Delegator) and their results are folded into
one decision (Synthesizer) -- EAR's declarative take on sub-agent
decomposition. Leave it False (the default) for the ordinary, single
shared call every workflow has always used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .persona import Persona
from .policy import Policy
from .step import Step


@dataclass
class Workflow:
    """A Workflow is an ordered list of Steps delegated to Personas, plus
    the Policies that govern it."""

    name: str
    personas: list[Persona] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    policies: list[Policy] = field(default_factory=list)
    parallel: bool = False

    def add_persona(self, persona: Persona) -> "Workflow":
        self.personas.append(persona)
        return self

    def add_step(self, instruction: str, persona: Optional[Persona] = None, name: str = "") -> "Workflow":
        """Add one ordered step, narrated in plain English and delegated to
        a Persona the runtime reasons it through."""
        self.steps.append(Step(instruction=instruction, persona=persona, name=name))
        return self

    def add_policy(self, policy: Policy) -> "Workflow":
        """Attach a Policy that governs this workflow; the runtime enforces
        it before the workflow's steps run."""
        self.policies.append(policy)
        return self

    def delegated_personas(self) -> list[Persona]:
        """Every Persona this workflow reasons through -- those delegated to
        a step and any stacked directly -- in order, de-duplicated by
        identity."""
        seen_ids: set[int] = set()
        ordered: list[Persona] = []
        candidates = [step.persona for step in self.steps if step.persona is not None]
        candidates += list(self.personas)
        for persona in candidates:
            if id(persona) not in seen_ids:
                seen_ids.add(id(persona))
                ordered.append(persona)
        return ordered
