"""Workflow -- an ordered list of Steps (each delegated to a Persona),
governed by its own Policies.

The author adds steps in plain English and delegates each to a Persona
(`add_step`), and attaches the Policies that govern this workflow
(`add_policy`). The runtime reasons the ordered steps out through their
delegated personas and enforces the workflow's policies before it runs.

`personas` and `add_persona` remain for the simpler case of stacking
personas directly with no per-step narration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .contract import Contract
from .persona import Persona
from .policy import Policy
from .step import Step


@dataclass
class Workflow:
    """A Workflow is an ordered list of Steps delegated to Personas, plus
    the Policies that govern it and, optionally, the Contract its decision
    must deliver (a `### Deliverable` section in workflow.md)."""

    name: str
    personas: list[Persona] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    policies: list[Policy] = field(default_factory=list)
    contract: Optional[Contract] = None
    # A deliberation pattern in plain English (`Pattern:` in workflow.md),
    # e.g. "adversarial debate, the risk officer has the last word" --
    # prose the Panel follows, never an enum.
    pattern: str = ""
    # Routing prose (`Routes:` in workflow.md), e.g. "if the grade is D or
    # E, skip to the customer note step" -- judged after each Journey leg;
    # the model chooses only among authored steps, never invents one.
    routes: str = ""
    # How many times a failed (raising) Journey leg is retried before the
    # journey gives up (`Retries:` in workflow.md, e.g. "retry a failed
    # leg twice"). None keeps plain crash-and-resume semantics.
    retry_budget: Optional[int] = None

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

    def to_markdown(self) -> str:
        """Render this workflow the way workflow.md stacks one -- a
        heading, its recognized fields, numbered steps (each delegated
        step suffixed `(Persona)`), and, if a Contract is attached, a
        `### Deliverable` section directly beneath. Read back by
        `Loader._load_workflows`/`_load_contract` against already-loaded
        personas and policies, matching cross-store composition order."""
        lines = [f"## {self.name}", ""]
        if self.pattern:
            lines.append(f"Pattern: {self.pattern}")
        if self.routes:
            lines.append(f"Routes: {self.routes}")
        if self.retry_budget is not None:
            lines.append(f"Retries: retry a failed leg {self.retry_budget} times")
        if self.policies:
            lines.append(f"Policies: {', '.join(policy.name for policy in self.policies)}")
        lines.append("")
        for number, step in enumerate(self.steps, start=1):
            suffix = f" ({step.persona.name})" if step.persona is not None else ""
            lines.append(f"{number}. {step.instruction}{suffix}")
        if self.contract is not None:
            lines += ["", "### Deliverable", ""]
            if self.contract.description:
                lines += [self.contract.description, ""]
            for field_ in self.contract.fields:
                lines.append(f"- {field_.name}: {field_.meaning}")
        return "\n".join(lines).rstrip() + "\n"

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
