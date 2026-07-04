"""TaskDefinition -- one atomic, independently reusable step, SIPOC-shaped.

Named `TaskDefinition` rather than the bare `Task` to avoid colliding with
`ear.kernel.Task` (a runtime's scheduled dispatch unit -- a different
concept: *when* something runs, not *what* one step of it does).

A Workflow's `Step` is inline and disposable: instruction plus a delegated
Persona, authored fresh (or copy-pasted) into every workflow that needs it.
A TaskDefinition is the same unit of work made a first-class, storable,
named object -- so "Sanity Check" or "Dashboard Validation" is written
once, catalogued in a TaskStore, and referenced by name from any workflow
that needs it, the same way a Skill is written once and referenced from
any persona.

The SIPOC fields (supplier / inputs / process / outputs / customer) carry
the value-chain context a bare instruction string doesn't: who/what feeds
this step, what it consumes, what it does, what it produces, and who
consumes that -- exactly the shape a reasoner benefits from recalling on a
recurring task's second and later runs, per the runtime's evidence/memory
retrieval path. `artifact` names the script or file the step's process
relies on (e.g. `validate_data.py`), so a rerun's reasoning can tell "reuse
this" from "none exists yet, write one".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .persona import Persona
from .section import Section
from .step import Step


@dataclass
class TaskDefinition:
    """One atomic, storable step: what it does (`instruction`) plus its
    SIPOC context and the artifact its process relies on."""

    name: str
    instruction: str = ""
    persona_name: str = ""
    supplier: str = ""
    inputs: str = ""
    process: str = ""
    outputs: str = ""
    customer: str = ""
    artifact: str = ""

    def to_step(self, persona: Optional[Persona] = None) -> Step:
        """This Task as a Workflow Step, delegated to `persona` (typically
        resolved from `persona_name` against a loaded persona catalogue)."""
        return Step(instruction=self.instruction, persona=persona, name=self.name)

    def sipoc(self) -> str:
        """The SIPOC block rendered as a natural-language summary -- what
        gets stacked into a recurring run's retrieved knowledge so the
        reasoner sees last time's supplier/input/process/output/customer
        shape without re-deriving it."""
        lines = [f"Step: {self.name}"]
        for label, value in (
            ("Supplier", self.supplier),
            ("Inputs", self.inputs),
            ("Process", self.process),
            ("Outputs", self.outputs),
            ("Customer", self.customer),
            ("Artifact", self.artifact),
        ):
            if value:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render this task as one heading with its SIPOC fields, then the
        instruction -- parsed back by `from_section`/`from_markdown`."""
        lines = [f"## {self.name}", ""]
        if self.persona_name:
            lines.append(f"Persona: {self.persona_name}")
        for label, value in (
            ("Supplier", self.supplier),
            ("Inputs", self.inputs),
            ("Process", self.process),
            ("Outputs", self.outputs),
            ("Customer", self.customer),
            ("Artifact", self.artifact),
        ):
            if value:
                lines.append(f"{label}: {value}")
        lines.append("")
        if self.instruction:
            lines.append(self.instruction)
        return "\n".join(lines).rstrip() + "\n"

    @classmethod
    def from_section(cls, section: Section) -> "TaskDefinition":
        body = section.body(
            field_keys=(
                "persona",
                "supplier",
                "inputs",
                "input",
                "process",
                "outputs",
                "output",
                "customer",
                "artifact",
            )
        )
        return cls(
            name=section.name,
            instruction=body.prose,
            persona_name=body.field("persona"),
            supplier=body.field("supplier"),
            inputs=body.field("inputs", "input"),
            process=body.field("process"),
            outputs=body.field("outputs", "output"),
            customer=body.field("customer"),
            artifact=body.field("artifact"),
        )

    @classmethod
    def from_markdown(cls, text: str) -> "TaskDefinition":
        from .section import parse_document

        document = parse_document(text)
        if not document.sections:
            raise ValueError("Task markdown has no heading to read a task from")
        return cls.from_section(document.sections[0])
