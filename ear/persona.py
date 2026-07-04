"""Persona -- skills stacked into a behavioural nature."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .skill import Skill


@dataclass
class Persona:
    """A Persona is a stack of Skills plus standing instructions."""

    name: str
    instructions: str = ""
    skills: list[Skill] = field(default_factory=list)

    def add_skill(self, skill: Skill) -> "Persona":
        self.skills.append(skill)
        return self

    def get_skill(self, name: str) -> Optional[Skill]:
        return next((s for s in self.skills if s.name == name), None)

    def to_markdown(self) -> str:
        """Render this persona the way persona.md stacks one -- a heading,
        a `Skills:` reference line, then the standing instructions. Read
        back by `Loader._load_personas` against an already-loaded skills
        catalogue, so cross-store composition (skills first, then
        personas) works the same as stacking within one file."""
        lines = [f"## {self.name}", ""]
        if self.skills:
            lines += [f"Skills: {', '.join(skill.name for skill in self.skills)}", ""]
        if self.instructions:
            lines.append(self.instructions)
        return "\n".join(lines).rstrip() + "\n"
