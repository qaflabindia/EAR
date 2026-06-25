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
