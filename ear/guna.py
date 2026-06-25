"""Guna -- the persona: skills stacked into a behavioural nature."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .vidya import Vidya


@dataclass
class Guna:
    """A Guna is a persona: a stack of Vidya skills plus standing instructions."""

    name: str
    instructions: str = ""
    skills: list[Vidya] = field(default_factory=list)

    def add_skill(self, skill: Vidya) -> "Guna":
        self.skills.append(skill)
        return self

    def get_skill(self, name: str) -> Optional[Vidya]:
        return next((s for s in self.skills if s.name == name), None)
