"""Persona -- skills stacked into a behavioural nature."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .skill import Skill
from .tool import Tool


@dataclass
class Persona:
    """A Persona is a stack of Skills (what it knows how to reason about)
    plus optional Tools (what it can act through) and standing instructions."""

    name: str
    instructions: str = ""
    skills: list[Skill] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)

    def add_skill(self, skill: Skill) -> "Persona":
        self.skills.append(skill)
        return self

    def get_skill(self, name: str) -> Optional[Skill]:
        return next((s for s in self.skills if s.name == name), None)

    def add_tool(self, tool: Tool) -> "Persona":
        self.tools.append(tool)
        return self

    def get_tool(self, name: str) -> Optional[Tool]:
        return next((t for t in self.tools if t.name == name), None)
