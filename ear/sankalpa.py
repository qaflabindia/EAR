"""Sankalpa -- the prompt: a resolved intent that starts a reasoning cycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sankalpa:
    """A Sankalpa is a prompt: the intent handed to the runtime."""

    text: str
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text
