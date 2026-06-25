"""Skill -- a single addressable capability a persona can invoke."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Skill:
    """A Skill gives a persona a job to do; prompts are stacked inside it."""

    name: str
    description: str = ""
    handler: Optional[Callable[..., Any]] = None
    source: Optional[str] = None

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        if self.handler is None:
            raise NotImplementedError(f"Skill '{self.name}' has no handler attached")
        return self.handler(*args, **kwargs)
