"""Vidya -- the skill: a single addressable capability a persona can invoke."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Vidya:
    """A Vidya is a skill: prompts are stacked inside it to give it a job to do."""

    name: str
    description: str = ""
    handler: Optional[Callable[..., Any]] = None
    source: Optional[str] = None

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        if self.handler is None:
            raise NotImplementedError(f"Vidya '{self.name}' has no handler attached")
        return self.handler(*args, **kwargs)
