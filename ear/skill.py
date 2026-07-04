"""Skill -- a single addressable capability a persona can invoke.

A Skill is a *stacked prompt*, not a code slot. The user names the skill and
writes what it should do in plain English (`prompt`); the runtime reasons
over that prompt with the active LLM. No Python handler is required -- the
whole point of EAR is that the user stacks prompts and the runtime does the
work via reasoning. A `handler` (or evolvable `source`) remains optional for
advanced users who want a deterministic Python implementation, but a skill
with just a name and a prompt is fully valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Skill:
    """A Skill gives a persona a job to do; prompts are stacked inside it.

    `prompt` is the natural-language instruction the runtime reasons over.
    `description` is a short summary used when cataloguing the skill. Neither
    requires the user to write code. `handler`/`source` stay optional for the
    advanced, deterministic case.
    """

    name: str
    prompt: str = ""
    description: str = ""
    handler: Optional[Callable[..., Any]] = None
    source: Optional[str] = None

    # Provenance -- who authored this capability and at what version. Carried
    # into reasoning so a decision's Evidence can be traced back to the exact
    # skill (and version) that shaped it; an auditability win, not just
    # metadata.
    version: str = ""
    author: str = ""

    def instruction(self) -> str:
        """The prompt the runtime stacks into reasoning. Falls back to the
        description, then the name, so a skill always contributes some
        natural-language signal even if only loosely specified."""
        return self.prompt or self.description or self.name

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Run the optional deterministic handler. Most skills have none --
        they are reasoned over as prompts -- so this is only for the advanced
        case where the user opted into a Python implementation."""
        if self.handler is None:
            raise NotImplementedError(
                f"Skill '{self.name}' has no handler attached -- it is a prompt-only "
                "skill, reasoned over by the runtime rather than invoked as code"
            )
        return self.handler(*args, **kwargs)
