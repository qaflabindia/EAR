"""Tool -- a capability declared to the runtime in plain English.

Tools are stacked in `memory.md` under the Tools strategy section: one
bullet per tool, `name: what it does`. The runtime surfaces the declared
tools to the Reasoner as part of its operating strategy, so the model knows
what is available to it and can decide, in natural language, when a tool is
the right move -- nothing about *when* to use a tool is hardcoded. An
optional `command` (backticked in the declaration) records how the tool is
invoked for integrations that execute it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Tool:
    """One declared tool: a name, a plain-English description of what it
    does, and optionally the command that invokes it."""

    name: str
    description: str = ""
    command: str = ""
    # Provenance: "authored" (hand-typed in memory.md, the default) or
    # "acquired" (declared at runtime by the Acquirer -- see ear/acquirer.py).
    # Only an acquired tool may be retired through code; an authored one is
    # edited by editing memory.md.
    origin: str = "authored"

    def describe(self) -> str:
        line = self.name
        if self.description:
            line += f": {self.description}"
        if self.command:
            line += f" (invoked via `{self.command}`)"
        return line
