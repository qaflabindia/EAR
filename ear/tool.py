"""Tool -- a declared capability the runtime can *act* through, not just
reason about.

EAR governs decisions; a Tool is how it governs *actions*. A Tool is
declared the same prompt-first way a Skill is: a name and a plain-English
`contract` are required (the contract is what an LLM reads to know what the
tool does), while a Python `handler` is the optional advanced layer -- so a
declared-but-unimplemented tool still keeps the pipeline usable. Every
actual invocation goes through the `Invoker`, which clears the Governor's
`ToolPolicy` gates and records the call as Evidence before the handler ever
runs -- governed, audited side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Tool:
    """A Tool is a named, contract-described capability. `permissions` are
    the capability tags a ToolPolicy can gate on (e.g. "read:bureau"). The
    `handler` is optional: without one, `run` returns a side-effect-free
    simulated result so a tool can be declared and governed before it is
    implemented."""

    name: str
    contract: str = ""
    handler: Optional[Callable[..., Any]] = None
    permissions: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """The plain-English contract stacked into reasoning, falling back
        to the name so a tool always contributes some signal."""
        return self.contract or self.name

    def run(self, **args: Any) -> Any:
        """Execute the tool. With a handler, that is the real action; without
        one, a simulated, side-effect-free placeholder so the runtime stays
        usable and testable for a tool that is declared but not yet built.

        Invoke through `Invoker`/`Runtime.invoke`, not directly, so the call
        is governed and audited -- `run` is the raw action underneath that
        gate."""
        if self.handler is None:
            rendered = ", ".join(f"{key}={value!r}" for key, value in args.items())
            return f"[simulated {self.name}({rendered})]"
        return self.handler(**args)
