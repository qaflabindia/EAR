"""Spawner -- spawn a subagent: a child Runtime scoped to one Persona,
reasoning one intent with the parent's model binding and strategy.

Subagent spawning is governed by the Subagent Spawning section of
`memory.md`: the strategy's prose decides whether spawning is allowed and
how many subagents a runtime may spawn, and the Spawner enforces those
limits the way the Governor enforces Policies -- by raising
`PermissionError` rather than silently proceeding. Each subagent gets its
own memory (its cycles do not pollute the parent's history) but shares the
parent's ModelBinding and Strategy, so it reasons with the same model and
the same enterprise vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Spawner:
    """A Spawner spawns subagent runtimes, bounded by the strategy stacked
    in memory.md (enabled/disabled, and an optional maximum)."""

    enabled: bool = True
    limit: Optional[int] = None
    spawned: list[Any] = field(default_factory=list)

    def spawn(self, runtime: Any, persona: Any, intent: Any) -> Any:
        """Spawn a subagent runtime scoped to `persona`, reason `intent`
        through it, and return the subagent's decision."""
        if not self.enabled:
            raise PermissionError("Subagent spawning is disabled by the runtime's strategy")
        if self.limit is not None and len(self.spawned) >= self.limit:
            raise PermissionError(f"Subagent limit of {self.limit} reached for runtime '{runtime.name}'")

        from .intent import Intent
        from .process import Process
        from .runtime import Runtime
        from .workflow import Workflow

        if not isinstance(intent, Intent):
            intent = Intent(text=str(intent))

        workflow = Workflow(name=f"{persona.name} Subagent Workflow")
        workflow.add_persona(persona)
        process = Process(
            name=f"{persona.name} Subagent",
            description=persona.instructions or f"A subagent scoped to the persona {persona.name}.",
        )
        process.add_workflow(workflow)

        subagent = Runtime(
            name=f"{runtime.name}::{persona.name}",
            model_binding=getattr(runtime, "model_binding", None),
            strategy=getattr(runtime, "strategy", None),
        )
        subagent.add_process(process)
        # Nested spawns count against the same budget: a subagent spawning
        # its own subagents cannot exceed the strategy's limit either.
        subagent.spawner = self
        # Isolation nests: a sandboxed runtime hands each subagent its own
        # child box under the parent's root, so a spawned instance runs in
        # its own workspace rather than the parent's.
        parent_sandbox = getattr(runtime, "sandbox", None)
        if parent_sandbox is not None:
            subagent.sandbox = parent_sandbox.child(persona.name)
            if getattr(runtime.tool_binder, "sandbox_tools", None):
                subagent.tool_binder.sandbox_tools = subagent.sandbox.as_tools()
        self.spawned.append(subagent)
        return subagent.reason(intent)
