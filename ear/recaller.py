"""Recaller -- recall the Memory context relevant to a cycle and snapshot
it, so what was actually remembered when a decision was made is itself
part of that decision's evidence trail."""

from __future__ import annotations

from dataclasses import dataclass

from .intent import Intent
from .memory import Memory


@dataclass
class Recaller:
    """A Recaller recalls a Memory's current context window."""

    def recall(self, memory: Memory, intent: Intent) -> str:
        return memory.context_window()
