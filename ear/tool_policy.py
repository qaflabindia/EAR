"""ToolPolicy -- governance for an *action*, the counterpart to Policy's
governance of a *decision*.

A ToolPolicy decides whether a tool may be called with the arguments
proposed. It is judged exactly like a Policy -- in natural language by an
LLM when a ModelBinding is active ("a bureau pull is only allowed when
consent is on file"), with a safe-eval `fallback_expression` over the
call's arguments (plus the special `tool` and `permissions` values) so it
stays enforceable offline. The `tool` field scopes which tool it governs
("*" for every tool), so a runtime can carry one broad policy and several
tool-specific ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .policy import Policy


@dataclass
class ToolPolicy:
    """A ToolPolicy governs a tool call. `permits` returns True when the
    call is allowed. It delegates the actual judgment to a Policy, so the
    LLM-judged / safe-eval-fallback / reject-unsafe-expression behaviour is
    identical to decision governance -- no second, divergent rule engine."""

    name: str
    statement: str = ""
    fallback_expression: str = ""
    tool: str = "*"

    def applies_to(self, tool_name: str) -> bool:
        return self.tool == "*" or self.tool == tool_name

    def permits(self, model_binding: Optional[Any] = None, **call_context: Any) -> bool:
        """Return True when the tool call satisfies this policy. `call_context`
        is the call's arguments plus `tool` (the tool name) and `permissions`
        (the tool's capability tags)."""
        return Policy(
            name=self.name,
            statement=self.statement,
            fallback_expression=self.fallback_expression,
        ).evaluate(model_binding=model_binding, **call_context)
