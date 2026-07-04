"""Invoker -- invoke a Tool under governance. The seam that makes EAR
*act* the way it *decides*: gated up front, audited after.

Every tool call runs through here so it is (1) cleared against the
Governor's ToolPolicies before anything happens -- a blocked call raises
`PermissionError` and never runs -- and (2) recorded, allowed or blocked,
into the current cycle's tool log, which `Runtime` folds into the cycle's
Evidence. So the runtime's actions land in the same audit trail as its
decisions, rather than happening off the books."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Invoker:
    """An Invoker invokes a Tool for a runtime: govern the call, run it (or
    block it), and record what happened."""

    def invoke(self, runtime: Any, tool: Any, **args: Any) -> Any:
        violations = runtime.governor.govern_tool(runtime, tool, **args)
        if violations:
            blocked_by = ", ".join(policy.name for policy in violations)
            self._record(runtime, tool, args, result=None, allowed=False, blocked_by=blocked_by)
            raise PermissionError(f"Tool '{tool.name}' blocked by: {blocked_by}")
        result = tool.run(**args)
        self._record(runtime, tool, args, result=result, allowed=True)
        return result

    @staticmethod
    def _record(runtime: Any, tool: Any, args: dict, result: Any, allowed: bool, blocked_by: str = "") -> None:
        log = getattr(runtime, "_cycle_tool_calls", None)
        if log is None:
            return
        entry: dict[str, Any] = {"tool": tool.name, "args": dict(args), "allowed": allowed}
        if allowed:
            entry["result"] = result
        else:
            entry["blocked_by"] = blocked_by
        log.append(entry)
