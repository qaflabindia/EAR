"""ToolBinder -- give the stack's declared tools execution, on the record.

Tools are *declared* in natural language (memory.md's Tools section) and
skills may carry a Python handler; the ToolBinder is where a declaration
meets an executable. `bind(name, handler)` attaches a callable to a
declared Tool, and any stacked Skill that carries a handler is bound
automatically for the workflows in the cycle's plan. The stack remains
the source of what exists: binding a name nothing in the stack declares
fails loudly -- code must never grow the runtime a capability the
natural-language authoring doesn't show.

*When* to use a tool stays the model's judgment: with bound tools present,
deliberation runs as EAR's native tool loop over them (the declared
description is what the model reads). Every invocation is a trail record
(stage `tool`) with the arguments, the result and the duration; a failing
tool never breaks the cycle -- the failure is recorded and handed back to
the model as text, and the model reasons on. Declared-but-unbound tools
stay what they always were: context the model knows about, surfaced
through the strategy narrative.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .section import normalize


@dataclass
class BoundTool:
    """One executable tool for a cycle: the declared name, the declared
    description the model reads, and the callable that runs."""

    name: str
    description: str
    handler: Callable[..., Any]
    # Parameter names supplied from outside the handler -- an MCP tool's
    # JSON input schema, say, where the handler is a generic forwarder and
    # introspecting it would tell the model nothing.
    parameter_names: Optional[list[str]] = None

    @property
    def identifier(self) -> str:
        mapped = "".join(ch if ch.isalnum() else "_" for ch in self.name.strip().lower())
        return mapped or "tool"

    @property
    def parameters(self) -> list[str]:
        """The tool's parameter names -- supplied explicitly (an MCP
        schema) or introspected from the handler, so the model is always
        told exactly what arguments a tool takes, from the standard library
        alone."""
        if self.parameter_names is not None:
            return list(self.parameter_names)
        import inspect

        try:
            signature = inspect.signature(self.handler)
        except (TypeError, ValueError):
            return []
        return [
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]


@dataclass
class ToolBinder:
    """A ToolBinder resolves the stack's declared tools and handler-skills
    into the executable, logged toolset a cycle deliberates with."""

    bindings: dict[str, Callable[..., Any]] = field(default_factory=dict)
    # Tools bound from connected MCP servers (see Runtime.connect_mcp):
    # already-executable BoundTools that join every cycle's toolset with the
    # same logging, budgets and tool-scoped policies as native tools.
    mcp_tools: list[BoundTool] = field(default_factory=list)
    # Confined file/shell tools from the runtime's Sandbox (see
    # Sandbox.as_tools), joined to the toolset the same way.
    sandbox_tools: list[BoundTool] = field(default_factory=list)
    # The runtime's own list/view/create/retire tools (see Runtime.acquirer
    # and Acquirer.as_tools) -- how a runtime grows its own toolset, joined
    # to the toolset the same way.
    acquirer_tools: list[BoundTool] = field(default_factory=list)
    # The basic toolsets (Toolsets in memory.md -- web access/search,
    # email; see ear/web.py, ear/mail.py) declared enabled for this
    # runtime, joined to the toolset the same way.
    basic_tools: list[BoundTool] = field(default_factory=list)
    # How many tool-loop steps a deliberation may take; execution mechanics,
    # not judgment -- the model decides what to call within the budget.
    max_iterations: int = 6

    @staticmethod
    def tool_key(name: str) -> str:
        """The lookup key for a tool name, case- and punctuation-folded, so
        the model naming a tool loosely still resolves to the right one."""
        return normalize(name)

    @staticmethod
    def parse_arguments(items: Any) -> dict[str, Any]:
        """Read a tool call's arguments into a typed kwargs dict, coerced by
        the same codec as intent context so numbers arrive as numbers and a
        genuinely multi-line value (a script's source, a whole file) arrives
        intact. `items` is normally the `name -> value` map `ChooseToolAction`
        already parsed via `section.argument_blocks`; a flat list of
        '- name: value' strings is also accepted for a caller that hasn't
        gone through that parser. Shared by every native tool loop
        (deliberation and panel turns alike)."""
        from .section import coerce

        if isinstance(items, dict):
            return {str(name).strip(): coerce(value) for name, value in items.items() if str(name).strip()}
        arguments: dict[str, Any] = {}
        for item in items or []:
            name, separator, value = str(item).partition(":")
            if separator and name.strip():
                arguments[name.strip()] = coerce(value)
        return arguments

    def bind(self, name: str, handler: Callable[..., Any]) -> "ToolBinder":
        self.bindings[normalize(name)] = handler
        return self

    def bound_tools(self, runtime: Any, plan: Optional[list] = None) -> list[BoundTool]:
        """The cycle's executable toolset: explicit bindings resolved
        against what the stack declares (Tools in memory.md, or any
        stacked skill), plus the plan's handler-carrying skills bound
        automatically. An explicit binding for a skill overrides the
        skill's own handler."""
        strategy = getattr(runtime, "strategy", None)
        declared = {normalize(tool.name): tool for tool in (getattr(strategy, "tools", None) or [])}
        stack_skills = self._skills(getattr(runtime, "processes", []) or [])
        plan_skills = self._skills_from_plan(plan or [])

        bound: dict[str, BoundTool] = {}
        for key, handler in self.bindings.items():
            if key in declared:
                tool = declared[key]
                bound[key] = BoundTool(name=tool.name, description=tool.description or tool.name, handler=handler)
            elif key in stack_skills:
                skill = stack_skills[key]
                bound[key] = BoundTool(name=skill.name, description=skill.instruction(), handler=handler)
            else:
                known = ", ".join(sorted({tool.name for tool in declared.values()} | {s.name for s in stack_skills.values()})) or "none"
                raise ValueError(
                    f"Tool binding '{key}' matches nothing the stack declares -- declare it under "
                    f"Tools in memory.md or as a skill first; declared: {known}"
                )
        for key, skill in plan_skills.items():
            if skill.handler is not None and key not in bound:
                bound[key] = BoundTool(name=skill.name, description=skill.instruction(), handler=skill.handler)
        for tool in self.mcp_tools:
            bound.setdefault(normalize(tool.name), tool)
        for tool in self.sandbox_tools:
            bound.setdefault(normalize(tool.name), tool)
        for tool in self.acquirer_tools:
            bound.setdefault(normalize(tool.name), tool)
        for tool in self.basic_tools:
            bound.setdefault(normalize(tool.name), tool)
        return list(bound.values())

    def logged_handler(self, runtime: Any, tool: BoundTool) -> Callable[..., Any]:
        """A tool's handler wrapped so every call lands on the reasoning
        trail and a failure returns to the model as text instead of
        breaking the cycle. The Reasoner's native tool loop invokes tools
        through this."""
        return self._logged(runtime, tool)

    @staticmethod
    def _logged(runtime: Any, tool: BoundTool) -> Callable[..., Any]:
        @functools.wraps(tool.handler)
        def invoke(*args: Any, **kwargs: Any) -> Any:
            log = getattr(runtime, "reasoning_log", None)
            # Tool-scoped policies are judged against this call's name and
            # arguments before it runs: a violation blocks *this* call, the
            # refusal returns to the model as text, and the record shows it.
            blocked, reason = ToolBinder._policy_block(runtime, tool, args, kwargs)
            if blocked is not None:
                refusal = f"Tool '{tool.name}' blocked by policy '{blocked.name}': {reason}"
                if log is not None:
                    log.record(
                        stage="tool",
                        inputs={
                            "tool": tool.name,
                            "arguments": {"args": list(args), "kwargs": dict(kwargs)},
                            "policy": blocked.name,
                        },
                        output=f"BLOCKED by policy '{blocked.name}'",
                        rationale=reason,
                    )
                return refusal
            started = time.monotonic()
            try:
                result = tool.handler(*args, **kwargs)
                outcome, failed = str(result), False
            except Exception as error:  # noqa: BLE001 -- the failure goes back to the model as text
                outcome, failed = f"Tool '{tool.name}' failed: {error}", True
                result = outcome
            if log is not None:
                log.record(
                    stage="tool",
                    inputs={
                        "tool": tool.name,
                        "arguments": {"args": list(args), "kwargs": dict(kwargs)},
                        "duration_ms": int((time.monotonic() - started) * 1000),
                    },
                    output=outcome if not failed else f"FAILED -- {outcome}",
                )
            return result

        return invoke

    @staticmethod
    def _policy_block(runtime: Any, tool: BoundTool, args: tuple, kwargs: dict) -> tuple[Optional[Any], str]:
        """The first tool-scoped policy this call violates, and why -- or
        (None, "") when every policy passes or none is declared. The call's
        name and arguments are the context judged, so a policy statement
        ('the wire transfer tool must not send over $10,000') or a fallback
        expression ('amount <= 10000') can both bite."""
        policies = getattr(runtime, "tool_policies", None) or []
        if not policies:
            return None, ""
        model_binding = getattr(runtime, "model_binding", None)
        context: dict[str, Any] = {"tool": tool.name, **kwargs}
        for index, value in enumerate(args):
            context[f"arg{index}"] = value
        for policy in policies:
            complies, rationale = policy.judge(model_binding=model_binding, **context)
            if not complies:
                return policy, rationale
        return None, ""

    @staticmethod
    def _skills_from_plan(plan: list) -> dict[str, Any]:
        skills: dict[str, Any] = {}
        for workflow in plan:
            for persona in workflow.delegated_personas():
                for skill in persona.skills:
                    skills.setdefault(normalize(skill.name), skill)
        return skills

    @classmethod
    def _skills(cls, processes: list) -> dict[str, Any]:
        skills: dict[str, Any] = {}
        for process in processes:
            skills.update(cls._skills_from_plan(getattr(process, "workflows", []) or []))
        return skills