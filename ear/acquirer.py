"""Acquirer -- lets a runtime list, inspect, and grow its own toolset.

EAR's tools are declared in plain English (Tools in memory.md) and bound
the same way any capability is: a name the stack knows about, wired to a
handler by the ToolBinder. The Acquirer adds one more layer, native and
dependency-free -- the runtime's own basic tools for *managing* that
toolset:

    list_tools    every tool, MCP tool, sandbox tool and skill the runtime
                  currently knows, one line each -- name, origin, bound or
                  declared-only
    view_tool     one entry's full declaration
    create_tool   declare a brand-new tool -- persisted to `.ear/tools.md`
                  (Section codec, reviewable and diffable) so the runtime's
                  toolset grows across restarts, the same way N1.6 persists
                  refined instructions
    retire_tool   remove a self-declared tool -- a rotation note replaces
                  the entry rather than a silent deletion

These four are themselves exposed as BoundTools (`as_tools`), so the model
can call them mid-deliberation: a tool that creates tools is how the
runtime evolves its own capability surface without a line of Python
changing -- the declaration lands in prose, on disk, exactly as if a human
author had typed it into memory.md by hand.

Declaring is not the same as binding, and the Acquirer never pretends
otherwise: what a newly-declared tool *does* still needs a handler (an MCP
server, a sandbox command, a Python binding) before it executes. Until
then it is context the model knows about, same as any hand-authored,
unbound Tool.

Only tools this module itself declared (`origin == "acquired"`) can be
retired through it -- a human-authored declaration in memory.md is edited
by editing memory.md, never rewritten by code.

Blast radius: when a Sandbox confines this runtime (see ear/sandbox.py --
the boundary a k8s pod's agent is meant to stay inside), `create_tool` and
`retire_tool` write `.ear/tools.md` *inside that sandbox's root*, through
its own confined `write_text`/`read_text`, never straight to the host --
the same containment `write_file`/`run_shell` already give the model. A
self-declared tool never lets an agent's writes reach further than every
other tool already lets it reach. Without a Sandbox, persistence stays
where it always was: the stack-level `.ear/tools.md` next to memory.md."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .section import normalize, parse_document
from .tool import Tool

_FIELD_KEYS = ("description", "command", "status")


@dataclass
class Acquirer:
    """Lists, inspects, and grows a runtime's declared toolset, natively."""

    # -- reading ---------------------------------------------------------------

    def list_tools(self, runtime: Any) -> str:
        """One line per capability the runtime currently knows: declared
        tools (authored and acquired), connected MCP servers' tools,
        sandbox tools, and skills -- name, origin, and whether it is bound
        to an executable handler this session."""
        strategy = getattr(runtime, "strategy", None)
        bound_keys = {normalize(bound.name) for bound in self._bound(runtime)}
        lines: list[str] = []
        for tool in getattr(strategy, "tools", None) or []:
            status = "bound" if normalize(tool.name) in bound_keys else "declared"
            origin = getattr(tool, "origin", "authored")
            lines.append(f"- {tool.name} ({origin}, {status}): {tool.description or 'no description'}")
        for mcp_tool in getattr(runtime.tool_binder, "mcp_tools", None) or []:
            lines.append(f"- {mcp_tool.name} (mcp, bound): {mcp_tool.description}")
        for sandbox_tool in getattr(runtime.tool_binder, "sandbox_tools", None) or []:
            lines.append(f"- {sandbox_tool.name} (sandbox, bound): {sandbox_tool.description}")
        for skill in self._skills(runtime):
            handler = "bound" if skill.handler is not None else "prompt-only"
            lines.append(f"- {skill.name} (skill, {handler}): {skill.description or skill.prompt or 'no description'}")
        return "\n".join(lines) if lines else "(no tools or skills declared)"

    def view_tool(self, runtime: Any, name: str) -> str:
        """The full declaration of one tool or skill, by name."""
        key = normalize(name)
        strategy = getattr(runtime, "strategy", None)
        for tool in getattr(strategy, "tools", None) or []:
            if normalize(tool.name) == key:
                parts = [
                    f"Name: {tool.name}",
                    f"Origin: {getattr(tool, 'origin', 'authored')}",
                    f"Description: {tool.description or '(none)'}",
                ]
                if tool.command:
                    parts.append(f"Command: {tool.command}")
                return "\n".join(parts)
        for bound in self._bound(runtime):
            if normalize(bound.name) == key:
                parts = [f"Name: {bound.name}", "Origin: bound", f"Description: {bound.description}"]
                if bound.parameters:
                    parts.append(f"Parameters: {', '.join(bound.parameters)}")
                return "\n".join(parts)
        for skill in self._skills(runtime):
            if normalize(skill.name) == key:
                parts = [f"Name: {skill.name}", "Origin: skill", f"Instruction: {skill.instruction()}"]
                if skill.version:
                    parts.append(f"Version: {skill.version}")
                if skill.author:
                    parts.append(f"Author: {skill.author}")
                return "\n".join(parts)
        return f"No tool or skill named '{name}' is declared."

    # -- growing -----------------------------------------------------------------

    def create_tool(self, runtime: Any, name: str, description: str, command: str = "") -> str:
        """Declare a brand-new tool: added to the running strategy's Tools
        and appended to `.ear/tools.md` when the runtime has a tools_path,
        so it survives past this session -- reviewable, diffable prose,
        never a database row. Refuses a name already declared (case- and
        punctuation-folded), so acquisition never shadows an existing
        capability.

        This only ever appends to `strategy.tools` -- `ToolBinder.bound_tools`
        (see tool_binder.py) never promotes a name out of that list into the
        executable set on its own; only an explicit `bind()`, a handler-
        carrying Skill, a connected MCP server, Sandbox tools, or the
        Acquirer's own meta-tools land there. So a name this method just
        declared is structurally unreachable through the native tool loop --
        the model naming it gets the same hallucinated-tool recovery a
        made-up name would -- until something else actually binds it."""
        strategy = getattr(runtime, "strategy", None)
        if strategy is None:
            raise ValueError("create_tool needs a runtime with a loaded strategy")
        refused = self._evolution_refusal(runtime, "create_tool", {"name": name})
        if refused:
            return refused
        key = normalize(name)
        if any(normalize(existing.name) == key for existing in strategy.tools):
            return f"A tool named '{name}' is already declared -- choose a different name or view it instead."
        tool = Tool(name=name, description=description, command=command, origin="acquired")
        strategy.tools.append(tool)
        sandbox, relpath = self._store(runtime)
        persisted = ""
        if relpath:
            self._append(sandbox, relpath, tool)
            where = f"inside the sandbox at {relpath}" if sandbox is not None else relpath
            persisted = f", persisted {where}"
        self._record(runtime, "create_tool", {"name": name, "description": description, "command": command}, f"declared '{name}'{persisted}")
        return (
            f"Declared tool '{name}'{persisted}. It is context to the model until a handler "
            "binds it (an MCP server, a sandbox command, or a Python binding)."
        )

    def retire_tool(self, runtime: Any, name: str, reason: str = "") -> str:
        """Remove a self-declared tool. Only tools this module created
        (`origin == 'acquired'`) can be retired this way -- a human-authored
        tool in memory.md is edited by editing memory.md. A rotation note
        replaces the entry in `.ear/tools.md` rather than silently deleting
        it."""
        strategy = getattr(runtime, "strategy", None)
        if strategy is None:
            raise ValueError("retire_tool needs a runtime with a loaded strategy")
        refused = self._evolution_refusal(runtime, "retire_tool", {"name": name})
        if refused:
            return refused
        key = normalize(name)
        match = next((tool for tool in strategy.tools if normalize(tool.name) == key), None)
        if match is None:
            return f"No tool named '{name}' is declared."
        if getattr(match, "origin", "authored") != "acquired":
            return f"'{match.name}' is authored in memory.md -- edit memory.md to remove it; retire_tool only manages acquired tools."
        strategy.tools.remove(match)
        sandbox, relpath = self._store(runtime)
        if relpath:
            self._mark_retired(sandbox, relpath, match.name, reason)
        self._record(runtime, "retire_tool", {"name": match.name, "reason": reason}, f"retired '{match.name}'")
        return f"Retired tool '{match.name}'."

    # -- model-facing surface ------------------------------------------------

    def as_tools(self, runtime: Any) -> list:
        """The Acquirer's own operations exposed as BoundTools, so a live
        deliberation can list, inspect, declare and retire tools the same
        way it calls any other tool -- on the trail, through the logged
        handler."""
        from .tool_binder import BoundTool

        acquirer = self

        def list_tools() -> str:
            return acquirer.list_tools(runtime)

        def view_tool(name: str) -> str:
            return acquirer.view_tool(runtime, name)

        def create_tool(name: str, description: str, command: str = "") -> str:
            return acquirer.create_tool(runtime, name, description, command)

        def retire_tool(name: str, reason: str = "") -> str:
            return acquirer.retire_tool(runtime, name, reason)

        return [
            BoundTool(
                name="list_tools",
                description="List every tool and skill this runtime currently knows, declared or bound.",
                handler=list_tools,
            ),
            BoundTool(
                name="view_tool",
                description="Show the full declaration of one named tool or skill.",
                handler=view_tool,
            ),
            BoundTool(
                name="create_tool",
                description=(
                    "Declare a brand-new tool by name and description (and optional command) -- "
                    "persisted so the runtime's toolset grows across restarts."
                ),
                handler=create_tool,
            ),
            BoundTool(
                name="retire_tool",
                description="Remove a previously self-declared tool by name, with a reason.",
                handler=retire_tool,
            ),
        ]

    # -- persistence: `.ear/tools.md` --------------------------------------------

    @staticmethod
    def load_tools(path: Union[str, Path], strategy: Any) -> list[Tool]:
        """Read `.ear/tools.md` and merge its active, non-retired entries
        into the strategy's declared tools -- skipping any name memory.md
        already declared, so the human-authored file always wins."""
        text = Path(path).read_text(encoding="utf-8")
        existing = {normalize(tool.name) for tool in strategy.tools}
        added: list[Tool] = []
        for section in parse_document(text).sections:
            body = section.body(field_keys=_FIELD_KEYS)
            if "retired" in (body.field("status") or "").lower():
                continue
            key = normalize(section.name)
            if key in existing:
                continue
            tool = Tool(
                name=section.name,
                description=body.field("description"),
                command=body.field("command"),
                origin="acquired",
            )
            strategy.tools.append(tool)
            added.append(tool)
            existing.add(key)
        return added

    @classmethod
    def _append(cls, sandbox: Optional[Any], relpath: str, tool: Tool) -> None:
        """Add one acquired tool's declaration to the end of `.ear/tools.md`
        -- never rewrites what is already there. Goes through `sandbox`
        (its confined `read_text`/`write_text`) when one is given, so a
        self-declared tool's own footprint never leaves the runtime's
        blast radius -- the same boundary `write_file`/`run_shell` already
        respect."""
        text = cls._read(sandbox, relpath) if cls._exists(sandbox, relpath) else "# Acquired Tools\n"
        if not text.endswith("\n"):
            text += "\n"
        block = [f"\n## {tool.name}", "", f"Description: {tool.description or '(none)'}"]
        if tool.command:
            block.append(f"Command: {tool.command}")
        block += ["Status: active", ""]
        cls._write(sandbox, relpath, text + "\n".join(block))

    @classmethod
    def _mark_retired(cls, sandbox: Optional[Any], relpath: str, name: str, reason: str) -> None:
        """Flip one tool's `Status:` line to retired, in place -- every
        other line of its section (and every other section) is carried
        forward untouched, so the file stays a full, honest history."""
        if not cls._exists(sandbox, relpath):
            return
        key = normalize(name)
        document = parse_document(cls._read(sandbox, relpath))
        out = ["# Acquired Tools", ""]
        for section in document.sections:
            lines = [line for line in section.lines if normalize(section.name) != key or not line.strip().lower().startswith("status:")]
            if normalize(section.name) == key:
                lines.append(f"Status: retired -- {reason}" if reason else "Status: retired")
            out.append(f"## {section.name}")
            out += lines
            out.append("")
        cls._write(sandbox, relpath, "\n".join(out).rstrip() + "\n")

    # -- helpers -------------------------------------------------------------

    def _evolution_refusal(self, runtime: Any, action: str, inputs: dict) -> str:
        """When an EvolutionPolicy governs this runtime (enable_evolution
        was called), growing or shrinking the toolset is a `tool_adapter`
        change and the policy's verdict applies here too -- the
        tools-that-create-tools loop never outruns the fence. With no
        policy enabled, acquisition behaves exactly as before: the fence
        exists only once the operator raises it. The refusal returns to
        the model as text, like a tool failure, and goes on the record."""
        policy = getattr(runtime, "evolution_policy", None)
        if policy is None:
            return ""
        refusal = policy.refusal("tool_adapter")
        if refusal is None:
            return ""
        self._record(runtime, action, inputs, f"refused by the evolution policy -- {refusal}")
        return f"Refused by the evolution policy: {refusal}"

    @staticmethod
    def _store(runtime: Any) -> tuple:
        """Where this runtime's acquired tools persist, and how to reach
        it. Inside its Sandbox when one confines this runtime -- so a
        self-created tool's file footprint stays inside the pod's isolated
        workspace, never the wider host, the same blast radius every other
        agent-writable file tool (`write_file`, `run_shell`) already keeps
        to. Otherwise the stack-level `.ear/tools.md` a plain host path
        names, exactly as an unsandboxed runtime persists today."""
        sandbox = getattr(runtime, "sandbox", None)
        if sandbox is not None:
            return sandbox, ".ear/tools.md"
        return None, getattr(runtime, "tools_path", None)

    @staticmethod
    def _read(sandbox: Optional[Any], relpath: str) -> str:
        if sandbox is not None:
            return sandbox.read_text(relpath)
        return Path(relpath).read_text(encoding="utf-8")

    @staticmethod
    def _write(sandbox: Optional[Any], relpath: str, text: str) -> None:
        if sandbox is not None:
            sandbox.write_text(relpath, text)
            return
        destination = Path(relpath)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")

    @staticmethod
    def _exists(sandbox: Optional[Any], relpath: str) -> bool:
        if sandbox is not None:
            return sandbox.exists(relpath)
        return Path(relpath).exists()

    @staticmethod
    def _bound(runtime: Any) -> list:
        return runtime.tool_binder.bound_tools(runtime)

    @staticmethod
    def _skills(runtime: Any) -> list:
        from .tool_binder import ToolBinder

        return list(ToolBinder._skills(getattr(runtime, "processes", []) or []).values())

    @staticmethod
    def _record(runtime: Any, action: str, inputs: dict, output: str) -> None:
        log = getattr(runtime, "reasoning_log", None)
        if log is None:
            return
        log.record(stage="acquisition", inputs={"action": action, **inputs}, output=output)
