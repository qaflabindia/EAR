"""CommandCentreServer -- package one acc-skills command centre as a native
MCP server, so a centre runs out-of-process and any EAR runtime attaches to
it with `Runtime.connect_mcp`.

The framework offers each command centre two bindings (architecture section
3.4). *In-process*, a centre's operations are deterministic Skill handlers
inside the runtime. *Out-of-process* -- the deployment-isolation option, and
the default for the operational plane -- the centre is its own MCP server:
another runtime reaches it over stdio JSON-RPC, the same protocol EAR's
native `McpClient` already speaks, and the centre's tools ride the runtime's
one tool loop with the same budgets, records and tool-scoped policies as any
other tool.

This module exposes a centre's recurring script pentad
(`init / load_state / update_state / evaluate / audit`) as MCP tools over
the Phase-1 `CommandCentreBackend` and `Constitution`:

    list_state                 the centre's state entries (names)
    load_state(name)           one state entry, as JSON
    update_state(name, value)  write one state entry (value is JSON)
    evaluate(context)          judge the centre's constitution against a context
    audit(entry)               append one line to the centre's ledger

`evaluate` runs the constitution's *deterministic* fallbacks (the server is
a plain subprocess with no model bound), so an out-of-process centre still
enforces its mechanically checkable rules; rules with no fallback report as
not-applicable, never as a silent pass. Standard library only: the protocol
is JSON over pipes, and that is the whole dependency.

Run one directly:

    python -m ear.mcp_command_centre path/to/acc-skills/afcc

and connect to it from a stack by declaring it in memory.md's MCP section,
exactly like any other MCP server.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from .enterprise import CommandCentre, Verdict

MCP_PROTOCOL_VERSION = "2024-11-05"

_TOOLS = [
    {
        "name": "list_state",
        "description": "List the command centre's persistent state entries by name.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "load_state",
        "description": "Read one state entry, returned as JSON.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "update_state",
        "description": "Write one state entry. 'value' is a JSON document.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
            "required": ["name", "value"],
        },
    },
    {
        "name": "evaluate",
        "description": (
            "Judge the centre's constitution against a context (a JSON object of "
            "facts) using its deterministic fallbacks; returns each rule's verdict "
            "and whether the context complies."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"context": {"type": "string"}},
            "required": ["context"],
        },
    },
    {
        "name": "audit",
        "description": "Append one entry (a JSON object) to the centre's audit ledger.",
        "inputSchema": {
            "type": "object",
            "properties": {"entry": {"type": "string"}},
            "required": ["entry"],
        },
    },
]


@dataclass
class CommandCentreServer:
    """A command centre served as MCP tools. The dispatch (`call`) is pure
    and synchronous, so it is testable directly; `serve` wraps it in the
    stdio JSON-RPC loop `McpClient` connects to."""

    centre: CommandCentre

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "CommandCentreServer":
        return cls(centre=CommandCentre.load(directory))

    def tools(self) -> list[dict]:
        return _TOOLS

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Dispatch one tool call, returning (text, is_error). Every failure
        is a loud, textual error returned to the caller -- the same contract
        EAR's tool loop already expects from any tool."""
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return f"unknown tool: {name}", True
            return handler(arguments), False
        except Exception as error:  # a tool failure returns to the model as text
            return f"{type(error).__name__}: {error}", True

    # -- the tool handlers --------------------------------------------------

    def _tool_list_state(self, _arguments: dict[str, Any]) -> str:
        return "\n".join(self.centre.state.list()) or "(no state)"

    def _tool_load_state(self, arguments: dict[str, Any]) -> str:
        return self.centre.state.read(str(arguments["name"]))

    def _tool_update_state(self, arguments: dict[str, Any]) -> str:
        name = str(arguments["name"])
        self.centre.state.write_json(name, json.loads(arguments["value"]))
        return f"updated state '{name}'"

    def _tool_evaluate(self, arguments: dict[str, Any]) -> str:
        context = json.loads(arguments["context"])
        if not isinstance(context, dict):
            raise ValueError("'context' must be a JSON object of facts")
        results = []
        for rule in self.centre.constitution._ordered():
            policy = rule.to_policy()
            complies, rationale = policy.judge(model_binding=None, **context)
            results.append(
                {
                    "rule": rule.policy_name,
                    "verdict": rule.verdict,
                    "complies": complies,
                    "blocking": Verdict.blocks(rule.verdict),
                    "rationale": rationale,
                }
            )
        violations = [r for r in results if not r["complies"] and r["blocking"]]
        return json.dumps(
            {"passed": not violations, "violations": [r["rule"] for r in violations], "rules": results},
            indent=2,
        )

    def _tool_audit(self, arguments: dict[str, Any]) -> str:
        entry = json.loads(arguments["entry"])
        path = self.centre.state.audit_trail_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return "recorded to the audit ledger"

    # -- the stdio JSON-RPC loop -------------------------------------------

    def serve(self, stdin: Any = None, stdout: Any = None) -> None:
        """Serve MCP over line-delimited JSON-RPC on stdin/stdout until the
        input closes -- `initialize`, `tools/list`, `tools/call`. The same
        minimal surface EAR's `McpClient` drives."""
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            method = message.get("method")
            if method == "notifications/initialized":
                continue
            request_id = message.get("id")
            if method == "initialize":
                result: Any = {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self.centre.slug, "version": "0.1.0"},
                }
            elif method == "tools/list":
                result = {"tools": self.tools()}
            elif method == "tools/call":
                params = message.get("params") or {}
                text, is_error = self.call(params.get("name"), params.get("arguments") or {})
                result = {"content": [{"type": "text", "text": text}], "isError": is_error}
            else:
                self._send(stdout, {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "method not found"}})
                continue
            self._send(stdout, {"jsonrpc": "2.0", "id": request_id, "result": result})

    @staticmethod
    def _send(stdout: Any, message: dict) -> None:
        stdout.write(json.dumps(message) + "\n")
        stdout.flush()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.stderr.write("usage: python -m ear.mcp_command_centre <command-centre-directory>\n")
        return 2
    CommandCentreServer.load(argv[1]).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
