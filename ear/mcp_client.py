"""McpClient -- EAR's own MCP (Model Context Protocol) client, spoken from
the Python standard library alone.

MCP is an open JSON-RPC 2.0 protocol; EAR speaks it directly, no SDK. This
module launches a declared server as a subprocess and talks to it over
stdio -- line-delimited JSON-RPC on the child's stdin/stdout -- performing
the handshake (`initialize` / `notifications/initialized`), listing its
tools (`tools/list`), and calling them (`tools/call`). The transport is a
few dozen lines because the protocol *is* the spec, and the spec is JSON
over pipes.

A connected server's tools are surfaced to the runtime as ordinary
BoundTools (see `Runtime.connect_mcp`): they run through the same logged
handler as any native tool, so every MCP call is a `tool` trail record
with its arguments, result and duration, obeys the same tool-loop budget,
and is judged by the same tool-scoped policies. A server that hangs or
answers with malformed JSON fails loudly as an `McpError`, never silently
-- and, wrapped by the binder, that failure returns to the model as text
like any other tool failure.

The server stays **declared in memory.md** exactly as before; connecting
one is the runtime reaching out to what the author already named, never a
capability that appears from nowhere.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

MCP_PROTOCOL_VERSION = "2024-11-05"
DEFAULT_TIMEOUT = 30.0


class McpError(RuntimeError):
    """An MCP server call failed -- launch, handshake, transport, timeout,
    a JSON-RPC error, or a malformed response. Loud by design: a server
    that misbehaves is a fact the runtime surfaces, never swallows."""


@dataclass
class McpTool:
    """One tool a connected server advertises: its name, the description
    the model reads, and the input parameter names from its JSON schema."""

    name: str
    description: str
    parameters: list[str] = field(default_factory=list)


@dataclass
class McpClient:
    """A live connection to one MCP server over stdio. `connect()` launches
    the command and handshakes; `list_tools()` and `call_tool()` speak
    JSON-RPC; `close()` shuts the subprocess down. Not reused across
    servers -- one client, one server, one process."""

    command: list[str]
    timeout: float = DEFAULT_TIMEOUT
    process: Optional[subprocess.Popen] = None
    _next_id: int = 0
    _lock: Any = field(default_factory=threading.Lock)

    # -- lifecycle -------------------------------------------------------------

    def connect(self) -> "McpClient":
        """Launch the server and perform the MCP handshake. Raises
        `McpError` if the command cannot start or the server does not
        complete `initialize`."""
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except (OSError, ValueError) as error:
            raise McpError(f"could not launch MCP server {self.command!r}: {error}") from error
        self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "ear", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized", {})
        return self

    def close(self) -> None:
        """Shut the server down -- close its stdin, then wait briefly and
        kill if it lingers. Idempotent."""
        if self.process is None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            self.process.kill()
        finally:
            self.process = None

    def __enter__(self) -> "McpClient":
        return self.connect()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- MCP methods -----------------------------------------------------------

    def list_tools(self) -> list[McpTool]:
        """The server's advertised tools (`tools/list`), each with the
        parameter names read from its JSON input schema."""
        result = self._request("tools/list", {})
        tools: list[McpTool] = []
        for entry in result.get("tools", []) or []:
            schema = entry.get("inputSchema") or {}
            parameters = list((schema.get("properties") or {}).keys())
            tools.append(
                McpTool(
                    name=str(entry.get("name", "")),
                    description=str(entry.get("description", "") or entry.get("name", "")),
                    parameters=parameters,
                )
            )
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke a server tool (`tools/call`) and return its text content.
        A JSON-RPC error, an `isError` result, or a malformed reply all
        raise `McpError`, so the caller (and, through the binder, the
        model) always learns the truth."""
        result = self._request("tools/call", {"name": name, "arguments": dict(arguments)})
        blocks = result.get("content") or []
        text = "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        if result.get("isError"):
            raise McpError(f"MCP tool {name!r} reported an error: {text or result}")
        return text

    # -- JSON-RPC over stdio ---------------------------------------------------

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """One JSON-RPC request/response round trip. Serialized by a lock
        so concurrent tool calls never interleave on the pipe."""
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise McpError(f"MCP server {self.command!r} is not connected")
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            self._write(message)
            return self._read_response(request_id, method)

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise McpError(f"MCP server {self.command!r} is not connected")
        with self._lock:
            self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _write(self, message: dict[str, Any]) -> None:
        assert self.process is not None and self.process.stdin is not None
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise McpError(f"MCP server {self.command!r} closed the pipe: {error}") from error

    def _read_response(self, request_id: int, method: str) -> dict[str, Any]:
        """Read lines until the response with our id arrives, skipping any
        notifications or unrelated messages the server interleaves. A dead
        pipe or a timeout raises loudly rather than blocking forever."""
        assert self.process is not None and self.process.stdout is not None
        result: dict[str, Any] = {}
        error_holder: dict[str, Any] = {}

        def pump() -> None:
            while True:
                line = self.process.stdout.readline()  # type: ignore[union-attr]
                if not line:
                    error_holder["error"] = McpError(
                        f"MCP server {self.command!r} closed before answering {method!r}"
                    )
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    continue  # a non-JSON log line on stdout is not our concern
                if not isinstance(message, dict) or message.get("id") != request_id:
                    continue
                if "error" in message:
                    error_holder["error"] = McpError(
                        f"MCP server {self.command!r} returned an error for {method!r}: {message['error']}"
                    )
                    return
                result.update(message.get("result") or {})
                error_holder["done"] = True
                return

        worker = threading.Thread(target=pump, daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            raise McpError(f"MCP server {self.command!r} did not answer {method!r} within {self.timeout}s")
        if "error" in error_holder:
            raise error_holder["error"]
        return result


def command_words(command: str) -> list[str]:
    """Split a declared launch command into argv, honouring quotes -- the
    same `shlex` the shell uses, so an author writes a natural command
    line, not a JSON array."""
    import shlex

    return shlex.split(command)
