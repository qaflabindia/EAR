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

Exactly one thread ever reads the server's stdout: the background `_pump`
loop `connect()` starts, for the connection's whole lifetime. Each
in-flight request registers a one-slot queue keyed by its JSON-RPC id, and
`_pump` routes whatever it reads to the matching queue (or drops it if
none is waiting -- a response that outlives its caller's timeout, or a
notification, is not an error). A single, persistent reader is what makes
a client-side timeout safe: nothing ever spawns a *second* reader that
could race the pump for the same bytes and silently steal a line meant for
a later call.

The server stays **declared in memory.md** exactly as before; connecting
one is the runtime reaching out to what the author already named, never a
capability that appears from nowhere.
"""

from __future__ import annotations

import json
import queue
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
    the command, starts the one background reader for the connection's
    life, and handshakes; `list_tools()` and `call_tool()` speak JSON-RPC;
    `close()` shuts the subprocess down. Not reused across servers -- one
    client, one server, one process."""

    command: list[str]
    timeout: float = DEFAULT_TIMEOUT
    process: Optional[subprocess.Popen] = None
    _next_id: int = field(default=0, init=False, repr=False)
    _call_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _write_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _pending_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _pending: dict = field(default_factory=dict, init=False, repr=False)
    _reader: Optional[threading.Thread] = field(default=None, init=False, repr=False)

    # -- lifecycle -------------------------------------------------------------

    def connect(self) -> "McpClient":
        """Launch the server, start the background reader, and perform the
        MCP handshake. Raises `McpError` if the command cannot start or the
        server does not complete `initialize`."""
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
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
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
        kill if it lingers. Idempotent. The reader thread exits on its own
        once the pipe closes."""
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
        """One JSON-RPC request/response round trip. `_call_lock` keeps
        calls to one full round trip at a time -- simple to reason about --
        but it is the single persistent `_pump` reader, not this lock, that
        actually prevents a timed-out call from racing a later one for the
        same bytes: there is never a second thread reading stdout to race
        with."""
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise McpError(f"MCP server {self.command!r} is not connected")
        with self._call_lock:
            with self._pending_lock:
                self._next_id += 1
                request_id = self._next_id
                inbox: "queue.Queue[Any]" = queue.Queue(maxsize=1)
                self._pending[request_id] = (method, inbox)
            try:
                self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
                try:
                    outcome = inbox.get(timeout=self.timeout)
                except queue.Empty:
                    raise McpError(f"MCP server {self.command!r} did not answer {method!r} within {self.timeout}s")
            finally:
                with self._pending_lock:
                    self._pending.pop(request_id, None)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise McpError(f"MCP server {self.command!r} is not connected")
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _write(self, message: dict[str, Any]) -> None:
        assert self.process is not None and self.process.stdin is not None
        with self._write_lock:
            try:
                self.process.stdin.write(json.dumps(message) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise McpError(f"MCP server {self.command!r} closed the pipe: {error}") from error

    def _pump(self) -> None:
        """The connection's one and only stdout reader, for its entire
        lifetime. Reads line by line and routes each JSON-RPC response to
        whichever `_request` call is still waiting on its id -- dropping
        anything nobody is waiting on any more (a response that outlives
        its caller's timeout, a notification, a stray log line). On EOF or
        a read failure, every still-pending call is woken with an error
        instead of hanging until its own timeout."""
        stream = self.process.stdout  # type: ignore[union-attr]
        try:
            while True:
                line = stream.readline()
                if not line:
                    self._drain_pending("the server closed the connection")
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    continue  # a non-JSON log line on stdout is not our concern
                if not isinstance(message, dict) or "id" not in message:
                    continue  # a notification, or not a response we're tracking
                self._deliver(message)
        except OSError as error:
            self._drain_pending(f"the reader failed: {error}")

    def _deliver(self, message: dict[str, Any]) -> None:
        with self._pending_lock:
            entry = self._pending.get(message.get("id"))
        if entry is None:
            return  # nobody is waiting on this id any more
        method, inbox = entry
        if "error" in message:
            outcome: Any = McpError(f"MCP server {self.command!r} returned an error for {method!r}: {message['error']}")
        else:
            outcome = message.get("result") or {}
        try:
            inbox.put_nowait(outcome)
        except queue.Full:
            pass  # the caller already gave up and this id was reclaimed

    def _drain_pending(self, reason: str) -> None:
        with self._pending_lock:
            entries = list(self._pending.values())
            self._pending.clear()
        for method, inbox in entries:
            error = McpError(f"MCP server {self.command!r} closed before answering {method!r}: {reason}")
            try:
                inbox.put_nowait(error)
            except queue.Full:
                pass


def command_words(command: str) -> list[str]:
    """Split a declared launch command into argv, honouring quotes -- the
    same `shlex` the shell uses, so an author writes a natural command
    line, not a JSON array."""
    import shlex

    return shlex.split(command)
