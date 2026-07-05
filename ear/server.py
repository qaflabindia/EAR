"""Server -- EAR as a running control-plane service: a small HTTP front
door onto the Kernel, so a fleet of runtime instances can be created,
driven and observed over the network.

It is the server-side face of the same picture: the Kernel is the
scheduler, each instance is a process with its own sandbox, memory and
trail, and the Server is the syscall interface -- create an instance,
submit work to it, ask how it is doing. Zero dependencies: the whole thing
is the standard library's threading HTTP server speaking JSON, with the
Kernel running the work behind it.

Solid by construction, not by afterthought:

- **Auth.** A bearer token, read from `EAR_SERVER_TOKEN` (never hardcoded)
  and compared in constant time; every request, including a health check,
  is refused without it. Unset means open, and the server says so loudly
  on start -- a development convenience you opt into, not a silent default.
- **Confinement.** Loading a stack is confined under a configured
  `stacks_root`; a path that escapes it is refused, the same discipline as
  the sandbox. No `stacks_root`, no loading arbitrary paths from the wire.
- **Resilience.** Request bodies are capped, malformed JSON is a 400 not a
  crash, and every handler is wrapped so one bad request can never take the
  server down. The routing itself is a pure function -- `handle(method,
  path, body) -> (status, payload)` -- so the whole API is testable without
  opening a socket.
- **Approval without a shared filesystem.** `Exchange`'s `approval.md`
  file-drop convention assumes the human and the runtime share a disk --
  not true across a network boundary. `POST /instances/{name}/approve`
  is the same release, spoken over the wire: it resubmits the instance's
  last intent with a `Verdict`/`Approver`/note attached, exactly as a
  second `Exchange.run()` would once the file appears.

    from ear import Server
    server = Server(stacks_root="./stacks", port=8080)
    server.serve()                     # blocking; Ctrl-C to stop

    python -m ear.server --stacks ./stacks --port 8080
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional, Union
from urllib.parse import parse_qsl

from .approval import Approval
from .intent import Intent
from .kernel import Kernel
from .llm import _ssl_context
from .section import normalize

MAX_BODY_BYTES = 1_048_576  # 1 MiB -- a control-plane request is small


class _ClientError(Exception):
    """A request the client got wrong -- surfaced as its HTTP status, never
    a 500."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(message)


@dataclass
class Server:
    """The control plane: an HTTP service over a Kernel."""

    kernel: Kernel = field(default_factory=Kernel)
    stacks_root: Optional[Union[str, Path]] = None
    host: str = "127.0.0.1"
    port: int = 8080
    token: Optional[str] = None
    # A stack can *declare* a tool in memory.md's Tools section, but a
    # declaration alone has no executable behind it -- the runtime never
    # auto-runs a `command` string, on purpose (see tool.py). When the
    # caller creating an instance is itself a remote system whose own
    # capabilities the stack's tools are meant to reach (rather than a
    # human hand-writing a Python handler), `bridge_url`/`bridge_token`
    # give every declared-but-unbound tool a generic handler: forward the
    # call as JSON, return the response text, same trail record as any
    # other tool. Set once for the whole server -- one deployment forwards
    # to one system.
    bridge_url: Optional[str] = None
    bridge_token: Optional[str] = None
    started_at: float = field(default_factory=time.monotonic)
    _httpd: Any = None
    _thread: Any = None
    # The most recent intent submitted per instance -- kept so `/approve` can
    # resubmit the same intent with a human's verdict attached. There is no
    # shared filesystem across the network boundary for the `approval.md`
    # file-drop convention `Exchange` uses, so the server is the one that
    # remembers what a parked cycle was actually asked to do.
    _last_intents: dict = field(default_factory=dict)
    # Per-instance context (e.g. org/task/session identifiers) merged into
    # every bridged tool call for that instance, so the remote system can
    # authorise and scope the call without EAR needing to understand what
    # any of those identifiers mean.
    _bridge_contexts: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token is None:
            self.token = os.environ.get("EAR_SERVER_TOKEN")
        if self.bridge_url is None:
            self.bridge_url = os.environ.get("EAR_BRIDGE_URL")
        if self.bridge_token is None:
            self.bridge_token = os.environ.get("EAR_BRIDGE_TOKEN")
        if self.stacks_root is not None:
            self.stacks_root = Path(self.stacks_root).resolve()

    # -- routing (a pure function -- the whole API, no socket) ----------------

    def handle(self, method: str, path: str, body: Optional[dict] = None) -> tuple:
        """Route one request to a (status, payload) pair. Pure and
        synchronous -- the socket layer only translates HTTP to and from
        this. A client mistake is its own status; anything unexpected is a
        500 with the reason, never an unhandled crash."""
        path, _, query = path.partition("?")
        parts = [segment for segment in path.strip("/").split("/") if segment]
        # GET requests carry no body over most HTTP clients (Node's fetch
        # refuses one outright) -- merge the query string in so a caller can
        # pass e.g. ?limit=50 without the server-side handlers needing to
        # know or care which side of the request a parameter arrived on.
        body = {**parse_qs_flat(query), **(body or {})}
        try:
            if method == "GET" and parts == ["health"]:
                return 200, self._health()
            if method == "GET" and parts == ["kernel"]:
                return 200, self.kernel.snapshot()
            if parts == ["instances"]:
                if method == "GET":
                    return 200, {"instances": self._instances()}
                if method == "POST":
                    return self._create(body)
            if len(parts) >= 2 and parts[0] == "instances":
                name = parts[1]
                if len(parts) == 2 and method == "DELETE":
                    return self._delete(name)
                if len(parts) == 3 and method == "POST" and parts[2] == "submit":
                    return self._submit(name, body)
                if len(parts) == 3 and method == "POST" and parts[2] == "approve":
                    return self._approve(name, body)
                if len(parts) == 3 and method == "GET" and parts[2] == "status":
                    return 200, self._status(name)
                if len(parts) == 3 and method == "GET" and parts[2] == "decision":
                    return 200, self._decision(name)
                if len(parts) == 3 and method == "GET" and parts[2] == "trail":
                    return 200, self._trail(name, body)
            return 404, {"error": "not found", "method": method, "path": path}
        except _ClientError as mistake:
            return mistake.status, {"error": mistake.message}
        except Exception as failure:  # noqa: BLE001 -- one bad request never takes the server down
            return 500, {"error": f"{type(failure).__name__}: {failure}"}

    # -- endpoints ------------------------------------------------------------

    def _health(self) -> dict:
        return {
            "status": "ok",
            "instances": len(self.kernel.instances),
            "pending": self.kernel.pending,
            "dispatched": len(self.kernel.history),
            "running": self.kernel.running,
            "uptime_s": round(time.monotonic() - self.started_at, 1),
        }

    def _instances(self) -> list:
        return [self._summary(name, runtime) for name, runtime in list(self.kernel.instances.items())]

    def _create(self, body: dict) -> tuple:
        name = _require(body, "name")
        if name in self.kernel.instances:
            raise _ClientError(409, f"instance '{name}' already exists")
        stack = body.get("stack")
        files = body.get("files")
        if files:
            runtime = self._load_inline_stack(name, files)
        elif stack:
            runtime = self._load_stack(name, str(stack))
        else:
            from .runtime import Runtime

            runtime = Runtime(name=name)
        self._bind_bridge_tools(name, runtime, body.get("bridge_context") or {})
        self.kernel.register(name, runtime)
        return 201, {"instance": name, "from_stack": bool(stack or files)}

    def _delete(self, name: str) -> tuple:
        self._instance(name)
        self.kernel.remove(name)
        return 200, {"instance": name, "removed": True}

    def _submit(self, name: str, body: dict) -> tuple:
        runtime = self._instance(name)
        text = _require(body, "intent")
        intent = Intent(text=str(text), context=dict(body.get("context") or {}))
        self._apply_credentials(runtime, body.get("credentials"))
        every = body.get("every")
        task = self.kernel.submit(
            name,
            intent,
            goal=body.get("goal"),
            every=float(every) if every is not None else None,
            delay=float(body.get("delay") or 0.0),
        )
        # Remembered so `/approve` can resubmit this same intent with a
        # verdict attached if it parks on an approval gate.
        self._last_intents[name] = intent
        return 202, {"task_id": task.id, "instance": name, "recurring": task.recurring}

    def _approve(self, name: str, body: dict) -> tuple:
        self._instance(name)
        intent = self._last_intents.get(name)
        if intent is None:
            raise _ClientError(409, f"instance '{name}' has no pending intent to approve")
        verdict = _require(body, "verdict")
        if isinstance(verdict, str):
            verdict = verdict.strip().lower() in ("approved", "approve", "true", "yes")
        approval = Approval(verdict=bool(verdict), approver=str(body.get("approver") or ""), note=str(body.get("note") or ""))
        task = self.kernel.submit(name, intent, approval=approval)
        return 202, {"task_id": task.id, "instance": name, "verdict": approval.verdict}

    @staticmethod
    def _apply_credentials(runtime: Any, credentials: Optional[dict]) -> None:
        """Override -- or, when the stack declared no model at all, supply --
        this instance's model binding with a per-request credential. One EAR
        server process serves many tenants' personas concurrently, so a key
        resolved once from `os.environ` (the single-tenant default, see
        `ModelBinding.resolve_api_key`) does not scale here, and neither does
        `loader.py`'s caution about not attaching a binding it can't yet
        resolve a key for (exactly right for a stack loaded once on a
        machine with one key; not for a stack whose key arrives per-request
        instead). A caller naming its own `provider`/`model` also sidesteps
        memory.md's prose-guessing entirely -- useful for a provider name
        EAR's own heuristics don't recognise. Resetting the cached `lm`
        makes a rotated key take effect on the next `activate()` rather than
        sticking to the first one seen."""
        if not credentials:
            return
        binding = getattr(runtime, "model_binding", None)
        provider, model = credentials.get("provider"), credentials.get("model")
        if binding is None and provider and model:
            from .model_binding import ModelBinding

            binding = ModelBinding(provider=str(provider), model=str(model), api_base=credentials.get("api_base"))
            runtime.model_binding = binding
        if binding is None:
            return
        api_key = credentials.get("api_key")
        if api_key:
            binding.api_key = str(api_key)
            binding.lm = None

    def _bind_bridge_tools(self, name: str, runtime: Any, bridge_context: dict) -> None:
        """Give every tool this stack declared -- but left unbound -- a
        generic handler that forwards the call to `bridge_url` as JSON.
        Declaring a tool never wires it to code by itself (see tool.py); this
        is that wiring, for the one case where "the code" is a remote system
        reachable over HTTP rather than a Python function in this process."""
        if not self.bridge_url:
            return
        self._bridge_contexts[name] = dict(bridge_context)
        strategy = getattr(runtime, "strategy", None)
        declared = getattr(strategy, "tools", None) or []
        binder = runtime.tool_binder
        for tool in declared:
            if normalize(tool.name) in binder.bindings:
                continue  # a stack-specific handler already claimed this name
            binder.bind(tool.name, self._bridge_handler(name, tool.name))

    def _bridge_handler(self, instance_name: str, tool_name: str) -> Callable[..., Any]:
        def handler(**kwargs: Any) -> str:
            payload = {
                "tool": tool_name,
                "input": kwargs,
                "context": self._bridge_contexts.get(instance_name, {}),
            }
            return _call_bridge(str(self.bridge_url), self.bridge_token, payload)

        handler.__name__ = f"bridge_{tool_name}"
        return handler

    def _status(self, name: str) -> dict:
        runtime = self._instance(name)
        summary = self._summary(name, runtime)
        summary["pending"] = sum(1 for task in list(self.kernel.queue) if task.instance == name)
        return summary

    def _decision(self, name: str) -> dict:
        runtime = self._instance(name)
        deliberations = [record for record in runtime.reasoning_log.records if record.stage == "deliberation"]
        return {"instance": name, "decision": deliberations[-1].output if deliberations else ""}

    def _trail(self, name: str, body: dict) -> dict:
        runtime = self._instance(name)
        limit = max(1, min(int(body.get("limit", 20)), 200))
        records = runtime.reasoning_log.records[-limit:]
        return {
            "instance": name,
            "records": [
                {
                    "cycle": r.cycle,
                    "stage": r.stage,
                    # Tool calls carry their name/arguments/duration in `inputs`
                    # (see ToolBinder._logged) -- surfaced here so a remote
                    # caller can render tool_requested/tool_completed without
                    # parsing the truncated free-text `output`.
                    "tool": r.inputs.get("tool") if r.stage == "tool" else None,
                    "ok": not r.output.startswith(("FAILED", "BLOCKED")) if r.stage == "tool" else None,
                    "output": r.output[:200],
                    "model": r.model,
                }
                for r in records
            ],
        }

    # -- helpers --------------------------------------------------------------

    def _instance(self, name: str) -> Any:
        runtime = self.kernel.instances.get(name)
        if runtime is None:
            raise _ClientError(404, f"no instance '{name}'")
        return runtime

    def _load_stack(self, name: str, stack: str) -> Any:
        if self.stacks_root is None:
            raise _ClientError(400, "loading a stack requires the server's stacks_root to be configured")
        target = (Path(self.stacks_root) / stack).resolve()
        root = Path(self.stacks_root)
        if target != root and root not in target.parents:
            raise _ClientError(400, f"stack '{stack}' escapes the stacks root")
        if not target.is_dir():
            raise _ClientError(404, f"no stack at '{stack}'")
        from .loader import load_runtime

        return load_runtime(target, name=name)

    def _load_inline_stack(self, name: str, files: dict) -> Any:
        """Build a stack from file contents sent inline in the request body,
        rather than requiring a caller (e.g. a LENS server in a different
        process/pod) to share a populated filesystem with this one. Written
        under the server's own `stacks_root` -- still confined by it, still
        loaded through the same `load_runtime` every on-disk stack uses --
        just populated over the wire instead of pre-existing on disk."""
        if self.stacks_root is None:
            raise _ClientError(400, "loading a stack requires the server's stacks_root to be configured")
        if not isinstance(files, dict) or not files:
            raise _ClientError(400, "'files' must be a non-empty object of filename -> content")
        from .loader import _FILE_CANDIDATES, load_runtime

        known = {filename for candidates in _FILE_CANDIDATES.values() for filename in candidates}
        target = (Path(self.stacks_root) / ".inline" / name).resolve()
        root = Path(self.stacks_root)
        if root not in target.parents:
            raise _ClientError(400, f"instance name '{name}' escapes the stacks root")
        for filename, content in files.items():
            if filename not in known:
                raise _ClientError(400, f"unknown stack file '{filename}' -- known files: {', '.join(sorted(known))}")
            if not isinstance(content, str):
                raise _ClientError(400, f"content for '{filename}' must be a string")
        target.mkdir(parents=True, exist_ok=True)
        for filename, content in files.items():
            (target / filename).write_text(content, encoding="utf-8")
        return load_runtime(target, name=name)

    @staticmethod
    def _summary(name: str, runtime: Any) -> dict:
        """A lightweight status for one instance -- health and progress from
        the trail, without building the whole dashboard body."""
        from .dashboard import _classify, _cycle_status, _freshness, _integrity

        log = runtime.reasoning_log
        records = log.records
        cycles = sorted({record.cycle for record in records})
        blocked = sum(1 for cycle in cycles if _cycle_status(log.for_cycle(cycle)) == "bad")
        pending = sum(
            1
            for record in records
            if record.stage in ("approval", "escalation")
            and ("pending" in record.output.lower() or "escalated" in record.output.lower())
        )
        failed = sum(1 for record in records if record.stage == "retry" and "exhausted" in record.output.lower())
        integrity = _integrity(log)
        status, reason = _classify(integrity, list(getattr(log, "export_errors", []) or []), failed, pending, blocked)
        last = max((record.timestamp for record in records), default=None)
        strategy = getattr(runtime, "strategy", None)
        tokens = sum(record.input_tokens + record.output_tokens for record in records)
        return {
            "instance": name,
            "status": status,
            "reason": reason,
            "freshness": _freshness(last, __import__("datetime").datetime.now(__import__("datetime").timezone.utc)),
            "cycles": len(cycles),
            "tokens": tokens,
            "dollars": strategy.dollars(
                sum(r.input_tokens for r in records), sum(r.output_tokens for r in records)
            )
            if strategy is not None
            else None,
            "latency_ms": sum(record.latency_ms for record in records),
            "sandboxed": getattr(runtime, "sandbox", None) is not None,
        }

    # -- the socket layer -----------------------------------------------------

    def start(self) -> "Server":
        """Start the background kernel and serve in a daemon thread."""
        self.kernel.start()
        self._httpd = ThreadingHTTPServer((self.host, self.port), _make_handler(self))
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="ear-server", daemon=True)
        self._thread.start()
        return self

    @property
    def address(self) -> tuple:
        return self._httpd.server_address if self._httpd is not None else (self.host, self.port)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        self.kernel.stop()

    def serve(self) -> None:  # pragma: no cover - blocking
        self.kernel.start()
        self._httpd = ThreadingHTTPServer((self.host, self.port), _make_handler(self))
        guard = "token-protected" if self.token else "OPEN — set EAR_SERVER_TOKEN to require auth"
        print(f"EAR server on http://{self.host}:{self.port}  ({guard})")
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def __enter__(self) -> "Server":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def parse_qs_flat(query: str) -> dict:
    """A URL query string as a flat dict (last value wins for a repeated
    key) -- every EAR control-plane parameter is scalar, so there is no
    need for `parse_qs`'s list-per-key shape."""
    return dict(parse_qsl(query))


def _require(body: dict, key: str) -> Any:
    if key not in body or body[key] in (None, ""):
        raise _ClientError(400, f"missing required field '{key}'")
    return body[key]


def _call_bridge(url: str, token: Optional[str], payload: dict) -> str:
    """POST one bridged tool call and return the remote system's text
    response. Raises loudly on any failure -- transport, auth, or an
    error response -- so the failure reaches the model as tool-call text
    like any other tool failure (ToolBinder._logged wraps every handler)."""
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, context=_ssl_context(), timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"bridge call to {url} failed ({error.code}): {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"bridge call to {url} failed: {error}") from error
    except ValueError as error:
        raise RuntimeError(f"bridge call to {url} returned malformed JSON: {error}") from error
    if isinstance(body, dict) and not body.get("ok", True):
        error = body.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        raise RuntimeError(str(message or "bridge call reported failure"))
    return json.dumps(body.get("output", body)) if isinstance(body, dict) else str(body)


def _make_handler(server: Server):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _authed(self) -> bool:
            if not server.token:
                return True
            presented = self.headers.get("Authorization", "")
            return hmac.compare_digest(presented, f"Bearer {server.token}")

        def _dispatch(self, method: str) -> None:
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                return self._send(413, {"error": "request body too large"})
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                return self._send(400, {"error": "invalid JSON body"})
            if not isinstance(body, dict):
                return self._send(400, {"error": "JSON body must be an object"})
            status, payload = server.handle(method, self.path, body)
            self._send(status, payload)

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:  # keep the terminal quiet
            return

    return Handler


def _main() -> None:  # pragma: no cover - CLI entry
    import argparse

    parser = argparse.ArgumentParser(prog="ear.server", description="Run the EAR control-plane server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--stacks", default=None, help="root directory of loadable stacks")
    args = parser.parse_args()
    Server(host=args.host, port=args.port, stacks_root=args.stacks).serve()


if __name__ == "__main__":  # pragma: no cover
    _main()
