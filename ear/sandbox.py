"""Sandbox -- bound how long a Tool's handler is allowed to run.

DeerFlow isolates tool execution behind Docker/Kubernetes sandbox
providers -- real OS-level isolation, with a provisioner service and a
managed filesystem layout. That is deployment infrastructure a library has
no business bundling (EAR stays a library, not a service), so this is
deliberately smaller and honest about what it does: a Sandbox is a seam a
Tool's handler runs through (`sandbox.run(handler, **kwargs)`), and the one
guarantee shipped here is a wall-clock timeout, using only the standard
library. It does not isolate memory, the filesystem or the network -- it
only stops a runaway handler from blocking a cycle forever. Swap in a
heavier Sandbox (a subprocess, a container) for real isolation; the seam
is what matters, not this implementation of it.

`InProcessSandbox` -- the default -- runs the handler directly, so a Tool
with no sandbox configured behaves exactly as it always has."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable


class SandboxError(RuntimeError):
    """Raised when a Sandbox cannot complete a handler's execution."""


class SandboxTimeout(SandboxError):
    """Raised when a handler exceeds a Sandbox's wall-clock timeout.

    Python has no safe way to force-kill a thread, so the handler may still
    be running in the background after this is raised -- a TimeoutSandbox
    bounds how long the *caller* waits, not how long the handler actually
    executes. Use a process- or container-based Sandbox if a handler must
    be forcibly stopped."""


@dataclass
class InProcessSandbox:
    """The default Sandbox: run the handler directly, in-process, with no
    isolation and no timeout -- today's behaviour, unchanged."""

    def run(self, handler: Callable[..., Any], **kwargs: Any) -> Any:
        return handler(**kwargs)


@dataclass
class TimeoutSandbox:
    """Run the handler on a daemon thread and bound how long the caller
    waits for it. Raises `SandboxTimeout` if the handler doesn't finish
    within `seconds` -- stdlib-only, so this needs no extra dependency. The
    thread is daemonic specifically so an abandoned, still-running handler
    can never hold up interpreter (or test-process) shutdown."""

    seconds: float = 30.0

    def run(self, handler: Callable[..., Any], **kwargs: Any) -> Any:
        outcome: dict[str, Any] = {}

        def target() -> None:
            try:
                outcome["value"] = handler(**kwargs)
            except BaseException as error:  # re-raised on the caller's thread below
                outcome["error"] = error

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.seconds)
        if thread.is_alive():
            raise SandboxTimeout(f"Handler exceeded {self.seconds}s")
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")
