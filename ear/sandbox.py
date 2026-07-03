"""Sandbox -- give each runtime instance its own filesystem-confined,
resource-limited workspace, built natively from the standard library.

What a heavyweight harness provides with Docker, EAR provides with
`pathlib`, `subprocess` and (on POSIX) `resource`: a per-instance root
directory the runtime's file tools are confined to, and a `run()` that
executes commands inside it with a wall-clock timeout, optional CPU and
memory rlimits, and an environment stripped of the ambient process's
secrets -- so a spawned command never inherits the deployment's API keys.

The boundary is enforcement, not judgment (the control-plane pattern the
whole runtime follows): code confines paths and caps resources; the model
still decides *what* to read, write or run, and every action lands on the
trail through the same logged tool handler as any tool. A path that tries
to escape the root raises `SandboxViolation` -- a `PermissionError`, so it
joins EAR's governance-stop family and, wrapped by the binder, returns to
the model as text.

Honesty, the way a serious system states it: a pure-stdlib `Sandbox` is a
*containment convention* for EAR's own cooperative file tools plus a
*resource and time boundary* for the commands it spawns. It is **not** a
security jail against hostile code -- `cwd` confinement is not `chroot`,
and a determined subprocess can still read outside the root. For a true
isolation boundary, plug an OS-container provider into the same seam: any
object exposing `resolve` / `read_text` / `write_text` / `run` /
`as_tools` can stand in for this `Sandbox` on `Runtime.sandbox`, and the
rest of the runtime never changes.

Each subagent spawned by the Spawner gets its own `child()` sandbox under
the parent's root, so isolation nests: a runtime instance -- and every
instance it spawns -- runs in its own box.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

_POSIX = os.name == "posix"

# The workspace scaffold every sandbox lays down, mirroring the areas a
# task-scoped agent expects: where it works, what it was given, what it
# produced.
_SCAFFOLD = ("workspace", "outputs", "uploads")

# Environment variables a spawned command may see. Deliberately excludes
# everything else -- credentials, tokens and the ambient config never reach
# a sandboxed process unless the caller passes them explicitly.
_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR")

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_OUTPUT = 64 * 1024


class SandboxViolation(PermissionError):
    """A path or action tried to leave the sandbox. A PermissionError, so a
    handler that treats governance stops as refusals keeps working, and the
    tool binder returns it to the model as text."""


@dataclass
class SandboxResult:
    """The outcome of one sandboxed command: what it printed, how it ended,
    and whether it was cut off by the time or output limits."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    timed_out: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def render(self) -> str:
        """The result as the text a tool hands back to the model."""
        parts: list[str] = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.stderr.strip():
            parts.append("[stderr]\n" + self.stderr.rstrip())
        status = "timed out" if self.timed_out else f"exit {self.returncode}"
        note = f"[{status}, {self.duration_ms} ms"
        if self.truncated:
            note += ", output truncated"
        parts.append(note + "]")
        return "\n".join(parts)


@dataclass
class Sandbox:
    """One runtime instance's isolated workspace: a confined filesystem and
    a governed command runner. Create with `Sandbox.create(...)`."""

    root: Path
    name: str = "sandbox"
    timeout: float = DEFAULT_TIMEOUT
    memory_mb: Optional[int] = None
    max_output_bytes: int = DEFAULT_MAX_OUTPUT
    env_allowlist: tuple = _ENV_ALLOWLIST
    ephemeral: bool = False
    _temp: bool = False

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        for area in _SCAFFOLD:
            (self.root / area).mkdir(exist_ok=True)

    @classmethod
    def create(
        cls,
        root: Optional[Union[str, Path]] = None,
        *,
        ephemeral: bool = False,
        name: str = "sandbox",
        **kwargs: Any,
    ) -> "Sandbox":
        """Open a sandbox. With no `root`, a fresh temporary directory is
        used (and cleaned up by `close()`); with a `root`, that directory
        is the box and is never deleted underneath the author."""
        temp = root is None
        resolved = Path(tempfile.mkdtemp(prefix="ear-sandbox-")) if temp else Path(root)
        sandbox = cls(root=resolved, name=name, ephemeral=ephemeral, **kwargs)
        sandbox._temp = temp
        return sandbox

    def child(self, name: str) -> "Sandbox":
        """A nested sandbox for a spawned subagent, rooted beneath this one
        -- so isolation nests instance-within-instance."""
        safe = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in name.strip().lower()) or "subagent"
        return Sandbox(
            root=self.root / "subagents" / safe,
            name=f"{self.name}/{safe}",
            timeout=self.timeout,
            memory_mb=self.memory_mb,
            max_output_bytes=self.max_output_bytes,
            env_allowlist=self.env_allowlist,
        )

    def close(self) -> None:
        """Remove the workspace if this sandbox created a temporary one.
        A sandbox rooted at an author-named directory is never deleted."""
        if self._temp and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.ephemeral:
            self.close()

    # -- confined filesystem --------------------------------------------------

    def resolve(self, relpath: Union[str, Path] = "") -> Path:
        """A path inside the sandbox, or a `SandboxViolation`. Absolute
        paths and any `..` that escapes the root are refused -- the whole
        point of the box."""
        rel = str(relpath)
        if os.path.isabs(rel):
            raise SandboxViolation(f"absolute path {rel!r} is outside the sandbox '{self.name}'")
        root = self.root.resolve()
        target = (root / rel).resolve()
        if target != root and root not in target.parents:
            raise SandboxViolation(f"path {rel!r} escapes the sandbox '{self.name}'")
        return target

    def read_text(self, relpath: Union[str, Path]) -> str:
        return self.resolve(relpath).read_text(encoding="utf-8")

    def write_text(self, relpath: Union[str, Path], text: str) -> int:
        target = self.resolve(relpath)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = str(text)
        target.write_text(payload, encoding="utf-8")
        return len(payload)

    def exists(self, relpath: Union[str, Path]) -> bool:
        try:
            return self.resolve(relpath).exists()
        except SandboxViolation:
            return False

    def remove(self, relpath: Union[str, Path]) -> bool:
        target = self.resolve(relpath)
        if target == self.root.resolve():
            raise SandboxViolation("refusing to remove the sandbox root itself")
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            return True
        if target.exists():
            target.unlink()
            return True
        return False

    def list(self, subdir: Union[str, Path] = "") -> list[str]:
        base = self.resolve(subdir)
        if not base.is_dir():
            return []
        return sorted(entry.name + ("/" if entry.is_dir() else "") for entry in base.iterdir())

    # -- governed execution ---------------------------------------------------

    def run(
        self,
        command: Union[str, list],
        timeout: Optional[float] = None,
        stdin: str = "",
        env: Optional[dict] = None,
    ) -> SandboxResult:
        """Run a command inside the sandbox: cwd is the root, the clock is
        capped, the environment carries no ambient secrets, and (on POSIX)
        CPU and memory rlimits apply. A timeout kills the whole process
        group. Never raises on a failing command -- the failure is in the
        result, so it can be handed back to the model as text."""
        argv = shlex.split(command) if isinstance(command, str) else [str(part) for part in command]
        if not argv:
            raise SandboxViolation("empty command")
        limit = self.timeout if timeout is None else timeout
        started = time.monotonic()
        kwargs: dict[str, Any] = {
            "cwd": str(self.root),
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "env": self._child_env(env),
        }
        if _POSIX:
            kwargs["start_new_session"] = True  # own process group, so a timeout kills children too
            kwargs["preexec_fn"] = self._rlimits()
        try:
            process = subprocess.Popen(argv, **kwargs)
        except OSError as error:
            return SandboxResult(
                returncode=127,
                stderr=f"could not launch {argv[0]!r} in sandbox '{self.name}': {error}",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        timed_out = False
        try:
            out, err = process.communicate(input=stdin, timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill(process)
            try:
                out, err = process.communicate(timeout=5)
            except Exception:  # noqa: BLE001 -- the process is being force-killed
                out, err = "", ""
        out, cut_out = self._truncate(out or "")
        err, cut_err = self._truncate(err or "")
        return SandboxResult(
            returncode=process.returncode if process.returncode is not None else -1,
            stdout=out,
            stderr=err,
            duration_ms=int((time.monotonic() - started) * 1000),
            timed_out=timed_out,
            truncated=cut_out or cut_err,
        )

    def _child_env(self, extra: Optional[dict]) -> dict:
        env = {key: os.environ[key] for key in self.env_allowlist if key in os.environ}
        env.setdefault("PATH", os.environ.get("PATH", "/usr/bin:/bin"))
        if extra:
            env.update({str(key): str(value) for key, value in extra.items()})
        return env

    def _rlimits(self):
        if not _POSIX:
            return None
        timeout, memory_mb = self.timeout, self.memory_mb

        def apply() -> None:  # pragma: no cover - runs in the forked child
            import resource

            try:
                cpu = int(timeout) + 1
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
            except (ValueError, OSError):
                pass
            if memory_mb:
                cap = int(memory_mb) * 1024 * 1024
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
                except (ValueError, OSError):
                    pass

        return apply

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        try:
            if _POSIX:
                import signal

                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                return
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            process.kill()
        except OSError:
            pass

    def _truncate(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.max_output_bytes:
            return text, False
        return text[: self.max_output_bytes] + "\n…[truncated]", True

    # -- as tools -------------------------------------------------------------

    def as_tools(self) -> list:
        """The sandbox exposed as EAR tools -- confined file read/write/list
        and a governed shell -- so a runtime's native tool loop can act
        inside its box, every call on the trail through the logged handler.
        A path escape surfaces as a tool failure returned to the model."""
        from .tool_binder import BoundTool

        sandbox = self

        def read_file(path: str) -> str:
            return sandbox.read_text(path)

        def write_file(path: str, content: str) -> str:
            written = sandbox.write_text(path, content)
            return f"wrote {written} characters to {path}"

        def list_files(subdir: str = "") -> str:
            entries = sandbox.list(subdir)
            return "\n".join(entries) if entries else "(empty)"

        def run_shell(command: str) -> str:
            return sandbox.run(command).render()

        return [
            BoundTool(name="read_file", description="Read a text file from the sandbox workspace.", handler=read_file),
            BoundTool(
                name="write_file",
                description="Write text to a file in the sandbox workspace (creates directories as needed).",
                handler=write_file,
            ),
            BoundTool(
                name="list_files",
                description="List the files in the sandbox workspace, or a subdirectory of it.",
                handler=list_files,
            ),
            BoundTool(
                name="run_shell",
                description="Run a shell command inside the sandbox -- confined to the workspace and time-limited.",
                handler=run_shell,
            ),
        ]
