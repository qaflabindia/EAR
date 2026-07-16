"""Coder -- the runtime writes code for itself, at runtime, safely.

After the model reasons, it can *author a new capability as code* and, once
the code clears every gate, bind it so later cycles have a tool they did not
have before. This is the sharpest edge in the whole framework, so the design
holds one invariant absolutely:

    **Model-authored code never runs in the kernel's own process.**

The kernel only ever *parses* the code (`ast.parse`, which does not execute
it) to reject what is malformed. Every actual execution -- the trial that
proves the code, and every later invocation of the bound tool -- happens as a
**subprocess inside the runtime's `Sandbox`**: a confined filesystem, a
wall-clock timeout, CPU/memory rlimits, and an environment stripped of
ambient secrets. There is no `exec`, no `eval`, no `compile`-and-call of the
model's text anywhere in this module. Self-coding is therefore self-
*extension*, never self-*injection*.

The path a piece of authored code walks:

1. **Author** (`author`) -- the model writes a small stdlib-only Python
   script to a fixed IO contract (read JSON args from `argv[1]`, print one
   JSON result). Authoring is judgment: offline, the Coder refuses rather
   than fabricate code.
2. **Validate** (`validate`) -- a deterministic floor: the script must parse,
   and must not use `exec`/`eval`/dynamic `__import__` tricks. Parsing runs
   no code.
3. **Trial** (`trial`) -- the script is written into the Sandbox and *run
   there* against the author's sample input; it must exit cleanly and produce
   the expected output.
4. **Gate** (`install`) -- the change walks the existing `Evolver`: the
   `EvolutionPolicy` must allow the `code_capability` kind, AAWDFC must judge
   it legitimate, AGCC/human approval applies where required, a Sandbox must
   confine it, the trial is the evaluation, and a rollback removes the file
   and unbinds the tool. Only a change that clears all of that is bound.
5. **Bind** -- a `BoundTool` whose handler runs the sandboxed script as a
   subprocess. Later cycles call it like any other tool, on the trail.

Standard library only.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .evolution import EvolutionChange
from .reasoning_log import model_name

CODE_KIND = "code_capability"

# Constructs the validation floor refuses outright: dynamic execution of text.
# The Sandbox is the real containment boundary; this is defence in depth, and
# a fast, legible refusal of the obviously-wrong.
_FORBIDDEN_CALLS = {"exec", "eval", "compile"}


@dataclass
class CodeCapability:
    """One capability authored as code: its name, what it does, the Python
    script, and the sample the trial proves it against."""

    name: str
    description: str
    source: str
    sample_input: str = "{}"
    expected: str = ""
    explanation: str = ""

    def slug(self) -> str:
        safe = "".join(ch if (ch.isalnum() or ch in "_") else "_" for ch in self.name.strip().lower())
        return safe.strip("_") or "capability"

    def relpath(self) -> str:
        return f".ear/coded/{self.slug()}.py"


@dataclass
class CodeReview:
    """The verdict of the beyond-suspicion review: whether the code is safe to
    trust, any suspicion that gives pause, and the basis of the ruling."""

    reasonable: bool
    suspicion: str = ""
    rationale: str = ""
    basis: str = ""


@dataclass
class Coder:
    """The runtime's own code author. Standalone, like the Optimizer and the
    Acquirer -- not a per-cycle stage. `install` is the one entry point that
    materializes and binds; everything before it is proposal.

    `require_review` (on by default) demands that a model judge the authored
    code safe *beyond any reasonable suspicion* before it is installed. It is
    **fail-closed**: with no reviewer bound, a self-authored capability is
    refused -- code the runtime wrote for itself is not trusted on a
    deterministic floor, because 'beyond suspicion' is a judgment nothing
    deterministic can stand in for."""

    require_review: bool = True

    def author(self, runtime: Any, spec: str, model_binding: Any = None) -> Optional[CodeCapability]:
        """The model writes a capability to spec. Authoring is judgment:
        with no model bound the Coder refuses -- it never fabricates code
        from a template. Returns None on refusal, recorded on the trail."""
        binding = model_binding if model_binding is not None else getattr(runtime, "model_binding", None)
        lm = getattr(binding, "lm", None) if binding is not None else None
        if binding is not None:
            binding.activate()
            lm = getattr(binding, "lm", None)
        log = getattr(runtime, "reasoning_log", None)
        if lm is None:
            if log is not None:
                log.record(
                    stage="coder",
                    inputs={"spec": spec},
                    output="REFUSED to author -- writing code is judgment, and no model is bound",
                    rationale="offline, the Coder never fabricates code from a template",
                )
            return None

        from .signatures import AuthorCode

        result = AuthorCode.run(lm, spec=spec)
        capability = CodeCapability(
            name=str(getattr(result, "name", "") or "capability"),
            description=str(getattr(result, "description", "") or spec[:80]),
            source=str(getattr(result, "code", "") or ""),
            sample_input=str(getattr(result, "sample_input", "") or "{}"),
            expected=str(getattr(result, "expected", "") or ""),
            explanation=f"Authored by the runtime to spec: {spec}",
        )
        if log is not None:
            log.record(
                stage="coder",
                inputs={"spec": spec, "name": capability.name},
                output=f"authored '{capability.name}' ({len(capability.source)} chars)",
                rationale=capability.description,
                model=model_name(binding),
            )
        return capability

    def validate(self, capability: CodeCapability) -> tuple[bool, str]:
        """The deterministic floor: the script must parse and must not run
        code dynamically. Parsing executes nothing -- `ast.parse` builds a
        tree, it does not run the module."""
        source = capability.source.strip()
        if not source:
            return False, "no code to validate"
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            return False, f"the authored code does not parse: {error}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                return False, f"the authored code calls '{node.func.id}(...)' -- dynamic code execution is refused"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__import__":
                return False, "the authored code uses __import__(...) -- dynamic imports are refused"
        return True, "parses and uses no dynamic execution"

    def review(self, runtime: Any, capability: CodeCapability, model_binding: Any = None) -> CodeReview:
        """Judge the authored code safe beyond any reasonable suspicion. The
        model reviews it against its stated purpose (`JudgeCodeReasonable`);
        approval is withheld on any suspicion. Fail-closed: with no reviewer
        bound, the review does not pass -- self-authored code is never
        certified beyond suspicion by a deterministic stand-in."""
        binding = model_binding if model_binding is not None else getattr(runtime, "model_binding", None)
        lm = getattr(binding, "lm", None) if binding is not None else None
        if binding is not None:
            binding.activate()
            lm = getattr(binding, "lm", None)

        if lm is None:
            verdict = CodeReview(
                reasonable=False,
                suspicion="no reviewer bound",
                rationale="a self-authored capability cannot be certified beyond suspicion without an LLM reviewer",
                basis="fail-closed floor (no model)",
            )
        else:
            from .signatures import JudgeCodeReasonable

            result = JudgeCodeReasonable.run(lm, purpose=capability.description or capability.name, code=capability.source)
            verdict = CodeReview(
                reasonable=bool(getattr(result, "reasonable", False)),
                suspicion=str(getattr(result, "suspicion", "") or "none"),
                rationale=str(getattr(result, "rationale", "") or "reviewed by the model"),
                basis="judged by the model",
            )
        log = getattr(runtime, "reasoning_log", None)
        if log is not None:
            log.record(
                stage="code_review",
                inputs={"name": capability.name, "suspicion": verdict.suspicion, "basis": verdict.basis},
                output=("CLEARED beyond suspicion" if verdict.reasonable else "WITHHELD -- not beyond suspicion"),
                rationale=verdict.rationale,
                model=model_name(binding) if verdict.basis == "judged by the model" else "",
            )
        return verdict

    def trial(self, runtime: Any, capability: CodeCapability) -> Any:
        """Write the script into the Sandbox and run it *there* against the
        author's sample input -- never in this process. Returns a small
        result object with `.passed` (clean exit and the expected substring
        present) for the Evolver's evaluation step."""
        sandbox = self._require_sandbox(runtime)
        sandbox.write_text(capability.relpath(), capability.source)
        outcome = sandbox.run(["python3", capability.relpath(), capability.sample_input or "{}"])
        passed = outcome.returncode == 0 and (not capability.expected or capability.expected in (outcome.stdout or ""))
        log = getattr(runtime, "reasoning_log", None)
        if log is not None:
            log.record(
                stage="coder",
                inputs={"name": capability.name, "sample": capability.sample_input},
                output=f"trial {'PASSED' if passed else 'FAILED'} (exit {outcome.returncode})",
                rationale=(outcome.stdout or outcome.stderr or "")[:200],
            )
        return _TrialResult(passed=passed, result=outcome)

    def install(self, runtime: Any, capability: CodeCapability, approval: Any = None, model_binding: Any = None) -> str:
        """Materialize and bind the capability -- but only through every gate.
        The validation floor runs first; then the `Evolver` walks the change
        (kind allowed, legitimacy, approval, sandbox, evaluation=trial,
        rollback). On promotion the tool is bound, running only in the
        Sandbox. Raises `EvolutionDenied`/`ApprovalRequired` on refusal, on
        the record."""
        from .evolution import EvolutionDenied, Evolver

        ok, reason = self.validate(capability)
        if not ok:
            log = getattr(runtime, "reasoning_log", None)
            if log is not None:
                log.record(stage="coder", inputs={"name": capability.name}, output=f"REFUSED -- {reason}")
            raise EvolutionDenied(CODE_KIND, reason)

        # Beyond-suspicion review: a model must judge the code safe before it
        # is installed (fail-closed offline). This is the gate that makes
        # self-authored code trustworthy, not merely well-formed.
        if self.require_review:
            verdict = self.review(runtime, capability, model_binding=model_binding)
            if not verdict.reasonable:
                raise EvolutionDenied(
                    CODE_KIND, f"code review withheld approval -- {verdict.suspicion}: {verdict.rationale}"
                )

        # A Sandbox is non-negotiable here, above whatever the policy says: no
        # self-authored code is materialized or run without one.
        self._require_sandbox(runtime)

        change = EvolutionChange(
            kind=CODE_KIND,
            name=capability.name,
            description=capability.description,
            explanation=capability.explanation or f"Self-authored capability: {capability.description}",
            payload={"relpath": capability.relpath()},
        )

        def apply() -> None:
            self._materialize(runtime, capability)

        def rollback() -> None:
            self._remove(runtime, capability)

        def evaluate() -> Any:
            return self.trial(runtime, capability)

        return Evolver().propose(
            runtime, change, apply=apply, rollback=rollback, approval=approval, evaluate=evaluate
        )

    # -- materialization and the sandboxed handler --------------------------

    def _materialize(self, runtime: Any, capability: CodeCapability) -> None:
        """Write the script into the Sandbox and bind a tool whose handler
        runs it there. The handler is the whole safety story: it shells to
        `python3 <script>` inside the box and returns its stdout -- it does
        not, and cannot, run the authored code in this process."""
        sandbox = self._require_sandbox(runtime)
        sandbox.write_text(capability.relpath(), capability.source)
        relpath = capability.relpath()

        def handler(**kwargs: Any) -> str:
            arguments = json.dumps(kwargs)
            outcome = sandbox.run(["python3", relpath, arguments])
            if outcome.returncode != 0:
                return f"capability '{capability.name}' failed (exit {outcome.returncode}): {outcome.stderr or 'no output'}"
            return outcome.stdout or ""

        handler.__name__ = capability.slug()
        # Bind through the self-extension channel (`acquirer_tools`), which
        # admits a newly-created tool without requiring it be pre-declared in
        # the stack -- the same door the Acquirer's create_tool uses.
        from .tool_binder import BoundTool

        binder = getattr(runtime, "tool_binder", None)
        if binder is not None:
            self._remove(runtime, capability)  # idempotent: replace any prior binding
            binder.acquirer_tools.append(
                BoundTool(name=capability.name, description=capability.description or capability.name, handler=handler)
            )

    def _remove(self, runtime: Any, capability: CodeCapability) -> None:
        sandbox = getattr(runtime, "sandbox", None)
        if sandbox is not None and sandbox.exists(capability.relpath()):
            sandbox.remove(capability.relpath())
        binder = getattr(runtime, "tool_binder", None)
        tools = getattr(binder, "acquirer_tools", None)
        if isinstance(tools, list):
            from .section import normalize

            wanted = normalize(capability.name)
            binder.acquirer_tools = [tool for tool in tools if normalize(getattr(tool, "name", "")) != wanted]

    @staticmethod
    def _require_sandbox(runtime: Any) -> Any:
        sandbox = getattr(runtime, "sandbox", None)
        if sandbox is None:
            raise ValueError(
                "self-authored code requires a Sandbox to run in -- confine the runtime "
                "before letting it code for itself (model-authored code never runs in-process)"
            )
        return sandbox


@dataclass
class _TrialResult:
    passed: bool
    result: Any = None
