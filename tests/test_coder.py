"""Tests for governed runtime self-coding (`ear/coder.py`).

The invariant under test: model-authored code never runs in the kernel's own
process. Authoring is judgment (offline it refuses); a deterministic floor
rejects malformed or dynamic-execution code without running it; the code is
trialled and later invoked only as a subprocess inside the Sandbox; and the
whole install walks the Evolver gate, rolling back a change whose trial fails.

Offline and deterministic -- the capabilities here are hand-written to stand
in for what the model would author, so the gate and the sandboxed execution
are exercised without a model. One live test covers the authoring judgment.
"""

from __future__ import annotations

import os

import pytest

from ear import Runtime
from ear.coder import CodeCapability, Coder
from ear.evolution import EvolutionDenied, EvolutionPolicy
from ear.evolution_loop import LegitimacyGate
from ear.sandbox import Sandbox

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_TEST_MODEL = os.environ.get("ANTHROPIC_TEST_MODEL", "claude-haiku-4-5")
requires_anthropic_key = pytest.mark.skipif(
    not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY is not set -- live-LLM tests are skipped"
)

_SQUARE = (
    "import sys, json\n"
    "args = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}\n"
    "n = args.get('n', 0)\n"
    "print(json.dumps({'square': n * n}))\n"
)


def _square_capability() -> CodeCapability:
    return CodeCapability(
        name="square",
        description="square a number",
        source=_SQUARE,
        sample_input='{"n": 5}',
        expected='"square": 25',
        explanation="self-authored to square numbers",
    )


@pytest.fixture()
def evolvable_runtime():
    runtime = Runtime(name="C")
    runtime.sandbox = Sandbox.create()
    runtime.enable_evolution(
        EvolutionPolicy(
            allowed_changes=["code_capability"],
            require_sandbox=True,
            require_evaluation=True,
            rollback_required=True,
            require_explanation=True,
        )
    )
    runtime.legitimacy_gate = LegitimacyGate()
    yield runtime
    runtime.sandbox.close()


# ---------------------------------------------------------------------------
# The validation floor -- parsing runs no code.
# ---------------------------------------------------------------------------


def test_validate_accepts_a_well_formed_script():
    ok, _reason = Coder().validate(_square_capability())
    assert ok


def test_validate_rejects_a_syntax_error():
    ok, reason = Coder().validate(CodeCapability(name="x", description="d", source="def (: nope"))
    assert not ok
    assert "does not parse" in reason


def test_validate_rejects_dynamic_execution():
    for source in ("exec('import os')", "eval('1+1')", "__import__('os').system('ls')"):
        ok, reason = Coder().validate(CodeCapability(name="x", description="d", source=source))
        assert not ok, source


def test_validate_rejects_empty():
    assert not Coder().validate(CodeCapability(name="x", description="d", source="   "))[0]


# ---------------------------------------------------------------------------
# Install -- gated, out-of-process, rolled back on failure.
# ---------------------------------------------------------------------------


def test_install_binds_a_capability_that_runs_out_of_process(evolvable_runtime):
    runtime = evolvable_runtime
    note = Coder().install(runtime, _square_capability())
    assert "promoted" in note

    tools = [t for t in runtime.tool_binder.acquirer_tools if t.name == "square"]
    assert len(tools) == 1
    # The bound handler shells to the sandboxed script -- the authored code
    # runs as a subprocess, never in this process.
    output = tools[0].handler(n=7)
    assert '"square": 49' in output


def test_install_requires_a_sandbox():
    runtime = Runtime(name="C")  # no sandbox
    runtime.enable_evolution(EvolutionPolicy(allowed_changes=["code_capability"]))
    with pytest.raises(ValueError):
        Coder().install(runtime, _square_capability())


def test_install_refuses_an_unallowed_kind(evolvable_runtime):
    runtime = evolvable_runtime
    runtime.enable_evolution(EvolutionPolicy(allowed_changes=[], require_sandbox=True))  # code_capability not allowed
    with pytest.raises(EvolutionDenied):
        Coder().install(runtime, _square_capability())


def test_a_failed_trial_rolls_back_and_does_not_bind(evolvable_runtime):
    runtime = evolvable_runtime
    broken = CodeCapability(
        name="broken",
        description="d",
        source="import sys, json\nprint(json.dumps({'oops': 1}))\n",
        sample_input="{}",
        expected='"square"',  # never present -> trial fails
        explanation="will fail its trial",
    )
    with pytest.raises(EvolutionDenied):
        Coder().install(runtime, broken)
    assert not any(t.name == "broken" for t in runtime.tool_binder.acquirer_tools)
    assert not runtime.sandbox.exists(broken.relpath())  # rolled back off disk too


def test_install_refuses_invalid_code_before_the_gate(evolvable_runtime):
    with pytest.raises(EvolutionDenied):
        Coder().install(evolvable_runtime, CodeCapability(name="x", description="d", source="def (: bad"))


def test_trial_runs_in_the_sandbox_and_reports_pass(evolvable_runtime):
    result = Coder().trial(evolvable_runtime, _square_capability())
    assert result.passed
    assert any("trial PASSED" in r.output for r in evolvable_runtime.reasoning_log.for_stage("coder"))


# ---------------------------------------------------------------------------
# Authoring is judgment.
# ---------------------------------------------------------------------------


def test_offline_authoring_refuses(evolvable_runtime):
    # No model bound: the Coder refuses rather than fabricate code.
    assert Coder().author(evolvable_runtime, "compute a moving average") is None
    assert any("REFUSED to author" in r.output for r in evolvable_runtime.reasoning_log.for_stage("coder"))


@requires_anthropic_key
def test_model_authors_a_working_capability(evolvable_runtime):
    from ear import ModelBinding

    runtime = evolvable_runtime
    runtime.model_binding = ModelBinding(provider="anthropic", model=ANTHROPIC_TEST_MODEL)
    capability = Coder().author(
        runtime,
        "Given a JSON object {\"numbers\": [..]}, return {\"total\": sum} of the numbers.",
    )
    assert capability is not None
    assert Coder().validate(capability)[0]
    # It installs and runs out-of-process.
    Coder().install(runtime, capability)
    tool = [t for t in runtime.tool_binder.acquirer_tools if t.name == capability.name][0]
    assert tool.handler(numbers=[1, 2, 3])
