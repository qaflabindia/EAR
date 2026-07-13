"""Tests for governed self-modification (ear/evolution.py).

All offline: the Evolver's gates are code -- default-deny, prohibition
over allow-list, explanation/approval/sandbox/rollback/evaluation
requirements, rollback on a failed evaluation or a crashed apply -- and
every verdict is an `evolution` trail record. No model is ever needed to
enforce a fence.
"""

from __future__ import annotations

import pytest

from ear import (
    Approval,
    ApprovalRequired,
    EvolutionChange,
    EvolutionDenied,
    EvolutionPolicy,
    Runtime,
    Strategy,
)

FULL_POLICY = EvolutionPolicy(
    allowed_changes=[
        "skill_prompt",
        "skill_creation",
        "strategy",
        "workflow_branch",
        "validation_rule",
        "tool_adapter",
    ],
    prohibited_changes=[
        "hard_policy",
        "approval_authority",
        "audit_logging",
        "data_access_boundary",
    ],
    require_sandbox=True,
    require_evaluation=True,
    require_explanation=True,
    require_human_approval_for=[
        "generated_code",
        "workflow_structure",
        "production_promotion",
    ],
    rollback_required=True,
)


def permissive_policy(**overrides) -> EvolutionPolicy:
    """A policy with every requirement relaxed, so individual gates can be
    turned back on one at a time."""
    settings = dict(
        allowed_changes=["skill_prompt", "tool_adapter"],
        prohibited_changes=["hard_policy"],
        require_sandbox=False,
        require_evaluation=False,
        require_explanation=False,
        require_human_approval_for=[],
        rollback_required=False,
    )
    settings.update(overrides)
    return EvolutionPolicy(**settings)


def change(kind: str = "skill_prompt", explanation: str = "tightens the grading prose") -> EvolutionChange:
    return EvolutionChange(kind=kind, name="risk_grade", description="rewrite the prompt", explanation=explanation)


class Passing:
    passed = True


class Failing:
    passed = False


# ---------------------------------------------------------------------------
# The policy's verdicts.
# ---------------------------------------------------------------------------


def test_policy_is_default_deny_for_unlisted_kinds():
    assert FULL_POLICY.permits("skill_prompt")
    assert not FULL_POLICY.permits("persona_rewrite")
    assert "default-deny" in FULL_POLICY.refusal("persona_rewrite")


def test_prohibition_wins_even_over_the_allow_list():
    policy = EvolutionPolicy(allowed_changes=["audit_logging"], prohibited_changes=["audit_logging"])
    assert not policy.permits("audit_logging")
    assert "prohibited" in policy.refusal("audit_logging")


def test_kinds_match_case_and_punctuation_insensitively():
    assert FULL_POLICY.permits("Skill Prompt")
    assert FULL_POLICY.permits("skill-prompt")
    assert not FULL_POLICY.permits("Hard Policy")
    assert FULL_POLICY.needs_approval("Generated Code")


def test_describe_names_the_fences():
    line = FULL_POLICY.describe()
    assert "skill_prompt" in line and "hard_policy" in line
    assert "sandbox" in line and "rollback" in line
    assert "generated_code" in line


# ---------------------------------------------------------------------------
# The Evolver's gates, in order.
# ---------------------------------------------------------------------------


def test_runtime_without_enable_evolution_refuses_every_change():
    runtime = Runtime(name="closed")
    with pytest.raises(EvolutionDenied, match="not enabled"):
        runtime.evolve(change(), apply=lambda: None)


def test_enable_evolution_lands_on_the_trail():
    runtime = Runtime(name="open").enable_evolution(FULL_POLICY)
    record = runtime.reasoning_log.for_stage("evolution")[0]
    assert "evolution enabled" in record.output


def test_prohibited_kind_is_denied_on_the_record():
    runtime = Runtime(name="fenced").enable_evolution(FULL_POLICY)
    with pytest.raises(EvolutionDenied, match="prohibited"):
        runtime.evolve(change(kind="audit_logging"), apply=lambda: None)
    denial = runtime.reasoning_log.for_stage("evolution")[-1]
    assert denial.output.startswith("DENIED")
    assert denial.inputs["kind"] == "audit_logging"


def test_missing_explanation_is_denied_when_required():
    runtime = Runtime(name="explain").enable_evolution(permissive_policy(require_explanation=True))
    with pytest.raises(EvolutionDenied, match="explanation"):
        runtime.evolve(change(explanation=""), apply=lambda: None)


def test_approval_gated_kind_parks_without_a_verdict():
    policy = permissive_policy(allowed_changes=["generated_code"], require_human_approval_for=["generated_code"])
    runtime = Runtime(name="gated").enable_evolution(policy)
    with pytest.raises(ApprovalRequired):
        runtime.evolve(change(kind="generated_code"), apply=lambda: None)
    parked = runtime.reasoning_log.for_stage("evolution")[-1]
    assert "PENDING" in parked.output


def test_rejected_verdict_denies_and_approved_verdict_releases():
    policy = permissive_policy(allowed_changes=["generated_code"], require_human_approval_for=["generated_code"])
    runtime = Runtime(name="gated").enable_evolution(policy)

    rejection = Approval(verdict=False, approver="lakshminarasimhan.santhanam@gigkri.com")
    with pytest.raises(EvolutionDenied, match="rejected"):
        runtime.evolve(change(kind="generated_code"), apply=lambda: None, approval=rejection)

    applied = []
    release = Approval(verdict=True, approver="lakshminarasimhan.santhanam@gigkri.com")
    note = runtime.evolve(change(kind="generated_code"), apply=lambda: applied.append(True), approval=release)
    assert applied == [True]
    assert "approved by lakshminarasimhan.santhanam@gigkri.com" in note


def test_sandbox_requirement_refuses_an_unconfined_runtime():
    runtime = Runtime(name="bare").enable_evolution(permissive_policy(require_sandbox=True))
    with pytest.raises(EvolutionDenied, match="[Ss]andbox"):
        runtime.evolve(change(), apply=lambda: None)

    runtime.sandbox = object()
    assert "promoted" in runtime.evolve(change(), apply=lambda: None)


def test_rollback_requirement_refuses_a_one_way_change():
    runtime = Runtime(name="doors").enable_evolution(permissive_policy(rollback_required=True))
    with pytest.raises(EvolutionDenied, match="rollback"):
        runtime.evolve(change(), apply=lambda: None)
    assert "promoted" in runtime.evolve(change(), apply=lambda: None, rollback=lambda: None)


def test_evaluation_requirement_refuses_an_unevaluated_change():
    runtime = Runtime(name="graded").enable_evolution(permissive_policy(require_evaluation=True))
    with pytest.raises(EvolutionDenied, match="evaluation"):
        runtime.evolve(change(), apply=lambda: None)


def test_failed_evaluation_rolls_the_change_back():
    runtime = Runtime(name="graded").enable_evolution(permissive_policy(require_evaluation=True))
    state = {"prompt": "before"}
    with pytest.raises(EvolutionDenied, match="rolled back"):
        runtime.evolve(
            change(),
            apply=lambda: state.update(prompt="after"),
            rollback=lambda: state.update(prompt="before"),
            evaluate=lambda: Failing(),
        )
    assert state["prompt"] == "before"


def test_passing_evaluation_promotes_and_records():
    runtime = Runtime(name="graded").enable_evolution(permissive_policy(require_evaluation=True))
    note = runtime.evolve(change(), apply=lambda: None, evaluate=lambda: Passing())
    assert "promoted 'risk_grade (skill_prompt)'" in note
    assert "evaluation passed" in note
    promotion = runtime.reasoning_log.for_stage("evolution")[-1]
    assert promotion.output == note


def test_crashed_apply_rolls_back_and_reraises():
    runtime = Runtime(name="crash").enable_evolution(permissive_policy())
    state = {"prompt": "before"}

    def explode():
        state["prompt"] = "after"
        raise RuntimeError("bad rewrite")

    with pytest.raises(RuntimeError, match="bad rewrite"):
        runtime.evolve(change(), apply=explode, rollback=lambda: state.update(prompt="before"))
    assert state["prompt"] == "before"
    failure = runtime.reasoning_log.for_stage("evolution")[-1]
    assert failure.output.startswith("FAILED") and "rolled back" in failure.output


# ---------------------------------------------------------------------------
# The Acquirer under the same fence: create_tool is a tool_adapter change.
# ---------------------------------------------------------------------------


def test_acquirer_refuses_creation_when_tool_adapter_is_not_allowed():
    runtime = Runtime(name="fenced")
    runtime.strategy = Strategy()
    runtime.enable_evolution(permissive_policy(allowed_changes=["skill_prompt"]))
    refusal = runtime.acquirer.create_tool(runtime, "wire_transfer", "Moves money.")
    assert refusal.startswith("Refused by the evolution policy")
    assert not runtime.strategy.tools


def test_acquirer_still_creates_when_tool_adapter_is_allowed():
    runtime = Runtime(name="open")
    runtime.strategy = Strategy()
    runtime.enable_evolution(permissive_policy())
    result = runtime.acquirer.create_tool(runtime, "rate_lookup", "Looks up a rate.")
    assert "Declared tool 'rate_lookup'" in result


def test_acquirer_unchanged_when_evolution_was_never_enabled():
    runtime = Runtime(name="asis")
    runtime.strategy = Strategy()
    result = runtime.acquirer.create_tool(runtime, "rate_lookup", "Looks up a rate.")
    assert "Declared tool 'rate_lookup'" in result


# ---------------------------------------------------------------------------
# Authoring the policy in memory.md.
# ---------------------------------------------------------------------------

EVOLUTION_MARKDOWN = """## Evolution

The runtime may improve itself within these fences. Trial every change in
the sandbox, evaluate it before promotion, explain it on the record, and
keep a rollback.

- Allowed: skill prompt, skill creation, strategy, workflow branch, validation rule, tool adapter
- Prohibited: hard policy, approval authority, audit logging, data access boundary
- Approval required: generated code, workflow structure, production promotion
"""


def test_strategy_reads_an_evolution_section():
    strategy = Strategy.from_markdown(EVOLUTION_MARKDOWN)
    policy = strategy.evolution_policy
    assert policy is not None
    assert policy.permits("skill_prompt")
    assert policy.permits("tool adapter")
    assert not policy.permits("hard_policy")
    assert not policy.permits("something_else")
    assert policy.needs_approval("production promotion")
    assert policy.require_sandbox and policy.require_evaluation
    assert policy.require_explanation and policy.rollback_required
    assert "sandbox" in strategy.evolution


def test_strategy_reads_relaxed_requirements_from_prose():
    strategy = Strategy.from_markdown(
        "## Evolution\n\n"
        "No sandbox is needed and the evaluation is optional; skip the rollback too.\n\n"
        "- Allowed: skill prompt\n"
    )
    policy = strategy.evolution_policy
    assert not policy.require_sandbox
    assert not policy.require_evaluation
    assert not policy.rollback_required
    assert policy.require_explanation


def test_strategy_leaves_policy_none_when_evolution_is_disabled():
    strategy = Strategy.from_markdown("## Evolution\n\nEvolution is disabled in this runtime.\n")
    assert strategy.evolution_policy is None
    assert Strategy.from_markdown("## Model Selection\n\nNo model.\n").evolution_policy is None
