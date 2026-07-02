"""Tests for the natural-language markdown authoring layer.

Everything here runs offline: the loader is structural, and the loaded
runtime exercises its deterministic fallbacks, so the whole stacked
authoring model -- skills.md -> persona.md -> workflow.md -> process.md ->
policy.md -> memory.md -> Runtime -- is testable with no LLM configured.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from ear import (
    Exchange,
    Intent,
    Loader,
    Persona,
    Runtime,
    SessionStore,
    Spawner,
    Strategy,
    load_runtime,
)
from ear.section import parse_document

EXAMPLE_STACK = Path(__file__).resolve().parent.parent / "examples" / "credit_risk_stack"

requires_anthropic_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY is not set in the environment -- live-LLM tests are skipped",
)


# ---------------------------------------------------------------------------
# The markdown section parser.
# ---------------------------------------------------------------------------


def test_parse_document_splits_title_preamble_and_sections():
    document = parse_document(
        "# Skills\n\nPrompts stacked into skills.\n\n"
        "## banding\nBand the profile.\n\n## grading\nGrade the bands.\n"
    )
    assert document.title == "Skills"
    assert document.preamble == "Prompts stacked into skills."
    assert [section.name for section in document.sections] == ["banding", "grading"]
    assert document.section_named("Banding") is not None


def test_section_body_extracts_only_known_fields():
    document = parse_document(
        "## Cap\nThe cap is firm: never exceed it.\n\nFallback: amount <= 100\nApplies to: runtime\n"
    )
    body = document.sections[0].body(field_keys=("fallback", "applies to"))
    assert body.field("fallback") == "amount <= 100"
    assert body.field("applies to") == "runtime"
    # The colon inside prose is not swallowed as a field.
    assert "The cap is firm: never exceed it." in body.prose


def test_section_body_collects_bullets_and_numbered_items():
    document = parse_document("## W\n1. First step.\n2. Second step.\n- a bullet\n")
    body = document.sections[0].body()
    assert body.numbered == ["First step.", "Second step."]
    assert body.bullets == ["a bullet"]


def test_parse_document_handles_crlf_and_missing_title():
    document = parse_document("## Only Section\r\nSome prose.\r\n")
    assert document.title == ""
    assert document.sections[0].name == "Only Section"
    assert document.sections[0].body().prose == "Some prose."


# ---------------------------------------------------------------------------
# Stacking a runtime from markdown, layer by layer.
# ---------------------------------------------------------------------------


def write_stack(directory: Path, **files: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, text in files.items():
        (directory / f"{filename}.md").write_text(text, encoding="utf-8")
    return directory


MINIMAL_STACK = dict(
    skills=(
        "# Skills\n\n## risk_grade\nCombine the score band and DTI band into a grade A-E.\n\n"
        "## decide\nDecide approve or decline against the grade.\n"
    ),
    persona=(
        "# Personas\n\n## Credit Risk Guru\nUnderwrite conservatively.\n\n"
        "Skills: risk_grade, decide\n"
    ),
    workflow=(
        "# Workflows\n\n## Underwriting Workflow\n\n"
        "1. Band the profile and assign a risk grade. (Credit Risk Guru)\n"
        "2. Decide approve or decline against the grade. (Credit Risk Guru)\n\n"
        "Policies: Loan Amount Cap\n"
    ),
    process=(
        "# Credit Risk Runtime\n\n## Underwrite Consumer Loan\n"
        "Evaluates a consumer loan application end to end.\n\n"
        "Workflows: Underwriting Workflow\n"
    ),
    policy=(
        "# Policies\n\n## Loan Amount Cap\nThe loan must not exceed $75,000.\n\n"
        "Fallback: loan_amount <= 75000\nApplies to: Underwriting Workflow\n\n"
        "## DTI Ceiling\nDebt-to-income must not exceed 0.43.\n\n"
        "Fallback: debt_to_income <= 0.43\nApplies to: runtime\n"
    ),
)


def test_load_runtime_stacks_every_layer(tmp_path):
    runtime = load_runtime(write_stack(tmp_path / "stack", **MINIMAL_STACK))

    assert runtime.name == "Credit Risk Runtime"
    assert [process.name for process in runtime.processes] == ["Underwrite Consumer Loan"]

    workflow = runtime.processes[0].workflows[0]
    assert workflow.name == "Underwriting Workflow"
    assert [step.instruction for step in workflow.steps] == [
        "Band the profile and assign a risk grade.",
        "Decide approve or decline against the grade.",
    ]
    assert all(step.persona.name == "Credit Risk Guru" for step in workflow.steps)

    persona = workflow.steps[0].persona
    assert [skill.name for skill in persona.skills] == ["risk_grade", "decide"]
    assert persona.skills[0].prompt == "Combine the score band and DTI band into a grade A-E."

    # Policy mapping: the cap governs the workflow, the ceiling the runtime.
    assert [policy.name for policy in workflow.policies] == ["Loan Amount Cap"]
    assert [policy.name for policy in runtime.policies] == ["DTI Ceiling"]


def test_loaded_runtime_reasons_offline_with_stacked_capabilities(tmp_path):
    runtime = load_runtime(write_stack(tmp_path / "stack", **MINIMAL_STACK))
    decision = runtime.reason(Intent(text="Underwrite a consumer loan", context={"loan_amount": 5000}))
    assert "Credit Risk Runtime" in decision
    assert "Credit Risk Guru" in decision


def test_loaded_runtime_enforces_runtime_policy_from_markdown(tmp_path):
    runtime = load_runtime(write_stack(tmp_path / "stack", **MINIMAL_STACK))
    with pytest.raises(PermissionError, match="DTI Ceiling"):
        runtime.reason(Intent(text="Underwrite a consumer loan", context={"debt_to_income": 0.60}))


def test_loaded_runtime_enforces_workflow_policy_from_markdown(tmp_path):
    runtime = load_runtime(write_stack(tmp_path / "stack", **MINIMAL_STACK))
    with pytest.raises(PermissionError, match="Loan Amount Cap"):
        runtime.reason(Intent(text="Underwrite a consumer loan", context={"loan_amount": 90000}))


def test_unknown_skill_reference_fails_loudly(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["persona"] = "# Personas\n\n## Guru\nBe careful.\n\nSkills: not_a_skill\n"
    with pytest.raises(ValueError, match="Unknown skill 'not_a_skill'.*risk_grade"):
        load_runtime(write_stack(tmp_path / "stack", **stack))


def test_unknown_persona_delegation_stays_in_the_instruction(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["workflow"] = (
        "# Workflows\n\n## Underwriting Workflow\n\n1. Check the profile. (Nobody Known)\n"
    )
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    step = runtime.processes[0].workflows[0].steps[0]
    assert step.instruction == "Check the profile. (Nobody Known)"
    assert step.persona is None


def test_persona_can_stack_skills_as_bullets_and_define_inline(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["persona"] = (
        "# Personas\n\n## Credit Risk Guru\nUnderwrite conservatively.\n\n"
        "- risk_grade\n- decide\n- escalate: Escalate anything ambiguous to a human reviewer.\n"
    )
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    persona = runtime.processes[0].workflows[0].steps[0].persona
    assert [skill.name for skill in persona.skills] == ["risk_grade", "decide", "escalate"]
    assert persona.skills[2].prompt == "Escalate anything ambiguous to a human reviewer."


def test_workflow_default_persona_field_delegates_every_step(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["workflow"] = (
        "# Workflows\n\n## Underwriting Workflow\n\nPersona: Credit Risk Guru\n\n"
        "1. Band the profile.\n2. Decide the application.\n"
    )
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    workflow = runtime.processes[0].workflows[0]
    assert all(step.persona.name == "Credit Risk Guru" for step in workflow.steps)


def test_workflow_unreferenced_by_any_process_is_wrapped_not_dropped(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["process"] = "# Credit Risk Runtime\n"
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    assert [process.name for process in runtime.processes] == ["Underwriting Workflow"]
    assert runtime.processes[0].workflows[0].name == "Underwriting Workflow"


def test_policy_without_applies_to_governs_the_runtime(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["policy"] = "# Policies\n\n## Blanket Control\nEvery decision must be explainable.\n"
    stack["workflow"] = "# Workflows\n\n## Underwriting Workflow\n\n1. Decide. (Credit Risk Guru)\n"
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    assert [policy.name for policy in runtime.policies] == ["Blanket Control"]


def test_missing_files_load_an_empty_runtime(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    runtime = load_runtime(directory)
    assert runtime.name == "empty"
    assert runtime.processes == []
    decision = runtime.reason(Intent(text="anything at all"))
    assert "empty" in decision


# ---------------------------------------------------------------------------
# The memory.md strategy: every setting read from natural language.
# ---------------------------------------------------------------------------

STRATEGY_MARKDOWN = """# Memory & Strategy

## Context History
Keep the 30 most recent cycles verbatim; compress anything older.

## Cross-Session Data
Persist memory, experience and adaptations to `.ear/session.json` between sessions.

## Subagent Spawning
Allow spawning up to 4 subagents, each scoped to a single persona.

## Model Selection
Reason with anthropic/claude-opus-4-8, reading the credential from
ANTHROPIC_API_KEY, at a temperature of 0.2.

## MCP
- credit_bureau: pulls credit reports and score history, via `bureau-mcp-server`
- core_banking: reads balances and repayment history, via `corebank-mcp-server`

## Tools
- amortization_calculator: computes the monthly payment for an amount, rate and term

## Skills Discovery
Rank processes by reading their descriptions against the intent, most relevant first.

## Ontological Settings
- risk grade: a letter from A to E, where A is the strongest credit
- decision: exactly one of approve or decline, never a hedge
"""


def test_strategy_reads_every_section_from_natural_language():
    strategy = Strategy.from_markdown(STRATEGY_MARKDOWN)

    assert strategy.history_capacity == 30
    assert strategy.session_enabled and strategy.session_path == ".ear/session.json"
    assert strategy.subagents_enabled and strategy.max_subagents == 4
    assert strategy.provider == "anthropic"
    assert strategy.model == "anthropic/claude-opus-4-8"
    assert strategy.api_key_env_var == "ANTHROPIC_API_KEY"
    assert strategy.temperature == 0.2
    assert [server.name for server in strategy.mcp_servers] == ["credit_bureau", "core_banking"]
    assert strategy.mcp_servers[0].command == "bureau-mcp-server"
    assert "pulls credit reports" in strategy.mcp_servers[0].description
    assert [tool.name for tool in strategy.tools] == ["amortization_calculator"]
    assert "most relevant first" in strategy.skills_discovery
    assert strategy.ontology.meaning_of("risk grade").startswith("a letter from A to E")


def test_strategy_reads_model_from_prose_without_a_slash():
    strategy = Strategy.from_markdown(
        "## Model Selection\nUse Anthropic's claude-haiku-4-5 with the key in MY_SECRET_KEY.\n"
    )
    assert strategy.provider == "anthropic"
    assert strategy.model == "claude-haiku-4-5"
    assert strategy.api_key_env_var == "MY_SECRET_KEY"


def test_strategy_prose_slashes_are_not_mistaken_for_models():
    strategy = Strategy.from_markdown(
        "## Model Selection\nDecide approve/decline offline; no model is configured.\n"
    )
    assert strategy.model == ""
    assert strategy.model_binding() is None


def test_strategy_disables_subagent_spawning_from_prose():
    strategy = Strategy.from_markdown("## Subagent Spawning\nNever spawn subagents in this runtime.\n")
    assert strategy.subagents_configured
    assert strategy.subagents_enabled is False


def test_strategy_narrative_carries_ontology_tools_and_mcp():
    narrative = Strategy.from_markdown(STRATEGY_MARKDOWN).narrative()
    assert "risk grade: a letter from A to E" in narrative
    assert "amortization_calculator" in narrative
    assert "credit_bureau" in narrative
    assert "Discovery guidance" in narrative


def test_loader_applies_strategy_to_memory_spawner_and_store(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["memory"] = STRATEGY_MARKDOWN
    directory = write_stack(tmp_path / "stack", **stack)
    runtime = load_runtime(directory)

    assert runtime.memory.capacity == 30
    assert runtime.spawner.enabled and runtime.spawner.limit == 4
    assert runtime.session_store is not None
    assert runtime.session_store.path == str(directory / ".ear" / "session.json")
    assert runtime.strategy.ontology.meaning_of("decision") != ""


def test_loader_attaches_model_binding_only_when_credential_resolves(tmp_path, monkeypatch):
    stack = dict(MINIMAL_STACK)
    stack["memory"] = "## Model Selection\nReason with anthropic/claude-opus-4-8; key in FAKE_STACK_KEY.\n"
    directory = write_stack(tmp_path / "stack", **stack)

    monkeypatch.delenv("FAKE_STACK_KEY", raising=False)
    assert load_runtime(directory).model_binding is None

    monkeypatch.setenv("FAKE_STACK_KEY", "not-a-real-key")
    binding = load_runtime(directory).model_binding
    assert binding is not None
    assert binding.model_id == "anthropic/claude-opus-4-8"
    assert binding.api_key_env_var == "FAKE_STACK_KEY"


# ---------------------------------------------------------------------------
# Cross-session data: memory survives into a fresh session.
# ---------------------------------------------------------------------------


def test_session_store_round_trips_memory_experience_and_adaptations(tmp_path):
    store = SessionStore(str(tmp_path / "state" / "session.json"))
    first = Runtime(name="Persistent-Runtime", session_store=store)
    first.reason(Intent(text="Underwrite the first loan", context={"loan_amount": 1000}))
    first.adaptations.learn_from(first.experience)
    store.save(first)

    second = Runtime(name="Persistent-Runtime")
    assert store.restore(second) is True
    assert second.memory.working[0].intent_text == "Underwrite the first loan"
    assert second.memory.working[0].evidence.basis.startswith("Resolved via")
    assert second.experience.observations == 1
    assert len(second.adaptations.impressions) == 1


def test_runtime_saves_session_after_every_cycle(tmp_path):
    path = tmp_path / "session.json"
    runtime = Runtime(name="Saving-Runtime", session_store=SessionStore(str(path)))
    runtime.reason(Intent(text="first cycle"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["runtime"] == "Saving-Runtime"
    assert len(payload["memory"]["working"]) == 1


def test_new_session_loaded_from_markdown_picks_up_prior_memory(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["memory"] = "## Cross-Session Data\nPersist everything to `state.json` between sessions.\n"
    directory = write_stack(tmp_path / "stack", **stack)

    first = load_runtime(directory)
    first.reason(Intent(text="Underwrite the first loan", context={"loan_amount": 1000}))

    second = load_runtime(directory)
    assert len(second.memory.working) == 1
    assert second.memory.working[0].intent_text == "Underwrite the first loan"


def test_session_store_restore_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "session.json"
    path.write_text("not json at all", encoding="utf-8")
    runtime = Runtime(name="Robust-Runtime")
    assert SessionStore(str(path)).restore(runtime) is False
    assert runtime.memory.working == []


# ---------------------------------------------------------------------------
# Markdown as the system-native input and output format.
# ---------------------------------------------------------------------------


def test_intent_reads_from_markdown_with_typed_context():
    intent = Intent.from_markdown(
        "# Underwrite a $18,500 personal loan for Priya Raman\n\n"
        "Requested for a kitchen renovation.\n\n"
        "## Context\n\n"
        "- loan_amount: 18500\n- debt_to_income: 0.28\n- existing_defaults: 0\n"
        "- first_time_borrower: yes\n- purpose: kitchen renovation\n"
    )
    assert intent.text.startswith("Underwrite a $18,500 personal loan")
    assert "kitchen renovation" in intent.text
    assert intent.context["loan_amount"] == 18500
    assert intent.context["debt_to_income"] == 0.28
    assert intent.context["existing_defaults"] == 0
    assert intent.context["first_time_borrower"] is True
    assert intent.context["purpose"] == "kitchen renovation"


def test_intent_round_trips_through_markdown():
    original = Intent(text="Underwrite a loan", context={"loan_amount": 5000, "purpose": "solar panels"})
    recovered = Intent.from_markdown(original.to_markdown())
    assert recovered.text == original.text
    assert recovered.context == original.context


def test_session_store_round_trips_through_markdown(tmp_path):
    store = SessionStore(str(tmp_path / "session.md"))
    first = Runtime(name="Markdown-Persistent-Runtime")
    first.reason(Intent(text="Underwrite the first loan", context={"loan_amount": 1000, "purpose": "a bike"}))
    first.memory.compressed.append("3 earlier cycles (e.g. approved twice)")
    first.adaptations.learn_from(first.experience)
    store.save(first)

    second = Runtime(name="Markdown-Persistent-Runtime")
    assert store.restore(second) is True
    entry = second.memory.working[0]
    assert entry.intent_text == "Underwrite the first loan"
    assert entry.context == {"loan_amount": 1000, "purpose": "a bike"}
    assert entry.decision == str(first.memory.working[0].decision)
    assert entry.evidence.basis.startswith("Resolved via")
    assert second.memory.compressed == ["3 earlier cycles (e.g. approved twice)"]
    assert second.experience.observations == 1
    assert second.experience.decision_counts == first.experience.decision_counts
    assert second.adaptations.impressions[0].insight == first.adaptations.impressions[0].insight


def test_reasoning_log_flushes_markdown_trail(tmp_path):
    path = tmp_path / "reasoning.md"
    runtime = Runtime(name="Markdown-Audited-Runtime")
    runtime.reasoning_log.path = str(path)
    runtime.reason(Intent(text="first cycle"))
    runtime.reason(Intent(text="second cycle"))

    trail = path.read_text(encoding="utf-8")
    assert "## Cycle 1 --" in trail and "## Cycle 2 --" in trail
    assert "### deliberation" in trail and "### explanation" in trail
    # Free text is blockquoted, so it can never be mistaken for structure.
    assert "> " in trail


def test_exchange_answers_intent_documents_with_decision_documents(tmp_path):
    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    (directory / "intents").mkdir()
    (directory / "intents" / "priya-raman.md").write_text(
        "# Underwrite a $18,500 personal loan for Priya Raman\n\n"
        "## Context\n\n- loan_amount: 18500\n- debt_to_income: 0.28\n",
        encoding="utf-8",
    )
    runtime = load_runtime(directory)
    written = Exchange(directory).run(runtime)

    assert [path.name for path in written] == ["priya-raman.md"]
    decision = (directory / "decisions" / "priya-raman.md").read_text(encoding="utf-8")
    assert decision.startswith("# Decision -- Underwrite a $18,500 personal loan for Priya Raman")
    assert "Status: decided" in decision
    assert "Credit Risk Guru" in decision
    assert "## Policy judgments" in decision

    # Idempotent: already-answered intents are not replayed.
    assert Exchange(directory).run(runtime) == []


def test_exchange_writes_blocked_decision_documents(tmp_path):
    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    (directory / "intent.md").write_text(
        "# Underwrite an oversized loan\n\n## Context\n\n- loan_amount: 90000\n",
        encoding="utf-8",
    )
    runtime = load_runtime(directory)
    written = Exchange(directory).run(runtime)

    assert [path.name for path in written] == ["decision.md"]
    decision = (directory / "decision.md").read_text(encoding="utf-8")
    assert "Status: BLOCKED" in decision
    assert "Loan Amount Cap" in decision
    assert "VIOLATED" in decision


def test_loader_defaults_session_and_audit_to_markdown(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["memory"] = (
        "## Cross-Session Data\nPersist everything between sessions.\n\n"
        "## Reasoning Audit Trail\nLog every reasoning step for review.\n"
    )
    directory = write_stack(tmp_path / "stack", **stack)
    runtime = load_runtime(directory)
    assert runtime.session_store.path == str(directory / ".ear" / "session.md")
    assert runtime.reasoning_log.path == str(directory / ".ear" / "reasoning.md")

    runtime.reason(Intent(text="Underwrite a consumer loan", context={"loan_amount": 5000}))
    second = load_runtime(directory)
    assert len(second.memory.working) == 1


def test_cycle_numbering_continues_across_sessions_in_one_trail(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["memory"] = "## Reasoning Audit Trail\nLog every reasoning step for review.\n"
    directory = write_stack(tmp_path / "stack", **stack)

    first = load_runtime(directory)
    first.reason(Intent(text="Underwrite a consumer loan", context={"loan_amount": 5000}))

    second = load_runtime(directory)
    second.reason(Intent(text="Underwrite another loan", context={"loan_amount": 6000}))

    trail = (directory / ".ear" / "reasoning.md").read_text(encoding="utf-8")
    assert "## Cycle 1 --" in trail and "## Cycle 2 --" in trail
    assert second.reasoning_log.cycle == 2


def test_declared_path_ignores_prose_mentions_of_stack_files():
    strategy = Strategy.from_markdown(
        "## Cross-Session Data\nAs declared here in memory.md, persist everything to .ear/session.md between sessions.\n"
    )
    assert strategy.session_path == ".ear/session.md"


# ---------------------------------------------------------------------------
# The reasoning audit trail.
# ---------------------------------------------------------------------------


def test_reasoning_log_records_every_stage_of_a_cycle(tmp_path):
    runtime = load_runtime(write_stack(tmp_path / "stack", **MINIMAL_STACK))
    runtime.reason(Intent(text="Underwrite a consumer loan", context={"loan_amount": 5000}))

    stages = [record.stage for record in runtime.reasoning_log.records]
    assert stages[0] == "intent"
    assert "policy" in stages and "discovery" in stages
    assert "deliberation" in stages and "explanation" in stages

    deliberation = runtime.reasoning_log.for_stage("deliberation")[0]
    # The full stacked prompt material is on the record for prompt review.
    assert "Credit Risk Guru" in deliberation.inputs["capabilities"]
    assert "risk_grade" in deliberation.inputs["capabilities"]
    assert deliberation.model == "deterministic-fallback"

    policy = runtime.reasoning_log.for_stage("policy")[0]
    assert policy.output == "complies"
    assert policy.rationale  # the "why" is never dropped


def test_reasoning_log_records_blocked_cycles_too(tmp_path):
    runtime = load_runtime(write_stack(tmp_path / "stack", **MINIMAL_STACK))
    with pytest.raises(PermissionError):
        runtime.reason(Intent(text="Underwrite a consumer loan", context={"debt_to_income": 0.60}))
    violated = [record for record in runtime.reasoning_log.for_stage("policy") if record.output == "VIOLATED"]
    assert [record.inputs["policy"] for record in violated] == ["DTI Ceiling"]
    assert "0.43" in violated[0].rationale


def test_reasoning_log_flushes_jsonl_after_each_cycle(tmp_path):
    path = tmp_path / "trail" / "reasoning.jsonl"
    runtime = Runtime(name="Audited-Runtime")
    runtime.reasoning_log.path = str(path)
    runtime.reason(Intent(text="first cycle"))
    runtime.reason(Intent(text="second cycle"))

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {line["cycle"] for line in lines} == {1, 2}
    assert {line["stage"] for line in lines} >= {"intent", "discovery", "deliberation", "explanation"}
    # Append-only: nothing was double-written.
    assert len(lines) == len(runtime.reasoning_log.records)


def test_reasoning_log_render_groups_by_cycle():
    runtime = Runtime(name="Rendered-Runtime")
    runtime.reason(Intent(text="one cycle"))
    rendered = runtime.reasoning_log.render()
    assert "=== Cycle 1" in rendered
    assert "[deliberation]" in rendered


def test_strategy_declares_the_audit_trail_in_natural_language():
    strategy = Strategy.from_markdown(
        "## Reasoning Audit Trail\nLog every reasoning step to `audit/trail.jsonl` for review.\n"
    )
    assert strategy.audit_enabled
    assert strategy.audit_path == "audit/trail.jsonl"


def test_loader_points_the_reasoning_log_at_the_declared_path(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["memory"] = "## Reasoning Audit Trail\nLog every reasoning step to `.ear/reasoning.jsonl`.\n"
    directory = write_stack(tmp_path / "stack", **stack)
    runtime = load_runtime(directory)
    assert runtime.reasoning_log.path == str(directory / ".ear" / "reasoning.jsonl")

    runtime.reason(Intent(text="Underwrite a consumer loan", context={"loan_amount": 5000}))
    assert (directory / ".ear" / "reasoning.jsonl").exists()


def test_policy_judge_returns_the_rationale_offline():
    from ear import Policy

    policy = Policy(name="Cap", fallback_expression="amount <= 10")
    complies, rationale = policy.judge(amount=5)
    assert complies is True and "amount <= 10" in rationale
    complies, rationale = policy.judge(amount=50)
    assert complies is False


# ---------------------------------------------------------------------------
# Dynamic-at-runtime stages: selection, scheduling and delegation are
# judgments the LLM makes per cycle, with deterministic fallbacks offline
# and an audit record either way.
# ---------------------------------------------------------------------------


def test_selector_falls_back_to_dedupe_and_logs_the_choice_offline():
    from ear import Process

    runtime = Runtime(name="Choosing-Runtime")
    first, second = Process(name="Underwrite Loan"), Process(name="Close Account")
    selected = runtime.selector.select(runtime, [first, second, first], intent=Intent(text="underwrite"))

    assert selected == [first, second]
    record = runtime.reasoning_log.for_stage("selection")[0]
    assert record.output == "Underwrite Loan, Close Account"
    assert record.model == "deterministic-fallback"


def test_selector_stays_silent_when_there_is_no_choice():
    from ear import Process

    runtime = Runtime(name="Quiet-Runtime")
    runtime.selector.select(runtime, [Process(name="Only Process")], intent=Intent(text="x"))
    assert runtime.reasoning_log.for_stage("selection") == []


def test_scheduler_falls_back_to_composition_order_and_logs_offline():
    from ear import Workflow

    runtime = Runtime(name="Ordering-Runtime")
    plan = [Workflow(name="Banding"), Workflow(name="Deciding")]
    scheduled = runtime.scheduler.schedule(plan, runtime=runtime, intent=Intent(text="underwrite"))

    assert scheduled == plan and scheduled is not plan
    record = runtime.reasoning_log.for_stage("scheduling")[0]
    assert record.output == "Banding, Deciding"
    assert record.model == "deterministic-fallback"


def test_delegator_leaves_steps_as_authored_offline():
    from ear import Workflow

    runtime = Runtime(name="Delegating-Runtime")
    persona = Persona(name="Credit Risk Guru")
    workflow = Workflow(name="Underwriting Workflow")
    workflow.add_step("Band the profile.", persona=persona)
    workflow.add_step("Write the customer note.")  # left undelegated by the author

    runtime.delegator.delegate(runtime, Intent(text="underwrite"), [workflow])
    assert workflow.steps[1].persona is None
    assert runtime.reasoning_log.for_stage("delegation") == []


def test_delegator_apply_only_binds_resolvable_assignments():
    from ear.delegator import Delegator
    from ear import Workflow

    persona = Persona(name="Customer Advocate")
    workflow = Workflow(name="W")
    workflow.add_step("Write the customer note.")
    undelegated = [(1, workflow.steps[0])]

    applied = Delegator._apply(undelegated, [persona], ["1: Customer Advocate", "9: Nobody", "garbage"])
    assert applied == [(1, persona)]
    assert workflow.steps[0].persona is persona


def test_runtime_reason_records_selection_between_processes(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["process"] = (
        "# Credit Risk Runtime\n\n## Underwrite Consumer Loan\nEvaluates a loan application.\n\n"
        "Workflows: Underwriting Workflow\n\n## Close Dormant Account\nCloses a dormant account.\n"
    )
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    runtime.reason(Intent(text="anything ambiguous", context={"loan_amount": 5000}))
    assert runtime.reasoning_log.for_stage("selection")


def test_unknown_reference_error_suggests_the_close_match(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["persona"] = "# Personas\n\n## Guru\nBe careful.\n\nSkills: risk_grad\n"
    with pytest.raises(ValueError, match="did you mean 'risk_grade'"):
        load_runtime(write_stack(tmp_path / "stack", **stack))


def test_recaller_and_auditor_stay_deterministic_and_silent_offline():
    runtime = Runtime(name="Offline-Runtime")
    runtime.reason(Intent(text="first cycle"))
    runtime.reason(Intent(text="second cycle"))
    # Offline recall is the full window and audit is the flag -- neither is
    # a judgment, so neither lands in the trail.
    assert runtime.reasoning_log.for_stage("recall") == []
    assert runtime.reasoning_log.for_stage("audit") == []
    entry = runtime.memory.working[-1]
    assert entry.evidence.sources["audited"] is True
    assert "first cycle" in entry.evidence.sources["recalled_memory"]


def test_adaptation_is_logged_when_distilled():
    runtime = Runtime(name="Adapting-Runtime")
    runtime.adapter.adapt_every = 1
    runtime.reason(Intent(text="underwrite something"))
    record = runtime.reasoning_log.for_stage("adaptation")[0]
    assert record.output == runtime.adaptations.impressions[0].insight
    assert record.model == "deterministic-fallback"


@requires_anthropic_key
def test_recall_and_audit_are_llm_judged_and_on_the_trail():
    from ear import ModelBinding

    binding = ModelBinding(provider="anthropic", model=os.environ.get("ANTHROPIC_TEST_MODEL", "claude-haiku-4-5"))
    runtime = Runtime(name="Live-Audited-Runtime", model_binding=binding)
    runtime.memory.record("approved a $5,000 bicycle loan", decision="approved at grade B")
    runtime.memory.record("cancelled a lunch reservation", decision="cancelled")

    runtime.reason(Intent(text="Underwrite another small bicycle loan", context={"loan_amount": 4000}))

    recall = runtime.reasoning_log.for_stage("recall")[0]
    assert recall.model == binding.model_id
    audit = runtime.reasoning_log.for_stage("audit")[0]
    assert audit.output.strip()
    assert runtime.memory.working[-1].evidence.sources["audit_assessment"] == audit.output


@requires_anthropic_key
def test_selector_chooses_the_relevant_process_with_llm():
    from ear import ModelBinding, Process

    binding = ModelBinding(provider="anthropic", model=os.environ.get("ANTHROPIC_TEST_MODEL", "claude-haiku-4-5"))
    binding.activate()
    runtime = Runtime(name="Choosing-Runtime", model_binding=binding)
    loan = Process(name="Underwrite Consumer Loan", description="Reviews a loan applicant's credit to decide approval.")
    lunch = Process(name="Cancel Lunch Reservation", description="Cancels a booked cafeteria lunch reservation.")

    selected = runtime.selector.select(runtime, [loan, lunch], intent=Intent(text="Review this loan applicant's credit"))
    assert loan in selected
    assert lunch not in selected
    record = runtime.reasoning_log.for_stage("selection")[0]
    assert record.model == binding.model_id


@requires_anthropic_key
def test_delegator_assigns_the_best_suited_persona_with_llm():
    from ear import ModelBinding, Skill, Workflow

    binding = ModelBinding(provider="anthropic", model=os.environ.get("ANTHROPIC_TEST_MODEL", "claude-haiku-4-5"))
    binding.activate()
    runtime = Runtime(name="Delegating-Runtime", model_binding=binding)

    guru = Persona(name="Credit Risk Guru", instructions="Underwrite conservatively.")
    guru.add_skill(Skill(name="assign_risk_grade", prompt="Grade the credit profile A-E."))
    advocate = Persona(name="Customer Advocate", instructions="Write plainly and kindly to applicants.")
    advocate.add_skill(Skill(name="write_customer_note", prompt="Draft a courteous decision note."))

    workflow = Workflow(name="Underwriting Workflow")
    workflow.add_step("Assign a risk grade.", persona=guru)
    workflow.add_persona(advocate)
    workflow.add_step("Write the customer note announcing the decision.")  # undelegated

    runtime.delegator.delegate(runtime, Intent(text="Underwrite a loan"), [workflow])
    assert workflow.steps[1].persona is advocate
    record = runtime.reasoning_log.for_stage("delegation")[0]
    assert "Customer Advocate" in record.output


# ---------------------------------------------------------------------------
# Subagent spawning.
# ---------------------------------------------------------------------------


def test_runtime_spawns_a_persona_scoped_subagent():
    runtime = Runtime(name="Parent-Runtime")
    persona = Persona(name="Credit Risk Guru", instructions="Underwrite conservatively.")
    decision = runtime.spawn(persona, "Review this application's risk")
    assert "Credit Risk Guru" in decision
    assert len(runtime.spawner.spawned) == 1
    # The subagent's cycle is recorded in its own memory, not the parent's.
    assert len(runtime.memory) == 0
    assert len(runtime.spawner.spawned[0].memory) == 1


def test_spawner_enforces_the_strategy_limit():
    runtime = Runtime(name="Parent-Runtime", spawner=Spawner(limit=1))
    persona = Persona(name="Guru")
    runtime.spawn(persona, "first")
    with pytest.raises(PermissionError, match="Subagent limit of 1"):
        runtime.spawn(persona, "second")


def test_spawner_disabled_by_strategy_refuses_to_spawn():
    runtime = Runtime(name="Parent-Runtime", spawner=Spawner(enabled=False))
    with pytest.raises(PermissionError, match="disabled"):
        runtime.spawn(Persona(name="Guru"), "anything")


# ---------------------------------------------------------------------------
# The shipped example stack loads and reasons end to end, offline.
# ---------------------------------------------------------------------------


def test_example_credit_risk_stack_loads_and_reasons(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    directory = tmp_path / "credit_risk_stack"
    shutil.copytree(EXAMPLE_STACK, directory)

    runtime = load_runtime(directory)
    assert runtime.name == "Credit Risk Enterprise Runtime"
    assert runtime.memory.capacity == 30
    assert runtime.spawner.limit == 4
    assert [process.name for process in runtime.processes] == ["Underwrite Consumer Loan"]
    workflow = runtime.processes[0].workflows[0]
    assert len(workflow.steps) == 4
    assert workflow.steps[3].persona.name == "Customer Advocate"
    assert [policy.name for policy in runtime.policies] == ["Debt-to-Income Ceiling", "Fair Lending Control"]
    assert [policy.name for policy in workflow.policies] == ["Loan Amount Cap"]

    decision = runtime.reason(
        Intent(
            text="Underwrite a $20,000 consumer loan",
            context={"loan_amount": 20000, "debt_to_income": 0.30, "credit_score": 720},
        )
    )
    assert "Credit Risk Guru" in decision

    with pytest.raises(PermissionError, match="Loan Amount Cap"):
        runtime.reason(Intent(text="Underwrite a large loan", context={"loan_amount": 90000}))
