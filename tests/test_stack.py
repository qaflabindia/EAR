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
import time
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
# Contracts: a workflow's Deliverable, declared in natural language,
# extracted and judged by the model at runtime, delivered as ## Data.
# ---------------------------------------------------------------------------

CONTRACT_WORKFLOW = (
    "# Workflows\n\n## Underwriting Workflow\n\n"
    "1. Band the profile and assign a risk grade. (Credit Risk Guru)\n"
    "2. Decide approve or decline against the grade. (Credit Risk Guru)\n\n"
    "Policies: Loan Amount Cap\n\n"
    "### Deliverable\n\nThe decision as structured facts.\n\n"
    "- decision: exactly one of approve or decline\n"
    "- risk grade: the letter grade from A to E\n"
)


def test_deliverable_section_becomes_the_workflow_contract(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["workflow"] = CONTRACT_WORKFLOW
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))

    workflow = runtime.processes[0].workflows[0]
    contract = workflow.contract
    assert contract is not None
    assert contract.name == "Underwriting Workflow Deliverable"
    assert [f.name for f in contract.fields] == ["decision", "risk grade"]
    assert contract.fields[1].identifier == "risk_grade"
    assert "structured facts" in contract.description
    # The Deliverable section is the workflow's contract, never a workflow
    # or a process of its own.
    assert [process.name for process in runtime.processes] == ["Underwrite Consumer Loan"]
    assert len(workflow.steps) == 2


def test_deliverable_with_no_workflow_above_fails_loudly(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["workflow"] = "# Workflows\n\n### Deliverable\n\n- decision: approve or decline\n"
    with pytest.raises(ValueError, match="no workflow above"):
        load_runtime(write_stack(tmp_path / "stack", **stack))


def test_deliverable_field_without_a_meaning_fails_loudly(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["workflow"] = CONTRACT_WORKFLOW.replace("- decision: exactly one of approve or decline", "- decision")
    with pytest.raises(ValueError, match="must be written as 'name: meaning'"):
        load_runtime(write_stack(tmp_path / "stack", **stack))


def test_contract_judge_falls_back_to_structural_presence():
    from ear import Contract

    contract = Contract(name="Deliverable").add_field("decision", "approve or decline")
    conforms, rationale = contract.judge({"decision": "approve"})
    assert conforms is True and "structural" in rationale
    conforms, rationale = contract.judge({"decision": "  "})
    assert conforms is False and "decision" in rationale


def test_offline_contract_extraction_is_skipped_on_the_record(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["workflow"] = CONTRACT_WORKFLOW
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    runtime.reason(Intent(text="Underwrite a consumer loan", context={"loan_amount": 5000}))

    record = runtime.reasoning_log.for_stage("contract")[0]
    assert "skipped" in record.output and "no model" in record.output
    # No fabricated data reaches the evidence.
    assert "data" not in runtime.memory.working[-1].evidence.sources


def test_data_section_round_trips_typed_values():
    from ear.exchange import data_from_decision_document

    document = (
        "# Decision -- x\n\n## Data\n\n"
        "- decision: approve\n- risk grade: B\n- monthly_payment: 372.5\n- defaults: 0\n"
    )
    data = data_from_decision_document(document)
    assert data == {"decision": "approve", "risk grade": "B", "monthly_payment": 372.5, "defaults": 0}


# ---------------------------------------------------------------------------
# The Examiner: markdown-native evaluation with honest offline grading.
# ---------------------------------------------------------------------------


def test_intent_from_markdown_skips_named_sections():
    intent = Intent.from_markdown(
        "# Underwrite a loan\n\n## Expected\n\nIt should be approved.\n\n## Context\n\n- loan_amount: 1000\n",
        skip_sections=("expected",),
    )
    assert "approved" not in intent.text
    assert intent.context == {"loan_amount": 1000}


def test_examiner_grades_structurally_offline_and_reports(tmp_path):
    from ear import Examiner

    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    evaluations = directory / "evaluations"
    evaluations.mkdir()
    (evaluations / "resolves.md").write_text(
        "# Underwrite a small consumer loan\n\n## Context\n\n- loan_amount: 5000\n\n"
        "## Expected\n\n- decision: resolved\n",
        encoding="utf-8",
    )
    (evaluations / "blocked.md").write_text(
        "# Underwrite an oversized loan\n\n## Context\n\n- loan_amount: 90000\n\n"
        "## Expected\n\n- decision: Loan Amount Cap\n",
        encoding="utf-8",
    )
    (evaluations / "prose-only.md").write_text(
        "# Underwrite anything\n\n## Expected\n\nThe decision must be conservative.\n",
        encoding="utf-8",
    )

    runtime = load_runtime(directory)
    examination = Examiner().examine(runtime, evaluations)

    assert examination.counts() == {"examined": 3, "passed": 2, "failed": 0, "ungraded": 1}
    assert examination.passed is True
    ungraded = next(result for result in examination.results if result.passed is None)
    assert "need a model" in ungraded.rationale

    report = (evaluations / "report.md").read_text(encoding="utf-8")
    assert "Passed: 2" in report and "Ungraded: 1" in report
    assert [record.output for record in runtime.reasoning_log.for_stage("evaluation")] == [
        "passed",
        "ungraded",
        "passed",
    ]


def test_examiner_failed_expectation_fails_the_suite(tmp_path):
    from ear import Examiner

    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    evaluations = directory / "evaluations"
    evaluations.mkdir()
    (evaluations / "impossible.md").write_text(
        "# Underwrite a loan\n\n## Context\n\n- loan_amount: 5000\n\n"
        "## Expected\n\n- decision: an outcome that cannot possibly appear\n",
        encoding="utf-8",
    )
    examination = Examiner().examine(load_runtime(directory), evaluations)
    assert examination.passed is False
    assert examination.results[0].verdict == "FAILED"


# ---------------------------------------------------------------------------
# Optimizer: the trail as the training corpus, reviews as the labels.
# ---------------------------------------------------------------------------


def test_optimizer_builds_trainset_from_jsonl_trail(tmp_path):
    from ear import Optimizer

    runtime = Runtime(name="Trained-Runtime")
    runtime.reasoning_log.path = str(tmp_path / "trail.jsonl")
    runtime.reason(Intent(text="first cycle", context={"loan_amount": 100}))
    runtime.reason(Intent(text="second cycle"))

    trainset = Optimizer().trainset_from_trail(runtime.reasoning_log.path)
    assert len(trainset) == 2
    assert trainset[0].intent == "first cycle"
    assert trainset[0].decision.startswith("[Trained-Runtime]")


def test_optimizer_builds_trainset_from_markdown_trail(tmp_path):
    from ear import Optimizer

    runtime = Runtime(name="MD-Runtime")
    runtime.reasoning_log.path = str(tmp_path / "trail.md")
    runtime.reason(Intent(text="first cycle"))

    trainset = Optimizer().trainset_from_trail(runtime.reasoning_log.path)
    assert len(trainset) == 1
    assert trainset[0].intent == "first cycle"
    assert trainset[0].decision.startswith("[MD-Runtime]")


def test_optimizer_reads_reviewer_verdicts_and_excludes_the_unlabelled(tmp_path):
    from ear import Optimizer

    decisions = tmp_path / "decisions"
    decisions.mkdir()
    (decisions / "reviewed.md").write_text(
        "# Decision -- Underwrite a loan\n\n## Intent\n\n> Underwrite a loan\n\n"
        "## Decision\n\n> Approved at grade B.\n\n"
        "## Review\n\nVerdict: correct\n\n> Well reasoned and within policy.\n",
        encoding="utf-8",
    )
    (decisions / "unreviewed.md").write_text(
        "# Decision -- Another loan\n\n## Decision\n\n> Declined.\n",
        encoding="utf-8",
    )
    (decisions / "ambiguous.md").write_text(
        "# Decision -- A third loan\n\n## Decision\n\n> Approved.\n\n## Review\n\nVerdict: maybe\n",
        encoding="utf-8",
    )

    labelled = Optimizer().verdicts_from_documents(decisions)
    assert len(labelled) == 1
    assert labelled[0].verdict is True
    assert labelled[0].intent == "Underwrite a loan"
    assert labelled[0].decision == "Approved at grade B."
    assert "Well reasoned" in labelled[0].note


def test_optimizer_metric_offline_uses_normalized_containment():
    from types import SimpleNamespace

    from ear import Optimizer

    metric = Optimizer().metric(None)
    assert metric(SimpleNamespace(decision="approve the loan"), "I would approve the loan today.") == 1.0
    assert metric(SimpleNamespace(decision="approve the loan"), "Declined outright.") == 0.0


# ---------------------------------------------------------------------------
# Live: contracts extracted and judged, evaluations graded, by a real model.
# ---------------------------------------------------------------------------

LIVE_MEMORY = (
    "# Memory & Strategy\n\n## Model Selection\n\n"
    "Reason with anthropic/claude-haiku-4-5, reading the credential from\n"
    "ANTHROPIC_API_KEY, at a temperature of 0.2.\n"
)


@requires_anthropic_key
def test_contract_extraction_delivers_typed_data_live(tmp_path):
    from ear.exchange import data_from_decision_document

    stack = dict(MINIMAL_STACK)
    stack["workflow"] = CONTRACT_WORKFLOW
    stack["memory"] = LIVE_MEMORY
    directory = write_stack(tmp_path / "stack", **stack)
    (directory / "intent.md").write_text(
        "# Underwrite a $12,000 personal loan for a strong applicant\n\n"
        "## Context\n\n- loan_amount: 12000\n- credit_score: 785\n- debt_to_income: 0.2\n"
        "- existing_defaults: 0\n",
        encoding="utf-8",
    )

    runtime = load_runtime(directory)
    Exchange(directory).run(runtime)

    record = runtime.reasoning_log.for_stage("contract")[-1]
    assert record.output == "conformant"
    data = data_from_decision_document((directory / "decision.md").read_text(encoding="utf-8"))
    assert "approve" in str(data.get("decision", "")).lower()
    assert data.get("risk grade")


@requires_anthropic_key
def test_examiner_grades_the_example_suite_live(tmp_path, monkeypatch):
    from ear import Examiner

    directory = tmp_path / "credit_risk_stack"
    shutil.copytree(EXAMPLE_STACK, directory)
    memory = (directory / "memory.md").read_text(encoding="utf-8")
    (directory / "memory.md").write_text(
        memory.replace("anthropic/claude-opus-4-8", "anthropic/claude-haiku-4-5"), encoding="utf-8"
    )

    runtime = load_runtime(directory)
    examination = Examiner().examine(runtime, directory / "evaluations")

    assert examination.counts()["ungraded"] == 0
    assert examination.passed is True
    assert (directory / "evaluations" / "report.md").exists()


# ---------------------------------------------------------------------------
# Panels: multi-persona deliberation, native -- the pattern is prose in
# workflow.md, turns and synthesis land on the trail, budgets are code.
# ---------------------------------------------------------------------------

PANEL_STACK = dict(
    MINIMAL_STACK,
    persona=(
        "# Personas\n\n## Credit Risk Guru\nUnderwrite conservatively.\n\n"
        "Skills: risk_grade\n\n"
        "## Customer Advocate\nArgue the applicant's side plainly.\n\nSkills: decide\n"
    ),
    workflow=(
        "# Workflows\n\n## Underwriting Workflow\n\n"
        "Pattern: adversarial debate; the Credit Risk Guru has the last word\n\n"
        "1. Assess the risk. (Credit Risk Guru)\n"
        "2. Make the applicant's case. (Customer Advocate)\n"
    ),
)


def test_pattern_is_authored_as_prose_on_the_workflow(tmp_path):
    runtime = load_runtime(write_stack(tmp_path / "stack", **PANEL_STACK))
    workflow = runtime.processes[0].workflows[0]
    assert workflow.pattern == "adversarial debate; the Credit Risk Guru has the last word"


def test_patterned_workflow_convenes_a_panel_offline(tmp_path):
    runtime = load_runtime(write_stack(tmp_path / "stack", **PANEL_STACK))
    decision = runtime.reason(Intent(text="Underwrite a marginal loan", context={"loan_amount": 5000}))

    turns = runtime.reasoning_log.for_stage("conversation")
    assert len(turns) == 4  # two rounds around a two-persona table
    assert {record.inputs["speaker"] for record in turns} == {"Credit Risk Guru", "Customer Advocate"}
    assert all("no model bound" in record.output for record in turns)

    assert "Credit Risk Guru" in decision and "Customer Advocate" in decision
    assert "no model bound" in decision  # the offline panel never fakes a judgment
    deliberation = runtime.reasoning_log.for_stage("deliberation")[0]
    assert deliberation.inputs["style"] == "adversarial debate; the Credit Risk Guru has the last word"
    assert "[Credit Risk Guru]" in deliberation.inputs["transcript"]


def test_pattern_with_a_single_persona_stays_single_voiced(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["workflow"] = (
        "# Workflows\n\n## Underwriting Workflow\n\nPattern: debate\n\n"
        "1. Decide approve or decline. (Credit Risk Guru)\n"
    )
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    runtime.reason(Intent(text="Underwrite a loan", context={"loan_amount": 5000}))
    assert runtime.reasoning_log.for_stage("conversation") == []


def test_panel_budget_is_enforced_in_code():
    from ear import Panel

    runtime = Runtime(name="Budgeted-Runtime")
    personas = [Persona(name="A"), Persona(name="B")]
    Panel(rounds=50, max_turns=5).convene(runtime, personas, Intent(text="anything"))
    assert len(runtime.reasoning_log.for_stage("conversation")) == 5


@requires_anthropic_key
def test_panel_debates_live(tmp_path):
    stack = dict(PANEL_STACK)
    stack["memory"] = LIVE_MEMORY
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    decision = runtime.reason(
        Intent(
            text="Underwrite a $9,000 personal loan for a marginal applicant",
            context={"loan_amount": 9000, "credit_score": 668, "debt_to_income": 0.41},
        )
    )
    turns = runtime.reasoning_log.for_stage("conversation")
    assert len(turns) == 4
    assert {record.inputs["speaker"] for record in turns} == {"Credit Risk Guru", "Customer Advocate"}
    assert all(record.model == "anthropic/claude-haiku-4-5" for record in turns)
    assert decision and "no model bound" not in decision


# ---------------------------------------------------------------------------
# Journeys: durable, resumable, step-wise execution -- every leg a full
# governed cycle, the state a markdown record, native.
# ---------------------------------------------------------------------------


def test_journey_completes_and_is_settled(tmp_path):
    from ear import Journey

    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    runtime = load_runtime(directory)
    journey = Journey(directory / "journeys" / "loan.md")

    status = journey.run(runtime, Intent(text="Underwrite a consumer loan", context={"loan_amount": 5000}))
    assert status == "completed"
    assert [leg.status for leg in journey.legs] == ["decided", "decided"]
    assert "Credit Risk Runtime" in journey.decision
    record = (directory / "journeys" / "loan.md").read_text(encoding="utf-8")
    assert "Status: completed" in record and "## Leg 2" in record
    assert runtime.reasoning_log.cycle == 2  # every leg was a full governed cycle

    # Settled: running again replays nothing.
    again = Journey(directory / "journeys" / "loan.md")
    assert again.run(runtime) == "completed"
    assert runtime.reasoning_log.cycle == 2


def test_journey_survives_a_crash_and_resumes_where_the_record_ends(tmp_path):
    from ear import Journey

    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    runtime = load_runtime(directory)
    calls = {"count": 0}
    orchestrate = runtime.orchestrator.orchestrate

    def crash_on_second(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("worker died mid-journey")
        return orchestrate(*args, **kwargs)

    runtime.orchestrator.orchestrate = crash_on_second
    journey = Journey(directory / "journeys" / "loan.md")
    with pytest.raises(RuntimeError, match="worker died"):
        journey.run(runtime, Intent(text="Underwrite a consumer loan", context={"loan_amount": 5000}))

    record = (directory / "journeys" / "loan.md").read_text(encoding="utf-8")
    assert "Status: in progress" in record and "## Leg 1" in record and "## Leg 2" not in record

    # A fresh runtime resumes exactly where the record ends.
    recovered = load_runtime(directory)
    resumed = Journey(directory / "journeys" / "loan.md")
    assert resumed.run(recovered) == "completed"
    assert recovered.reasoning_log.cycle == 1  # only the lost leg was walked
    assert resumed.legs[0].decision  # leg one's decision came from the record, not a replay


def test_journey_blocks_hard_and_parks_for_approval(tmp_path):
    from ear import Approval, Journey

    directory = approval_stack(tmp_path)
    runtime = load_runtime(directory)
    journey = Journey(directory / "journeys" / "big-loan.md")
    status = journey.run(runtime, Intent(text="Underwrite a large loan", context={"loan_amount": 60000}))
    assert status == "PENDING APPROVAL"
    cycles_when_parked = runtime.reasoning_log.cycle

    # Without a verdict the journey stays parked and replays nothing.
    assert Journey(journey.path).run(runtime) == "PENDING APPROVAL"
    assert runtime.reasoning_log.cycle == cycles_when_parked

    released = Journey(journey.path)
    assert released.run(runtime, approval=Approval(verdict=True, approver="lakshmi@gigkri.com")) == "completed"
    assert "approved by lakshmi@gigkri.com" in [r.output for r in runtime.reasoning_log.for_stage("approval")]

    blocked = Journey(directory / "journeys" / "huge-loan.md")
    assert blocked.run(runtime, Intent(text="Underwrite a huge loan", context={"loan_amount": 90000})) == "BLOCKED"
    assert "Status: BLOCKED" in blocked.path.read_text(encoding="utf-8")


def test_journey_refuses_to_resume_over_a_changed_stack(tmp_path):
    from ear import Journey

    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    runtime = load_runtime(directory)
    journey = Journey(directory / "journeys" / "loan.md")
    journey.run(runtime, Intent(text="Underwrite a consumer loan", context={"loan_amount": 5000}))

    changed = (directory / "workflow.md").read_text(encoding="utf-8").replace(
        "Band the profile and assign a risk grade.", "Do something entirely different."
    )
    (directory / "workflow.md").write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="forge the record"):
        Journey(journey.path).run(load_runtime(directory))


def test_journey_first_run_needs_an_intent_and_a_stack(tmp_path):
    from ear import Journey

    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    runtime = load_runtime(directory)
    with pytest.raises(ValueError, match="needs an intent"):
        Journey(directory / "journeys" / "empty.md").run(runtime)
    with pytest.raises(ValueError, match="no workflow steps"):
        Journey(tmp_path / "nowhere.md").run(Runtime(name="Empty-Runtime"), Intent(text="x"))


# ---------------------------------------------------------------------------
# LangGraph: the stack compiled to a graph -- every node a full governed
# cycle, halting at the first non-decided status, checkpointable.
# ---------------------------------------------------------------------------


def test_compiled_graph_runs_each_step_as_a_governed_cycle(tmp_path):
    pytest.importorskip("langgraph", reason="langgraph not installed")
    from ear.integrations.langgraph_backend import compile_to_graph

    runtime = load_runtime(write_stack(tmp_path / "stack", **MINIMAL_STACK))
    graph = compile_to_graph(runtime)
    final = graph.invoke({"intent": "Underwrite a consumer loan", "context": {"loan_amount": 5000}})

    assert final["status"] == "decided"
    assert [entry["step"] for entry in final["steps"]] == [
        "Band the profile and assign a risk grade.",
        "Decide approve or decline against the grade.",
    ]
    assert "Credit Risk Runtime" in final["decision"]
    # Second step reasoned with the first step's conclusion in view.
    assert "Earlier steps concluded" not in final["steps"][0]["decision"]
    # Each node was a full governed cycle: two cycles on the trail, two
    # memories, policies judged in both.
    assert runtime.reasoning_log.cycle == 2
    assert len(runtime.memory.working) == 2
    assert len(runtime.reasoning_log.for_stage("policy")) >= 2


def test_compiled_graph_halts_at_the_first_blocked_step(tmp_path):
    pytest.importorskip("langgraph", reason="langgraph not installed")
    from ear.integrations.langgraph_backend import compile_to_graph

    runtime = load_runtime(write_stack(tmp_path / "stack", **MINIMAL_STACK))
    final = compile_to_graph(runtime).invoke(
        {"intent": "Underwrite an oversized loan", "context": {"loan_amount": 90000}}
    )
    assert final["status"] == "BLOCKED"
    assert "Loan Amount Cap" in final["decision"]
    assert len(final["steps"]) == 1  # the gate stopped the graph, not just the step


def test_compiled_graph_checkpoints_between_steps(tmp_path):
    pytest.importorskip("langgraph", reason="langgraph not installed")
    from langgraph.checkpoint.memory import MemorySaver

    from ear.integrations.langgraph_backend import compile_to_graph

    runtime = load_runtime(write_stack(tmp_path / "stack", **MINIMAL_STACK))
    graph = compile_to_graph(runtime, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "underwriting-1"}}
    graph.invoke({"intent": "Underwrite a consumer loan", "context": {"loan_amount": 5000}}, config)

    saved = graph.get_state(config).values
    assert saved["status"] == "decided"
    assert len(saved["steps"]) == 2


def test_runtime_node_routes_governance_as_status(tmp_path):
    pytest.importorskip("langgraph", reason="langgraph not installed")
    from langgraph.graph import END, StateGraph

    from ear.integrations.langgraph_backend import StackState, runtime_node

    runtime = load_runtime(approval_stack(tmp_path))
    graph = StateGraph(StackState)
    graph.add_node("ear", runtime_node(runtime))
    graph.set_entry_point("ear")
    graph.add_edge("ear", END)
    compiled = graph.compile()

    decided = compiled.invoke({"intent": "Underwrite a small loan", "context": {"loan_amount": 5000}})
    assert decided["status"] == "decided"

    parked = compiled.invoke({"intent": "Underwrite a large loan", "context": {"loan_amount": 60000}})
    assert parked["status"] == "PENDING APPROVAL"
    assert "Large Loan Human Approval" in parked["decision"]


def test_compile_to_graph_refuses_an_empty_stack():
    pytest.importorskip("langgraph", reason="langgraph not installed")
    from ear.integrations.langgraph_backend import compile_to_graph

    with pytest.raises(ValueError, match="no workflow steps"):
        compile_to_graph(Runtime(name="Empty-Runtime"))


# ---------------------------------------------------------------------------
# Executable tools: declarations meet handlers in the ToolBinder, the model
# decides when to call, and every invocation lands on the trail.
# ---------------------------------------------------------------------------

TOOLS_MEMORY = (
    "# Memory & Strategy\n\n## Tools\n\n"
    "- amortization_calculator: computes the monthly payment for an amount, an annual rate in percent, and a term in months\n"
)


def monthly_payment(amount: float, annual_rate_percent: float, months: int) -> float:
    """Compute a fixed monthly payment for a loan."""
    monthly_rate = annual_rate_percent / 100 / 12
    if monthly_rate == 0:
        return round(amount / months, 2)
    factor = (1 + monthly_rate) ** months
    return round(amount * monthly_rate * factor / (factor - 1), 2)


def test_tool_binder_resolves_declared_tools_and_rejects_strays(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["memory"] = TOOLS_MEMORY
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    runtime.bind_tool("amortization_calculator", monthly_payment)

    bound = runtime.tool_binder.bound_tools(runtime)
    assert [tool.name for tool in bound] == ["amortization_calculator"]
    assert "monthly payment" in bound[0].description  # the declared description, not the docstring
    assert bound[0].identifier == "amortization_calculator"

    runtime.bind_tool("undeclared_gadget", monthly_payment)
    with pytest.raises(ValueError, match="matches nothing the stack declares.*amortization_calculator"):
        runtime.tool_binder.bound_tools(runtime)


def test_handler_skills_bind_automatically_and_bindings_override():
    from ear import Persona, Skill, ToolBinder, Workflow

    persona = Persona(name="Analyst")
    persona.add_skill(Skill(name="fetch_score", prompt="Fetch the credit score.", handler=lambda applicant: 700))
    workflow = Workflow(name="W")
    workflow.add_persona(persona)
    runtime = Runtime(name="Skilled-Runtime")

    binder = ToolBinder()
    bound = binder.bound_tools(runtime, plan=[workflow])
    assert [tool.name for tool in bound] == ["fetch_score"]
    assert bound[0].handler("anyone") == 700

    # An explicit binding for the same skill overrides its own handler;
    # explicit bindings resolve against the runtime's whole stack, so the
    # workflow joins a process first.
    from ear import Process

    holder = Process(name="P")
    holder.add_workflow(workflow)
    runtime.add_process(holder)
    binder.bind("fetch_score", lambda applicant: 750)
    bound = binder.bound_tools(runtime, plan=[workflow])
    assert bound[0].handler("anyone") == 750


def test_offline_deliberation_lists_tools_without_invoking_them(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["memory"] = TOOLS_MEMORY
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    runtime.bind_tool("amortization_calculator", monthly_payment)
    runtime.reason(Intent(text="Underwrite a consumer loan", context={"loan_amount": 5000}))

    deliberation = runtime.reasoning_log.for_stage("deliberation")[0]
    assert deliberation.inputs["tools"] == ["amortization_calculator"]
    assert runtime.reasoning_log.for_stage("tool") == []


def test_logged_tool_wrapper_contains_failures_on_the_record():
    from ear.tool_binder import BoundTool, ToolBinder

    def broken(query: str) -> str:
        raise RuntimeError("bureau offline")

    runtime = Runtime(name="Contained-Runtime")
    wrapped = ToolBinder._logged(runtime, BoundTool(name="credit_bureau_check", description="checks", handler=broken))
    result = wrapped("anything")
    assert "failed: bureau offline" in str(result)
    record = runtime.reasoning_log.for_stage("tool")[0]
    assert record.output.startswith("FAILED")
    assert record.inputs["tool"] == "credit_bureau_check"
    assert "duration_ms" in record.inputs


def test_langchain_adapter_binds_duck_typed_tools(tmp_path):
    from ear.integrations.langchain_backend import bind_langchain_tool

    class InvokeTool:
        name = "amortization_calculator"
        description = "computes payments"

        def invoke(self, query):
            return f"invoked: {query}"

    class RunTool:
        name = "amortization_calculator"
        description = "computes payments"

        def run(self, query):
            return f"ran: {query}"

    stack = dict(MINIMAL_STACK)
    stack["memory"] = TOOLS_MEMORY
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))

    bind_langchain_tool(runtime.tool_binder, InvokeTool())
    bound = runtime.tool_binder.bound_tools(runtime)
    assert bound[0].handler("q") == "invoked: q"

    bind_langchain_tool(runtime.tool_binder, RunTool())
    bound = runtime.tool_binder.bound_tools(runtime)
    assert bound[0].handler("q") == "ran: q"


def test_deliberator_backend_seam_stays_on_the_trail():
    class TypedBackend:
        def deliberate(self, runtime, intent, plan=None, research=None):
            return "typed decision from the backend"

    runtime = Runtime(name="Seamed-Runtime")
    runtime.orchestrator.executor.performer.deliberator.backend = TypedBackend()
    decision = runtime.reason(Intent(text="anything"))
    assert decision == "typed decision from the backend"
    record = runtime.reasoning_log.for_stage("deliberation")[0]
    assert record.model == "backend:TypedBackend"
    assert record.output == decision


@requires_anthropic_key
def test_react_invokes_the_bound_calculator_live(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["memory"] = TOOLS_MEMORY + (
        "\n## Model Selection\n\nReason with anthropic/claude-haiku-4-5, reading the credential from\n"
        "ANTHROPIC_API_KEY, at a temperature of 0.2.\n"
    )
    directory = write_stack(tmp_path / "stack", **stack)
    runtime = load_runtime(directory)
    runtime.bind_tool("amortization_calculator", monthly_payment)

    decision = runtime.reason(
        Intent(
            text=(
                "Underwrite a $12,000 personal loan at 6 percent annual interest over 36 months; "
                "compute the exact monthly payment with the amortization calculator before deciding."
            ),
            context={"loan_amount": 12000, "credit_score": 760, "debt_to_income": 0.2},
        )
    )
    assert decision
    invocations = runtime.reasoning_log.for_stage("tool")
    assert invocations, "the model never called the bound tool"
    assert invocations[0].inputs["tool"] == "amortization_calculator"
    assert not invocations[0].output.startswith("FAILED")


# ---------------------------------------------------------------------------
# Approval gates: a violated Approval-required policy parks the cycle for
# a human verdict instead of blocking it; the verdict is a markdown
# document, and everything lands on the trail.
# ---------------------------------------------------------------------------

APPROVAL_POLICY = (
    "# Policies\n\n## Loan Amount Cap\nThe loan must not exceed $75,000.\n\n"
    "Fallback: loan_amount <= 75000\nApplies to: runtime\n\n"
    "## Large Loan Human Approval\nLoans above $50,000 need a human approver.\n\n"
    "Fallback: loan_amount <= 50000\nApproval: required\nApplies to: runtime\n"
)


def approval_stack(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["policy"] = APPROVAL_POLICY
    stack["workflow"] = (
        "# Workflows\n\n## Underwriting Workflow\n\n"
        "1. Decide approve or decline. (Credit Risk Guru)\n"
    )
    return write_stack(tmp_path / "stack", **stack)


def test_loader_reads_the_approval_field_and_rejects_the_unreadable(tmp_path):
    runtime = load_runtime(approval_stack(tmp_path))
    by_name = {policy.name: policy for policy in runtime.policies}
    assert by_name["Large Loan Human Approval"].approval_required is True
    assert by_name["Loan Amount Cap"].approval_required is False

    stack = dict(MINIMAL_STACK)
    stack["workflow"] = "# Workflows\n\n## Underwriting Workflow\n\n1. Decide. (Credit Risk Guru)\n"
    stack["policy"] = "# Policies\n\n## Odd Gate\nSomething.\n\nApproval: perhaps\n"
    with pytest.raises(ValueError, match="unreadable Approval field"):
        load_runtime(write_stack(tmp_path / "unreadable", **stack))

    stack["policy"] = "# Policies\n\n## Ungated\nSomething.\n\nApproval: not required\n"
    runtime = load_runtime(write_stack(tmp_path / "negated", **stack))
    assert runtime.policies[0].approval_required is False


def test_violated_approval_gate_parks_the_cycle_on_the_record(tmp_path):
    from ear import ApprovalRequired

    runtime = load_runtime(approval_stack(tmp_path))
    with pytest.raises(ApprovalRequired, match="Large Loan Human Approval") as parked:
        runtime.reason(Intent(text="Underwrite a large loan", context={"loan_amount": 60000}))

    assert isinstance(parked.value, PermissionError)  # existing handlers keep working
    policy_record = next(
        record for record in runtime.reasoning_log.for_stage("policy")
        if record.inputs["policy"] == "Large Loan Human Approval"
    )
    assert policy_record.output == "PENDING APPROVAL"
    pending = runtime.reasoning_log.for_stage("approval")[0]
    assert "PENDING -- human approval required" in pending.output
    assert runtime.reasoning_log.for_stage("usage")  # a parked cycle is accounted too
    assert len(runtime.memory) == 0  # nothing was decided, nothing is remembered


def test_approved_verdict_releases_the_gate_and_is_on_the_record(tmp_path):
    from ear import Approval

    runtime = load_runtime(approval_stack(tmp_path))
    approval = Approval(verdict=True, approver="lakshmi@gigkri.com", note="Collateral covers it.")
    decision = runtime.reason(Intent(text="Underwrite a large loan", context={"loan_amount": 60000}), approval=approval)

    assert decision
    released = runtime.reasoning_log.for_stage("approval")[0]
    assert released.output == "approved by lakshmi@gigkri.com"
    assert "Collateral covers it." in released.rationale
    assert len(runtime.memory) == 1


def test_rejected_verdict_blocks_like_any_violation(tmp_path):
    from ear import Approval, ApprovalRequired

    runtime = load_runtime(approval_stack(tmp_path))
    approval = Approval(verdict=False, approver="lakshmi@gigkri.com")
    with pytest.raises(PermissionError, match="Large Loan Human Approval") as blocked:
        runtime.reason(Intent(text="Underwrite a large loan", context={"loan_amount": 60000}), approval=approval)
    assert not isinstance(blocked.value, ApprovalRequired)  # a rejection is a block, not a park
    assert runtime.reasoning_log.for_stage("approval")[0].output == "REJECTED by lakshmi@gigkri.com"


def test_hard_block_wins_over_a_pending_gate(tmp_path):
    runtime = load_runtime(approval_stack(tmp_path))
    from ear import ApprovalRequired

    with pytest.raises(PermissionError, match="Loan Amount Cap") as blocked:
        runtime.reason(Intent(text="Underwrite a huge loan", context={"loan_amount": 90000}))
    assert not isinstance(blocked.value, ApprovalRequired)


def test_approval_document_round_trips():
    from ear import Approval

    text = Approval(verdict=True, approver="lakshmi@gigkri.com", note="Fine by me.").to_markdown("A big loan")
    read = Approval.from_markdown(text)
    assert read.verdict is True
    assert read.approver == "lakshmi@gigkri.com"
    assert read.note == "Fine by me."
    assert Approval.from_markdown("# Approval\n\nVerdict: perhaps\n").verdict is None


def test_exchange_parks_releases_and_stays_idempotent(tmp_path):
    directory = approval_stack(tmp_path)
    (directory / "intents").mkdir()
    (directory / "intents" / "big-loan.md").write_text(
        "# Underwrite a $60,000 loan\n\n## Context\n\n- loan_amount: 60000\n",
        encoding="utf-8",
    )
    runtime = load_runtime(directory)
    exchange = Exchange(directory)

    exchange.run(runtime)
    parked = (directory / "decisions" / "big-loan.md").read_text(encoding="utf-8")
    assert "Status: PENDING APPROVAL" in parked
    assert "## Awaiting approval" in parked and "Large Loan Human Approval" in parked

    # No approval document yet: nothing to do.
    assert exchange.run(runtime) == []

    (directory / "approvals").mkdir()
    (directory / "approvals" / "big-loan.md").write_text(
        "# Approval -- Underwrite a $60,000 loan\n\n"
        "Verdict: approved\nApprover: lakshmi@gigkri.com\n\n> Collateral covers it.\n",
        encoding="utf-8",
    )
    written = exchange.run(runtime)
    assert [path.name for path in written] == ["big-loan.md"]
    released = (directory / "decisions" / "big-loan.md").read_text(encoding="utf-8")
    assert "Status: decided" in released
    assert "## Approval" in released and "lakshmi@gigkri.com" in released

    # Released cycles are settled: a third run replays nothing.
    assert exchange.run(runtime) == []


def test_exchange_writes_blocked_document_on_a_rejected_verdict(tmp_path):
    directory = approval_stack(tmp_path)
    (directory / "intent.md").write_text(
        "# Underwrite a $60,000 loan\n\n## Context\n\n- loan_amount: 60000\n", encoding="utf-8"
    )
    runtime = load_runtime(directory)
    exchange = Exchange(directory)
    exchange.run(runtime)
    (directory / "approval.md").write_text("# Approval\n\nVerdict: rejected\nApprover: lakshmi\n", encoding="utf-8")
    exchange.run(runtime)
    final = (directory / "decision.md").read_text(encoding="utf-8")
    assert "Status: BLOCKED" in final and "Large Loan Human Approval" in final


def test_exchange_refuses_an_unreadable_approval_verdict(tmp_path):
    directory = approval_stack(tmp_path)
    (directory / "intent.md").write_text(
        "# Underwrite a $60,000 loan\n\n## Context\n\n- loan_amount: 60000\n", encoding="utf-8"
    )
    runtime = load_runtime(directory)
    exchange = Exchange(directory)
    exchange.run(runtime)
    (directory / "approval.md").write_text("# Approval\n\nVerdict: perhaps\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no readable Verdict"):
        exchange.run(runtime)


@requires_anthropic_key
def test_approval_gate_parks_and_releases_live(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["policy"] = APPROVAL_POLICY
    stack["memory"] = LIVE_MEMORY
    directory = write_stack(tmp_path / "stack", **stack)
    (directory / "intent.md").write_text(
        "# Underwrite a $62,000 personal loan for a strong applicant\n\n"
        "## Context\n\n- loan_amount: 62000\n- credit_score: 790\n- debt_to_income: 0.15\n",
        encoding="utf-8",
    )
    runtime = load_runtime(directory)
    exchange = Exchange(directory)
    exchange.run(runtime)
    assert "Status: PENDING APPROVAL" in (directory / "decision.md").read_text(encoding="utf-8")

    (directory / "approval.md").write_text(
        "# Approval\n\nVerdict: approved\nApprover: lakshmi@gigkri.com\n\n> Reviewed; proceed.\n",
        encoding="utf-8",
    )
    exchange.run(runtime)
    released = (directory / "decision.md").read_text(encoding="utf-8")
    assert "Status: decided" in released and "## Approval" in released
    approvals = [record.output for record in runtime.reasoning_log.for_stage("approval")]
    assert "approved by lakshmi@gigkri.com" in approvals


# ---------------------------------------------------------------------------
# Observability: the trail fans out to exporters, cycles carry usage, and
# the OpenTelemetry adapter maps records to spans (one cycle per trace).
# ---------------------------------------------------------------------------


class CollectingExporter:
    def __init__(self, fail: bool = False):
        self.records = []
        self.flushes = 0
        self.fail = fail

    def export(self, record):
        if self.fail:
            raise RuntimeError("exporter down")
        self.records.append(record)

    def flush(self):
        self.flushes += 1


def test_reasoning_log_fans_out_to_exporters_exactly_once():
    exporter = CollectingExporter()
    runtime = Runtime(name="Exported-Runtime")
    runtime.reasoning_log.exporters.append(exporter)

    runtime.reason(Intent(text="first cycle"))
    assert [record.stage for record in exporter.records][0] == "intent"
    assert exporter.records[-1].stage == "usage"
    assert exporter.flushes == 1

    seen = len(exporter.records)
    runtime.reasoning_log.flush()  # nothing pending -- no double export
    assert len(exporter.records) == seen


def test_exporter_failure_never_breaks_a_cycle(tmp_path):
    runtime = Runtime(name="Resilient-Runtime")
    runtime.reasoning_log.path = str(tmp_path / "trail.md")
    runtime.reasoning_log.exporters.append(CollectingExporter(fail=True))

    decision = runtime.reason(Intent(text="first cycle"))
    assert decision  # the cycle completed
    assert runtime.reasoning_log.export_errors
    assert "exporter down" in runtime.reasoning_log.export_errors[0]
    # The file on disk stays the canonical record.
    assert "## Cycle 1" in (tmp_path / "trail.md").read_text(encoding="utf-8")


def test_usage_record_closes_every_cycle_offline():
    runtime = Runtime(name="Accounted-Runtime")
    runtime.reason(Intent(text="a cycle"))
    usage = runtime.reasoning_log.for_stage("usage")[0]
    assert "0 model calls" in usage.output and "ms" in usage.output
    assert usage.inputs["model_calls"] == 0
    assert isinstance(usage.inputs["latency_ms"], int)


def test_usage_is_recorded_on_blocked_cycles_too(tmp_path):
    runtime = load_runtime(write_stack(tmp_path / "stack", **MINIMAL_STACK))
    with pytest.raises(PermissionError):
        runtime.reason(Intent(text="Underwrite", context={"debt_to_income": 0.60}))
    assert runtime.reasoning_log.for_stage("usage")


def test_otel_exporter_maps_cycles_to_traces_and_records_to_spans():
    sdk = pytest.importorskip("opentelemetry.sdk.trace", reason="opentelemetry-sdk not installed")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from ear.integrations.otel_backend import OpenTelemetryExporter

    memory_exporter = InMemorySpanExporter()
    provider = sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))

    runtime = Runtime(name="Traced-Runtime")
    runtime.reasoning_log.exporters.append(OpenTelemetryExporter(tracer_provider=provider))
    runtime.reason(Intent(text="a traced cycle"))

    spans = memory_exporter.get_finished_spans()
    by_name = {span.name for span in spans}
    assert {"cycle 1", "intent", "deliberation", "explanation", "usage"} <= by_name
    root = next(span for span in spans if span.name == "cycle 1")
    children = [span for span in spans if span.name != "cycle 1"]
    assert all(span.context.trace_id == root.context.trace_id for span in children)
    deliberation = next(span for span in spans if span.name == "deliberation")
    assert deliberation.attributes["ear.cycle"] == 1
    assert "a traced cycle" in deliberation.attributes["ear.inputs"]


def test_loader_attaches_otel_exporter_when_the_audit_prose_names_it(tmp_path):
    pytest.importorskip(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        reason="opentelemetry-exporter-otlp not installed",
    )
    stack = dict(MINIMAL_STACK)
    stack["memory"] = (
        "## Reasoning Audit Trail\nLog every reasoning step to `.ear/reasoning.md` "
        "and export the trail over OpenTelemetry as well.\n"
    )
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    assert [type(exporter).__name__ for exporter in runtime.reasoning_log.exporters] == ["OpenTelemetryExporter"]


# ---------------------------------------------------------------------------
# Knowledge and the Librarian: declared sources, retrieval on the record,
# citations in evidence and decision documents.
# ---------------------------------------------------------------------------

KNOWLEDGE_MEMORY = (
    "# Memory & Strategy\n\n## Knowledge\n\n"
    "The reference material the Librarian may consult and cite.\n\n"
    "- underwriting manual: `knowledge/manual.md`\n"
)

MANUAL = (
    "# Underwriting Manual\n\n## Section 4.2 -- Marginal applicants\n\n"
    "A grade C applicant whose debt-to-income band is not low must be declined.\n\n"
    "## Section 9.9 -- Office plants\n\nWater the plants on Fridays.\n"
)


def knowledge_stack(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["memory"] = KNOWLEDGE_MEMORY
    directory = write_stack(tmp_path / "stack", **stack)
    (directory / "knowledge").mkdir()
    (directory / "knowledge" / "manual.md").write_text(MANUAL, encoding="utf-8")
    return directory


def test_strategy_reads_knowledge_sources_and_rejects_urls():
    strategy = Strategy.from_markdown(KNOWLEDGE_MEMORY)
    assert strategy.knowledge_sources == [("underwriting manual", "knowledge/manual.md")]
    assert "reference material" in strategy.knowledge

    with pytest.raises(ValueError, match="URL sources are not supported"):
        Strategy.from_markdown("## Knowledge\n\n- manual: https://example.com/manual\n")


def test_loader_builds_knowledge_chunked_by_section(tmp_path):
    runtime = load_runtime(knowledge_stack(tmp_path))
    knowledge = runtime.librarian.knowledge
    assert knowledge is not None and len(knowledge) == 2
    assert knowledge.passages[0].source == "underwriting manual -- manual.md § Section 4.2 -- Marginal applicants"
    assert "must be declined" in knowledge.passages[0].text


def test_loader_fails_loudly_on_a_missing_knowledge_source(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["memory"] = "## Knowledge\n\n- manual: `knowledge/absent.md`\n"
    with pytest.raises(ValueError, match="matched no files"):
        load_runtime(write_stack(tmp_path / "stack", **stack))


def test_librarian_retrieves_structurally_offline_with_citations(tmp_path):
    directory = knowledge_stack(tmp_path)
    (directory / "intent.md").write_text(
        "# Underwrite a marginal grade C applicant whose debt-to-income band is not low\n\n"
        "## Context\n\n- loan_amount: 9000\n",
        encoding="utf-8",
    )
    runtime = load_runtime(directory)
    Exchange(directory).run(runtime)

    retrieval = runtime.reasoning_log.for_stage("retrieval")[0]
    assert "structural retrieval" in retrieval.rationale
    assert "Section 4.2" in retrieval.output

    evidence = runtime.memory.working[-1].evidence
    assert any("Section 4.2" in citation for citation in evidence.sources["citations"])
    decision_document = (directory / "decision.md").read_text(encoding="utf-8")
    assert "## Sources" in decision_document and "underwriting manual" in decision_document

    deliberation = runtime.reasoning_log.for_stage("deliberation")[0]
    assert "never as" in deliberation.inputs["knowledge"]
    assert "must be declined" in deliberation.inputs["knowledge"]


def test_llamaindex_adapter_maps_nodes_by_duck_typing():
    from ear.integrations.llamaindex_backend import LlamaIndexRetriever
    from ear import Passage

    class StubNode:
        def __init__(self, text, metadata):
            self.text = text
            self.metadata = metadata

    class StubWrapper:
        def __init__(self, node):
            self.node = node

    class StubRetriever:
        def retrieve(self, query):
            return [
                StubWrapper(StubNode("Section one text.", {"file_name": "manual.md"})),
                StubNode("Loose node text.", {}),
            ]

    passages = LlamaIndexRetriever(StubRetriever(), source_label="corpus").retrieve("anything")
    assert passages == [
        Passage(source="manual.md", text="Section one text."),
        Passage(source="corpus", text="Loose node text."),
    ]

    # And the same object drops straight into the Librarian's seam.
    runtime = Runtime(name="Retriever-Runtime")
    runtime.librarian.retriever = LlamaIndexRetriever(StubRetriever())
    research = runtime.librarian.research(runtime, Intent(text="anything"))
    assert research is not None and research.citations[0] == "manual.md"
    assert runtime.reasoning_log.for_stage("retrieval")


@requires_anthropic_key
def test_retrieval_cites_the_manual_live(tmp_path):
    directory = tmp_path / "credit_risk_stack"
    shutil.copytree(EXAMPLE_STACK, directory)
    memory = (directory / "memory.md").read_text(encoding="utf-8")
    (directory / "memory.md").write_text(
        memory.replace("anthropic/claude-opus-4-8", "anthropic/claude-haiku-4-5"), encoding="utf-8"
    )
    (directory / "intents").mkdir()
    # A unique reference defeats the LM's prompt cache, so this test
    # genuinely exercises live usage accounting, not a cached replay.
    reference = int(time.time())
    (directory / "intents" / "marginal.md").write_text(
        f"# Underwrite a $9,500 personal loan for a marginal applicant (reference {reference})\n\n"
        "## Context\n\n- loan_amount: 9500\n- credit_score: 665\n- debt_to_income: 0.41\n"
        "- annual_income: 52000\n- existing_defaults: 0\n",
        encoding="utf-8",
    )
    runtime = load_runtime(directory)
    Exchange(directory).run(runtime)

    retrieval = runtime.reasoning_log.for_stage("retrieval")[-1]
    assert retrieval.model == "anthropic/claude-haiku-4-5"
    decision_document = (directory / "decisions" / "marginal.md").read_text(encoding="utf-8")
    assert "## Sources" in decision_document and "underwriting manual" in decision_document

    usage = runtime.reasoning_log.for_stage("usage")[-1]
    assert usage.inputs["model_calls"] > 0
    assert usage.inputs["input_tokens"] > 0


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
    assert [policy.name for policy in workflow.policies] == ["Loan Amount Cap", "Large Loan Human Approval"]
    assert workflow.policies[1].approval_required is True

    decision = runtime.reason(
        Intent(
            text="Underwrite a $20,000 consumer loan",
            context={"loan_amount": 20000, "debt_to_income": 0.30, "credit_score": 720},
        )
    )
    assert "Credit Risk Guru" in decision

    with pytest.raises(PermissionError, match="Loan Amount Cap"):
        runtime.reason(Intent(text="Underwrite a large loan", context={"loan_amount": 90000}))
