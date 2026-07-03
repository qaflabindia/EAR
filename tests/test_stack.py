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
    KnowledgeSource,
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

    assert examination.counts() == {
        "examined": 3,
        "passed": 2,
        "failed": 0,
        "ungraded": 1,
        "criteria": 0,
        "criteria_passed": 0,
        "criteria_failed": 0,
    }
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
    from ear import Optimizer

    metric = Optimizer().metric(None)
    assert metric("approve the loan", "I would approve the loan today.") == 1.0
    assert metric("approve the loan", "Declined outright.") == 0.0


def test_optimizer_refine_is_a_no_op_without_a_model():
    from ear import Optimizer
    from ear.optimizer import Example
    from ear.signatures import ReasonAboutIntent

    before = ReasonAboutIntent.instruction
    result = Optimizer().refine(ReasonAboutIntent, [Example(intent="x", decision="y")], model_binding=None)
    assert result == before  # reflection is a judgment; no model, no change


def test_native_judgment_parses_markdown_sections_from_a_stub_lm():
    from ear.judgment import Field, Judgment

    class StubLM:
        def complete(self, prompt, system=""):
            return "## complies\n\nyes\n\n## rationale\n\nThe amount is within the limit.\n"

    judgment = Judgment(
        instruction="Judge it.",
        inputs=[Field("context")],
        outputs=[Field("complies", kind="bool"), Field("rationale", kind="text")],
    )
    result = judgment.run(StubLM(), context="amount 5")
    assert result.complies is True
    assert "within the limit" in result.rationale


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
    # Speakers are judged now, and consensus may conclude the panel before
    # the budget of four -- but never before both sides have spoken.
    assert 2 <= len(turns) <= 4
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
# N1: LM hardening, per-judgment accounting, pricing, instruction search,
# demos and persisted instructions -- reasoning & optimization depth.
# ---------------------------------------------------------------------------


class FakeReply:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


ANTHROPIC_REPLY = {
    "content": [{"type": "text", "text": "## decision\n\nfine\n"}],
    "usage": {"input_tokens": 12, "output_tokens": 5},
}


def test_lm_retries_transient_failures_and_records_them(monkeypatch):
    import urllib.error

    from ear.llm import LM

    attempts = {"count": 0}

    def flaky_urlopen(request, context=None, timeout=None):
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise urllib.error.HTTPError(request.full_url, 529, "overloaded", {}, None)
        return FakeReply(ANTHROPIC_REPLY)

    monkeypatch.setattr("ear.llm.urllib.request.urlopen", flaky_urlopen)
    monkeypatch.setattr("ear.llm.time.sleep", lambda seconds: None)

    lm = LM(model="anthropic/test-model", api_key="k")
    reply = lm.complete("anything")
    assert "fine" in reply
    assert attempts["count"] == 3
    entry = lm.history[-1]
    assert entry["retries"] == 2
    assert entry["usage"] == {"prompt_tokens": 12, "completion_tokens": 5}
    assert entry["latency_ms"] >= 0


def test_lm_fails_fast_on_non_retryable_errors(monkeypatch):
    import urllib.error

    from ear.llm import LM, LMError

    attempts = {"count": 0}

    def unauthorized(request, context=None, timeout=None):
        attempts["count"] += 1
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr("ear.llm.urllib.request.urlopen", unauthorized)
    with pytest.raises(LMError, match="401"):
        LM(model="anthropic/test-model", api_key="bad").complete("anything")
    assert attempts["count"] == 1  # auth errors never retry


class ScriptedLM:
    """A stub LM answering with fixed markdown sections and recording
    history the way the real one does."""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.history = []
        self.prompts = []

    def complete(self, prompt, system=""):
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if self.replies else "## decision\n\nok\n"
        self.history.append(
            {"usage": {"prompt_tokens": 10, "completion_tokens": 3}, "latency_ms": 7, "retries": 0}
        )
        return reply


def test_per_judgment_usage_rides_the_stage_record():
    from ear import ModelBinding, Policy

    binding = ModelBinding(provider="anthropic", model="test")
    binding.lm = ScriptedLM(["## complies\n\nyes\n\n## rationale\n\nWithin the cap.\n"])
    runtime = Runtime(name="Accounted-Runtime", model_binding=binding)
    runtime.add_policy(Policy(name="Cap", statement="Stay under the cap."))

    runtime.governor.govern(runtime, Intent(text="check", context={"amount": 5}))
    record = runtime.reasoning_log.for_stage("policy")[0]
    assert record.input_tokens == 10 and record.output_tokens == 3
    assert record.latency_ms == 7
    assert "10+3 tok" in record.render()


def test_fallback_judgments_are_never_billed(tmp_path):
    runtime = load_runtime(write_stack(tmp_path / "stack", **MINIMAL_STACK))
    runtime.reason(Intent(text="Underwrite a consumer loan", context={"loan_amount": 5000}))
    for record in runtime.reasoning_log.records:
        assert record.input_tokens == 0 and record.output_tokens == 0


def test_pricing_is_declared_in_prose_and_prices_usage():
    strategy = Strategy.from_markdown(
        "## Pricing\n\nInput tokens cost $3 per million; output tokens cost $15 per million.\n"
    )
    assert strategy.input_rate_per_million == 3.0
    assert strategy.output_rate_per_million == 15.0
    assert strategy.dollars(1_000_000, 200_000) == pytest.approx(6.0)
    assert Strategy().dollars(1000, 1000) is None  # undeclared -> never invented


def test_usage_record_carries_dollars_only_when_priced():
    from ear import ModelBinding

    binding = ModelBinding(provider="anthropic", model="test")
    binding.lm = ScriptedLM()
    binding.lm.history.append({"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}, "retries": 1})
    runtime = Runtime(name="Priced-Runtime", model_binding=binding)
    runtime.strategy = Strategy.from_markdown("## Pricing\n\nInput tokens cost $3 per million.\n")

    runtime._record_usage(started=time.monotonic(), calls_before=0)
    usage = runtime.reasoning_log.for_stage("usage")[0]
    assert usage.inputs["cost"] == pytest.approx(3.0)
    assert "~$3.0" in usage.output and "1 retried" in usage.output


def test_search_refuses_without_a_model():
    from ear import Optimizer
    from ear.judgment import Field, Judgment
    from ear.optimizer import Example

    judgment = Judgment(instruction="Decide.", inputs=[Field("intent")], outputs=[Field("decision")])
    with pytest.raises(ValueError, match="optimization is judgment"):
        Optimizer().search(judgment, [Example(intent="x", decision="y")], model_binding=None)


def test_demos_render_into_the_prompt_in_answer_shape():
    from ear.judgment import Field, Judgment

    judgment = Judgment(
        instruction="Decide.",
        inputs=[Field("intent")],
        outputs=[Field("decision")],
        demos=[{"intent": "a small loan", "decision": "approve"}],
    )
    prompt = judgment.render_prompt({"intent": "a big loan"})
    assert "Worked example 1:" in prompt
    assert prompt.index("a small loan") < prompt.index("Now the task at hand:") < prompt.index("a big loan")


def test_select_demos_prefers_reviewed_examples_within_budget():
    from ear import Optimizer
    from ear.judgment import Field, Judgment
    from ear.optimizer import Example
    from ear.signatures import ReasonAboutIntent

    judgment = Judgment(
        instruction=ReasonAboutIntent.instruction,
        inputs=list(ReasonAboutIntent.inputs),
        outputs=list(ReasonAboutIntent.outputs),
    )
    examples = [
        Example(intent="unreviewed", decision="maybe"),
        Example(intent="approved one", decision="approve", verdict=True),
        Example(intent="wrong one", decision="nonsense", verdict=False),
    ]
    demos = Optimizer().select_demos(judgment, examples, budget_chars=200)
    assert demos[0]["intent"] == "approved one"  # reviewed first
    assert all(demo["intent"] != "wrong one" for demo in demos)  # never the judged-wrong
    assert judgment.demos == demos


def test_instructions_persist_and_reload_with_demos(tmp_path):
    from ear import Optimizer
    from ear.judgment import Field, Judgment

    judgment = Judgment(
        instruction="The refined instruction.",
        inputs=[Field("intent")],
        outputs=[Field("decision")],
        demos=[{"intent": "a loan", "decision": "approve"}],
    )
    optimizer = Optimizer()
    path = optimizer.save_instructions(tmp_path / ".ear" / "instructions.md", {"MyJudgment": judgment})

    fresh = Judgment(instruction="The shipped default.", inputs=[Field("intent")], outputs=[Field("decision")])
    applied = optimizer.load_instructions(path, {"MyJudgment": fresh})
    assert applied == ["MyJudgment"]
    assert fresh.instruction == "The refined instruction."
    assert fresh.demos == [{"intent": "a loan", "decision": "approve"}]


@requires_anthropic_key
def test_search_improves_or_keeps_the_baseline_live():
    from ear import ModelBinding, Optimizer
    from ear.judgment import Field, Judgment
    from ear.optimizer import Example

    binding = ModelBinding(provider="anthropic", model=os.environ.get("ANTHROPIC_TEST_MODEL", "claude-haiku-4-5"))
    binding.activate()
    # A scratch copy so the search never mutates the shared registry.
    judgment = Judgment(
        instruction="Decide approve or decline for the loan intent. Answer with one word.",
        inputs=[Field("intent", "The loan request")],
        outputs=[Field("decision", "approve or decline, one word", "str")],
    )
    examples = [
        Example(intent="A $5,000 loan, credit score 790, no debts", decision="approve", verdict=True),
        Example(intent="A $9,000 loan for a defaulted borrower with no income", decision="decline", verdict=True),
    ]
    outcome = Optimizer().search(
        judgment, examples, model_binding=binding, generations=1, candidates=2, holdout=1.0
    )
    assert outcome.best_score >= outcome.baseline_score
    assert outcome.evaluations >= 2
    assert judgment.instruction  # the winner (or the kept baseline) is in place


# ---------------------------------------------------------------------------
# Observability: the trail fans out to exporters (a native protocol --
# anything with export(record)) and every cycle carries usage accounting.
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


def test_strategy_reads_file_and_url_knowledge_sources():
    strategy = Strategy.from_markdown(KNOWLEDGE_MEMORY)
    assert strategy.knowledge_sources == [
        KnowledgeSource(name="underwriting manual", pattern="knowledge/manual.md")
    ]
    assert "reference material" in strategy.knowledge

    declared = Strategy.from_markdown(
        "## Knowledge\n\n- market brief: https://example.com/brief.md, refetch weekly\n"
    ).knowledge_sources
    assert declared == [
        KnowledgeSource(name="market brief", url="https://example.com/brief.md", refresh_days=7.0)
    ]

    with pytest.raises(ValueError, match="declares no path"):
        Strategy.from_markdown("## Knowledge\n\n- manual\n")


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


def test_librarian_retriever_seam_accepts_any_passage_source():
    from ear import Passage

    class CustomRetriever:
        def retrieve(self, query):
            return [Passage(source="manual.md", text="Section one text.")]

    runtime = Runtime(name="Retriever-Runtime")
    runtime.librarian.retriever = CustomRetriever()
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


# ---------------------------------------------------------------------------
# N2: evaluation & knowledge depth -- BM25 narrowing, the persisted gist
# index, URL knowledge sources, report history with regression diffs,
# rubric criteria, and A/B stack comparison.
# ---------------------------------------------------------------------------


def test_bm25_ranks_rare_terms_above_repeated_common_ones():
    from ear import Knowledge, Passage

    knowledge = Knowledge(
        passages=[
            Passage(
                source="manual § plants",
                text="Loan loan loan loan loan loan loan paperwork is filed on Fridays with the loan clerk.",
            ),
            Passage(source="manual § subordination", text="A subordination agreement reprioritizes the loan lien."),
        ]
    )
    # "subordination" is rare (idf-heavy); the first passage's seven
    # repetitions of the common word "loan" saturate instead of winning.
    ranked = knowledge.candidates("subordination of a loan", limit=2)
    assert ranked[0].source == "manual § subordination"


def test_gist_bridges_the_synonym_phrasing_word_matching_misses():
    from ear import Knowledge, Passage

    debt = Passage(
        source="manual § 4.2",
        text="A grade C applicant whose debt-to-income band is not low must be declined.",
    )
    plants = Passage(source="manual § 9.9", text="Water the office plants on Fridays.")
    knowledge = Knowledge(passages=[plants, debt])
    query = "heavy existing borrowings"

    # Without the gist index the synonym phrasing shares no word with any
    # passage: no signal, so the corpus-order fallback surfaces plants.
    assert knowledge.narrowing() == "BM25 over passage text alone (no gist index)"
    assert knowledge.candidates(query, limit=1)[0].source == "manual § 9.9"

    debt.gist = "whether an applicant with heavy existing borrowings or high debt can be approved for a loan"
    assert knowledge.narrowing() == "BM25 over passage text and index gists"
    assert knowledge.candidates(query, limit=1)[0].source == "manual § 4.2"


def test_gist_index_persists_by_content_hash_and_retires_on_edit(tmp_path):
    from ear import Knowledge

    document = "# M\n\n## Ratios\n\nDebt banding governs declines.\n\n## Plants\n\nWater plants on Fridays.\n"
    knowledge = Knowledge().add_document("manual", "m.md", document)
    lm = ScriptedLM(
        ["## gist\n\nhow heavy borrowings force a decline\n", "## gist\n\ncaring for office greenery\n"]
    )
    assert knowledge.build_gists(lm) == 2

    index_path = tmp_path / ".ear" / "index.md"
    knowledge.write_index(index_path, model_label="anthropic/test")
    index_text = index_path.read_text(encoding="utf-8")
    assert "anthropic/test" in index_text and "office greenery" in index_text

    # A fresh load of the same corpus reuses every gist without a model.
    reloaded = Knowledge().add_document("manual", "m.md", document)
    assert reloaded.load_index(index_path) == 2
    assert not reloaded.missing_gists()

    # Editing one section retires only that entry: the edited passage
    # hashes differently, misses the index, and needs re-gisting.
    edited = document.replace("Water plants", "Water plants twice")
    partially = Knowledge().add_document("manual", "m.md", edited)
    assert partially.load_index(index_path) == 1
    assert [passage.source for passage in partially.missing_gists()] == ["manual -- m.md § Plants"]


def test_url_knowledge_source_fetches_once_and_reuses_the_cache(tmp_path, monkeypatch):
    fetches = []

    def fake_fetch(url, timeout=60):
        fetches.append(url)
        return "# Brief\n\n## Outlook\n\nRates are expected to hold.\n"

    monkeypatch.setattr("ear.loader.fetch_text", fake_fetch)
    stack = dict(MINIMAL_STACK)
    stack["memory"] = "## Knowledge\n\n- market brief: https://example.com/brief.md\n"
    directory = write_stack(tmp_path / "stack", **stack)

    runtime = load_runtime(directory)
    assert fetches == ["https://example.com/brief.md"]
    assert (directory / ".ear" / "knowledge" / "market-brief.md").exists()
    assert any("Rates are expected to hold" in p.text for p in runtime.librarian.knowledge.passages)

    # No refresh cadence declared: the cache stands indefinitely.
    load_runtime(directory)
    assert len(fetches) == 1


def test_url_knowledge_source_honors_the_declared_refresh_cadence(tmp_path, monkeypatch):
    from ear.llm import LMError

    fetches = []

    def fake_fetch(url, timeout=60):
        fetches.append(url)
        if len(fetches) == 3:
            raise LMError("network down")
        return "# Brief\n\nRates hold.\n"

    monkeypatch.setattr("ear.loader.fetch_text", fake_fetch)
    stack = dict(MINIMAL_STACK)
    stack["memory"] = "## Knowledge\n\n- market brief: https://example.com/brief.md, refetch weekly\n"
    directory = write_stack(tmp_path / "stack", **stack)

    load_runtime(directory)
    load_runtime(directory)  # fresh cache -> no refetch
    assert len(fetches) == 1

    cached = directory / ".ear" / "knowledge" / "market-brief.md"
    eight_days_ago = time.time() - 8 * 86400
    os.utime(cached, (eight_days_ago, eight_days_ago))
    load_runtime(directory)  # stale by the declared week -> refetch
    assert len(fetches) == 2

    os.utime(cached, (eight_days_ago, eight_days_ago))
    runtime = load_runtime(directory)  # refetch fails -> the stale cache stands
    assert len(fetches) == 3
    assert any("Rates hold" in p.text for p in runtime.librarian.knowledge.passages)


def test_url_knowledge_source_with_no_cache_fails_loudly(tmp_path, monkeypatch):
    from ear.llm import LMError

    def failing_fetch(url, timeout=60):
        raise LMError("unreachable")

    monkeypatch.setattr("ear.loader.fetch_text", failing_fetch)
    stack = dict(MINIMAL_STACK)
    stack["memory"] = "## Knowledge\n\n- market brief: https://example.com/brief.md\n"
    with pytest.raises(ValueError, match="no cached copy"):
        load_runtime(write_stack(tmp_path / "stack", **stack))


def test_retrieval_record_names_its_narrowing_basis(tmp_path):
    directory = knowledge_stack(tmp_path)
    runtime = load_runtime(directory)
    runtime.reason(Intent(text="Underwrite a marginal grade C applicant", context={"loan_amount": 9000}))
    retrieval = runtime.reasoning_log.for_stage("retrieval")[0]
    assert retrieval.inputs["narrowing"] == "BM25 over passage text alone (no gist index)"


def test_reports_archive_and_diff_newly_failing_and_passing(tmp_path):
    from ear import Examiner

    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    evaluations = directory / "evaluations"
    evaluations.mkdir()
    passing = "# Underwrite a loan\n\n## Context\n\n- loan_amount: 5000\n\n## Expected\n\n- decision: resolved\n"
    failing = passing.replace("resolved", "unobtainium")
    (evaluations / "steady.md").write_text(passing, encoding="utf-8")
    (evaluations / "flaky.md").write_text(passing, encoding="utf-8")

    runtime = load_runtime(directory)
    first = Examiner().examine(runtime, evaluations)
    assert first.prior_verdicts is None  # no history yet -> nothing to diff
    assert "Changes Since Last Report" not in (evaluations / "report.md").read_text(encoding="utf-8")
    assert len(list((evaluations / "reports").glob("*.md"))) == 1

    (evaluations / "flaky.md").write_text(failing, encoding="utf-8")
    second = Examiner().examine(runtime, evaluations)
    assert second.changes() == {"newly failing": ["flaky"], "newly passing": [], "still failing": []}

    third = Examiner().examine(runtime, evaluations)
    assert third.changes()["still failing"] == ["flaky"]

    (evaluations / "flaky.md").write_text(passing, encoding="utf-8")
    fourth = Examiner().examine(runtime, evaluations)
    assert fourth.changes() == {"newly failing": [], "newly passing": ["flaky"], "still failing": []}
    report = (evaluations / "report.md").read_text(encoding="utf-8")
    assert "- newly passing: flaky" in report
    assert len(list((evaluations / "reports").glob("*.md"))) == 4


def test_rubric_criteria_are_ungraded_offline_never_faked(tmp_path):
    from ear import Examiner

    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    evaluations = directory / "evaluations"
    evaluations.mkdir()
    (evaluations / "rubric.md").write_text(
        "# Underwrite a loan\n\n## Context\n\n- loan_amount: 5000\n\n"
        "## Expected\n\n- decision: resolved\n- names the responsible persona\n- states a concrete amount\n",
        encoding="utf-8",
    )
    runtime = load_runtime(directory)
    examination = Examiner().examine(runtime, evaluations)

    result = examination.results[0]
    assert result.passed is True  # the field expectation still grades structurally
    assert [criterion.passed for criterion in result.criteria] == [None, None]
    assert all("needs a model" in criterion.rationale for criterion in result.criteria)
    report = (evaluations / "report.md").read_text(encoding="utf-8")
    assert "Rubric:" in report and "- ungraded: names the responsible persona" in report


def test_rubric_criteria_grade_separately_and_a_failed_criterion_fails(tmp_path):
    from ear import Examiner, ModelBinding

    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    runtime = load_runtime(directory)
    binding = ModelBinding(provider="anthropic", model="test")
    binding.lm = ScriptedLM()
    runtime.model_binding = binding

    graded, rationale, criteria = Examiner()._grade(
        runtime,
        expected_prose="The loan is approved.",
        expected_fields={},
        criteria=["cites the underwriting manual", "names a concrete amount"],
        outcome="Approved at $5,000 per the manual.",
    )
    # ScriptedLM without a script answers '## decision' sections, which the
    # bool field reads as unparseable -- so script the three judgments.
    binding.lm = ScriptedLM(
        [
            "## passed\n\nyes\n\n## rationale\n\nMatches the expectation.\n",
            "## passed\n\nno\n\n## rationale\n\nNo citation appears.\n",
            "## passed\n\nyes\n\n## rationale\n\n$5,000 is concrete.\n",
        ]
    )
    graded, rationale, criteria = Examiner()._grade(
        runtime,
        expected_prose="The loan is approved.",
        expected_fields={},
        criteria=["cites the underwriting manual", "names a concrete amount"],
        outcome="Approved at $5,000 per the manual.",
    )
    assert graded is False  # the failed criterion fails the evaluation
    assert [criterion.passed for criterion in criteria] == [False, True]
    assert "No citation" in criteria[0].rationale


def test_compare_refuses_without_a_model(tmp_path):
    from ear import Examiner

    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    evaluations = directory / "evaluations"
    evaluations.mkdir()
    (evaluations / "any.md").write_text("# Underwrite\n\n## Expected\n\nApproved.\n", encoding="utf-8")
    runtime_a = load_runtime(directory, name="A")
    runtime_b = load_runtime(directory, name="B")
    with pytest.raises(ValueError, match="never written down"):
        Examiner().compare(runtime_a, runtime_b, evaluations)
    assert not (evaluations / "comparison.md").exists()


def test_compare_prefers_on_the_record_with_a_scripted_judge(tmp_path):
    from ear import Examiner, ModelBinding

    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    evaluations = directory / "evaluations"
    evaluations.mkdir()
    (evaluations / "small-loan.md").write_text(
        "# Underwrite a loan\n\n## Context\n\n- loan_amount: 5000\n\n## Expected\n\nThe loan is handled.\n",
        encoding="utf-8",
    )
    runtime_a = load_runtime(directory, name="Stack-A")
    runtime_b = load_runtime(directory, name="Stack-B")
    binding = ModelBinding(provider="anthropic", model="test")
    binding.lm = ScriptedLM(
        ["## preference\n\nB\n\n## rationale\n\nB states the amount plainly.\n"]
    )

    # The referee is a dedicated judge binding, independent of the two
    # contestants -- both runtimes reason with their deterministic
    # pipeline, and the script is consumed by the preference alone.
    comparison = Examiner().compare(runtime_a, runtime_b, evaluations, judge=binding)
    assert comparison.counts() == {"A": 0, "B": 1, "tie": 0}
    assert comparison.results[0].preference == "B"

    rendered = (evaluations / "comparison.md").read_text(encoding="utf-8")
    assert "A: Stack-A vs B: Stack-B" in rendered and "Preferred B: 1" in rendered
    record = runtime_a.reasoning_log.for_stage("comparison")[0]
    assert record.output == "B" and "amount plainly" in record.rationale


@requires_anthropic_key
def test_gist_index_builds_once_and_reloads_live(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["memory"] = (
        KNOWLEDGE_MEMORY
        + "\n## Model Selection\n\nReason with anthropic/claude-haiku-4-5. "
        "The key lives in ANTHROPIC_API_KEY.\n"
    )
    directory = write_stack(tmp_path / "stack", **stack)
    (directory / "knowledge").mkdir()
    (directory / "knowledge" / "manual.md").write_text(MANUAL, encoding="utf-8")

    runtime = load_runtime(directory)
    index_path = directory / ".ear" / "index.md"
    assert index_path.exists()
    assert index_path.read_text(encoding="utf-8").count("## Passage") == 2
    indexing = runtime.reasoning_log.for_stage("indexing")[0]
    assert indexing.inputs["gists_written"] == 2
    assert indexing.input_tokens and indexing.output_tokens  # indexing is on the record, billed

    # The corpus indexes once: a fresh load reuses every gist by content
    # hash and asks the model for nothing.
    reloaded = load_runtime(directory)
    assert not reloaded.reasoning_log.for_stage("indexing")
    assert not reloaded.librarian.knowledge.missing_gists()
    assert reloaded.librarian.knowledge.narrowing() == "BM25 over passage text and index gists"


def test_model_id_at_sentence_end_keeps_no_period():
    strategy = Strategy.from_markdown(
        "## Model Selection\n\nReason with anthropic/claude-haiku-4-5. The key lives in ANTHROPIC_API_KEY.\n"
    )
    assert strategy.model == "anthropic/claude-haiku-4-5"
    assert strategy.provider == "anthropic"


# ---------------------------------------------------------------------------
# N3: execution depth -- prose-authored routing, leg retry policies, the
# journey runner with escalation, event documents, and dynamic panels.
# ---------------------------------------------------------------------------

ROUTED_WORKFLOW = (
    "# Workflows\n\n## Underwriting Workflow\n\n"
    "Routes: if the risk grade is C or worse, skip straight to the decline note; "
    "otherwise continue in order.\n\n"
    "1. Band the profile and assign a risk grade. (Credit Risk Guru)\n"
    "2. Prepare the approval paperwork. (Credit Risk Guru)\n"
    "3. Write the decline note. (Credit Risk Guru)\n\n"
    "Policies: Loan Amount Cap\n"
)


def test_routes_retries_and_escalation_are_authored_in_prose(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["workflow"] = ROUTED_WORKFLOW.replace(
        "Routes:", "Retries: retry a failed leg twice before giving up.\nRoutes:"
    )
    stack["policy"] = MINIMAL_STACK["policy"].replace(
        "Fallback: loan_amount <= 75000\nApplies to: Underwriting Workflow\n",
        "Fallback: loan_amount <= 75000\nApplies to: Underwriting Workflow\n"
        "Approval: required\nEscalate: after 3 days\n",
    )
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    workflow = runtime.processes[0].workflows[0]
    assert "skip straight to the decline note" in workflow.routes
    assert workflow.retry_budget == 2
    cap = workflow.policies[0]
    assert cap.escalation == "after 3 days" and cap.escalation_days == 3.0

    bad = dict(MINIMAL_STACK)
    bad["workflow"] = MINIMAL_STACK["workflow"].replace(
        "Policies:", "Retries: whenever it feels right\nPolicies:"
    )
    with pytest.raises(ValueError, match="no readable count"):
        load_runtime(write_stack(tmp_path / "bad-retries", **bad))

    bad_policy = dict(MINIMAL_STACK)
    bad_policy["policy"] = MINIMAL_STACK["policy"].replace(
        "Applies to: runtime\n", "Applies to: runtime\nEscalate: someday, surely\n"
    )
    with pytest.raises(ValueError, match="no readable period"):
        load_runtime(write_stack(tmp_path / "bad-escalate", **bad_policy))


def test_strategy_reads_the_leg_retry_budget_from_prose():
    strategy = Strategy.from_markdown(
        "## Execution Resilience\n\nRetry a failed leg twice before giving up.\n"
    )
    assert strategy.leg_retry_budget == 2
    assert "Retry a failed leg" in strategy.execution
    assert Strategy.from_markdown("## Execution\n\nNo retries; fail fast.\n").leg_retry_budget == 0


def test_routed_journey_offline_continues_in_order_and_says_so(tmp_path):
    from ear import Journey

    stack = dict(MINIMAL_STACK)
    stack["workflow"] = ROUTED_WORKFLOW
    directory = write_stack(tmp_path / "stack", **stack)
    runtime = load_runtime(directory)
    journey = Journey(directory / "journeys" / "loan.md")
    status = journey.run(runtime, Intent(text="Underwrite a loan", context={"loan_amount": 5000}))

    assert status == "completed"
    assert [leg.step for leg in journey.legs] == [1, 2, 3]
    routings = runtime.reasoning_log.for_stage("routing")
    assert len(routings) == 3
    assert all("no model bound" in record.rationale for record in routings)
    assert routings[-1].output == "conclude the journey"


def test_routing_judgment_never_invents_a_step_and_honors_the_revisit_budget(tmp_path):
    from ear import Journey, ModelBinding, Workflow

    binding = ModelBinding(provider="anthropic", model="test")
    runtime = Runtime(name="Routed-Runtime", model_binding=binding)
    workflow = Workflow(name="W", routes="loop until done")
    workflow.add_step("first")
    workflow.add_step("second")
    authored = [(workflow, step) for step in workflow.steps]
    journey = Journey(tmp_path / "j.md", intent_text="x")

    from ear.journey import Leg

    leg = Leg(number=1, workflow="W", instruction="first", decision="d", status="decided", step=1)

    # A step number the stack never authored is refused, never improvised.
    binding.lm = ScriptedLM(["## next step\n\n9\n\n## rationale\n\nJump far.\n"])
    assert journey._route(runtime, workflow, authored, leg, {1: 1}) == 2
    assert "names no authored step" in runtime.reasoning_log.for_stage("routing")[-1].rationale

    # A legal loop is honored -- until the revisit budget refuses it.
    binding.lm = ScriptedLM(["## next step\n\n1\n\n## rationale\n\nDo it again.\n"])
    assert journey._route(runtime, workflow, authored, leg, {1: 1}) == 1
    binding.lm = ScriptedLM(["## next step\n\n1\n\n## rationale\n\nAgain, forever.\n"])
    assert journey._route(runtime, workflow, authored, leg, {1: 3}) == 2
    assert "revisit budget" in runtime.reasoning_log.for_stage("routing")[-1].rationale

    # 'conclude' ends the journey; 'next' continues in order.
    binding.lm = ScriptedLM(["## next step\n\nconclude\n\n## rationale\n\nNothing left.\n"])
    assert journey._route(runtime, workflow, authored, leg, {1: 1}) is None
    binding.lm = ScriptedLM(["## next step\n\nnext\n\n## rationale\n\nIn order.\n"])
    assert journey._route(runtime, workflow, authored, leg, {1: 1}) == 2


def test_journey_retries_a_failing_leg_within_the_declared_budget(tmp_path):
    from ear import Journey

    stack = dict(MINIMAL_STACK)
    stack["workflow"] = MINIMAL_STACK["workflow"].replace(
        "Policies:", "Retries: retry a failed leg twice before giving up.\nPolicies:"
    )
    directory = write_stack(tmp_path / "stack", **stack)
    runtime = load_runtime(directory)
    attempts = {"count": 0}
    true_reason = runtime.reason

    def flaky_reason(intent, approval=None):
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise RuntimeError("transient worker failure")
        return true_reason(intent, approval=approval)

    runtime.reason = flaky_reason
    journey = Journey(directory / "journeys" / "loan.md")
    status = journey.run(runtime, Intent(text="Underwrite a loan", context={"loan_amount": 5000}))

    assert status == "completed"
    retries = runtime.reasoning_log.for_stage("retry")
    assert len(retries) == 2
    assert all("retrying" in record.output for record in retries)
    assert retries[0].inputs["error"] == "transient worker failure"


def test_journey_fails_on_the_record_when_the_retry_budget_is_exhausted(tmp_path):
    from ear import Journey

    stack = dict(MINIMAL_STACK)
    stack["workflow"] = MINIMAL_STACK["workflow"].replace(
        "Policies:", "Retries: retry once.\nPolicies:"
    )
    directory = write_stack(tmp_path / "stack", **stack)
    runtime = load_runtime(directory)

    def doomed_reason(intent, approval=None):
        raise RuntimeError("the worker never comes back")

    runtime.reason = doomed_reason
    journey = Journey(directory / "journeys" / "loan.md")
    status = journey.run(runtime, Intent(text="Underwrite a loan", context={"loan_amount": 5000}))

    assert status == "FAILED"
    assert journey.legs[-1].status == "FAILED"
    assert "retry budget exhausted" in journey.legs[-1].decision
    record = (directory / "journeys" / "loan.md").read_text(encoding="utf-8")
    assert "Status: FAILED" in record
    assert "exhausted" in runtime.reasoning_log.for_stage("retry")[-1].output
    # Settled: another run replays nothing.
    assert Journey(journey.path).run(runtime) == "FAILED"


def test_journey_without_a_declared_budget_keeps_crash_semantics(tmp_path):
    from ear import Journey

    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    runtime = load_runtime(directory)

    def crashing_reason(intent, approval=None):
        raise RuntimeError("worker died")

    runtime.reason = crashing_reason
    with pytest.raises(RuntimeError, match="worker died"):
        Journey(directory / "journeys" / "loan.md").run(
            runtime, Intent(text="Underwrite", context={"loan_amount": 5000})
        )
    assert runtime.reasoning_log.for_stage("retry") == []


def test_journey_consumes_event_documents_exactly_once(tmp_path):
    from ear import Journey

    directory = write_stack(tmp_path / "stack", **MINIMAL_STACK)
    runtime = load_runtime(directory)
    journeys = directory / "journeys"

    # Crash after the first leg so the journey waits mid-walk.
    calls = {"count": 0}
    true_reason = runtime.reason

    def crash_on_second(intent, approval=None):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("worker died mid-journey")
        return true_reason(intent, approval=approval)

    runtime.reason = crash_on_second
    with pytest.raises(RuntimeError):
        Journey(journeys / "loan.md").run(runtime, Intent(text="Underwrite", context={"loan_amount": 5000}))

    (journeys / "events").mkdir()
    (journeys / "events" / "loan-appraisal.md").write_text(
        "# Event -- appraisal received\n\n## Context\n\n- appraisal_value: 250000\n",
        encoding="utf-8",
    )
    runtime.reason = true_reason
    resumed = Journey(journeys / "loan.md")
    assert resumed.run(runtime) == "completed"
    assert resumed.context["appraisal_value"] == 250000
    event = runtime.reasoning_log.for_stage("event")[0]
    assert event.inputs["event"] == "loan-appraisal.md"
    record = (journeys / "loan.md").read_text(encoding="utf-8")
    assert "## Events" in record and "loan-appraisal.md" in record
    assert "appraisal_value: 250000" in record

    # A fresh run consumes nothing twice.
    again = Journey(journeys / "loan.md")
    again.run(runtime)
    assert len(runtime.reasoning_log.for_stage("event")) == 1


def gated_stack(tmp_path, escalate=""):
    # One step per workflow: an approval waives the gate for one governed
    # cycle, so the gated walk is a single leg -- the same shape as
    # approval_stack above.
    stack = dict(MINIMAL_STACK)
    stack["workflow"] = (
        "# Workflows\n\n## Underwriting Workflow\n\n"
        "1. Decide approve or decline. (Credit Risk Guru)\n"
    )
    stack["policy"] = (
        "# Policies\n\n## Loan Amount Cap\nThe loan must not exceed $75,000.\n\n"
        "Fallback: loan_amount <= 75000\nApplies to: Underwriting Workflow\n\n"
        "## Large Loan Approval\nLoans over $50,000 need a human.\n\n"
        "Fallback: loan_amount <= 50000\nApplies to: Underwriting Workflow\n"
        "Approval: required\n" + (f"Escalate: {escalate}\n" if escalate else "")
    )
    return write_stack(tmp_path / "stack", **stack)


def test_runner_resumes_releases_and_escalates_in_one_pass(tmp_path):
    from ear import Journey, Journeys

    directory = gated_stack(tmp_path, escalate="after 3 days")
    runtime = load_runtime(directory)
    journeys = directory / "journeys"

    # a: crashes on its first leg -- the record already exists, in progress.
    def crash(intent, approval=None):
        raise RuntimeError("died")

    true_reason = runtime.reason
    runtime.reason = crash
    with pytest.raises(RuntimeError):
        Journey(journeys / "a.md").run(runtime, Intent(text="Underwrite", context={"loan_amount": 5000}))
    runtime.reason = true_reason

    # b: parks on the approval gate; a human verdict awaits in approvals/.
    assert (
        Journey(journeys / "b.md").run(runtime, Intent(text="Underwrite big", context={"loan_amount": 60000}))
        == "PENDING APPROVAL"
    )
    (journeys / "approvals").mkdir()
    (journeys / "approvals" / "b.md").write_text(
        "# Approval\n\nVerdict: approved\nApprover: lakshmi@gigkri.com\n\n> Collateral covers it.\n",
        encoding="utf-8",
    )

    # c: parks on the same gate with no approval -- and a declared deadline.
    assert (
        Journey(journeys / "c.md").run(runtime, Intent(text="Underwrite big", context={"loan_amount": 61000}))
        == "PENDING APPROVAL"
    )

    four_days = 4 * 86400
    outcomes = Journeys().run_all(runtime, journeys, now=time.time() + four_days)
    assert outcomes["a.md"] == "completed"
    assert outcomes["b.md"] == "completed"
    assert outcomes["c.md"] == "ESCALATED"

    c_record = (journeys / "c.md").read_text(encoding="utf-8")
    assert "Status: ESCALATED" in c_record and "## Escalation" in c_record
    assert "deadline has passed" in c_record
    escalation = runtime.reasoning_log.for_stage("escalation")[0]
    assert escalation.inputs["journey"] == "c.md"

    # A second pass is idempotent: settled stays settled, escalated stays
    # escalated with no duplicate record, and an approval still releases c.
    again = Journeys().run_all(runtime, journeys, now=time.time() + four_days)
    assert again == {"a.md": "completed", "b.md": "completed", "c.md": "ESCALATED"}
    assert len(runtime.reasoning_log.for_stage("escalation")) == 1

    (journeys / "approvals" / "c.md").write_text(
        "# Approval\n\nVerdict: approved\nApprover: lakshmi@gigkri.com\n", encoding="utf-8"
    )
    released = Journeys().run_all(runtime, journeys, now=time.time() + four_days)
    assert released["c.md"] == "completed"


def test_runner_refuses_an_unreadable_approval_on_the_record(tmp_path):
    from ear import Journey, Journeys

    directory = gated_stack(tmp_path)
    runtime = load_runtime(directory)
    journeys = directory / "journeys"
    Journey(journeys / "big.md").run(runtime, Intent(text="Underwrite big", context={"loan_amount": 60000}))
    (journeys / "approvals").mkdir()
    (journeys / "approvals" / "big.md").write_text("# Approval\n\nVerdict: perhaps\n", encoding="utf-8")

    outcomes = Journeys().run_all(runtime, journeys)
    assert outcomes["big.md"] == "PENDING APPROVAL"
    refusal = runtime.reasoning_log.for_stage("approval")[-1]
    assert "unreadable verdict" in refusal.output


def test_panel_speaker_is_judged_and_concludes_early_after_all_have_spoken():
    from ear import ModelBinding, Panel

    binding = ModelBinding(provider="anthropic", model="test")
    binding.lm = ScriptedLM(
        [
            # Turn 1: the model tries to conclude before anyone spoke -- refused.
            "## speaker\n\nconclude\n\n## rationale\n\nSeems obvious.\n",
            "## statement\n\nGrade C; decline.\n",
            # Turn 2: the model picks the advocate by name.
            "## speaker\n\nCustomer Advocate\n\n## rationale\n\nThe other side must answer.\n",
            "## statement\n\nAgreed -- decline, with a referral.\n",
            # Turn 3: both have spoken; consensus concludes the panel early.
            "## speaker\n\nconclude\n\n## rationale\n\nBoth agree on decline.\n",
            "## decision\n\nDecline, with a referral to the secured product.\n",
        ]
    )
    runtime = Runtime(name="Panel-Runtime", model_binding=binding)
    personas = [Persona(name="Credit Risk Guru"), Persona(name="Customer Advocate")]
    decision = Panel(rounds=3).convene(runtime, personas, Intent(text="Underwrite a marginal loan"))

    turns = runtime.reasoning_log.for_stage("conversation")
    assert len(turns) == 2  # concluded early, well under the budget of 6
    assert "conclusion refused" in turns[0].inputs["chosen_by"]
    assert turns[1].inputs["chosen_by"] == "model"
    assert turns[1].inputs["choice_rationale"] == "The other side must answer."
    deliberation = runtime.reasoning_log.for_stage("deliberation")[0]
    assert deliberation.inputs["concluded_early"] == "Both agree on decline."
    assert decision == "Decline, with a referral to the secured product."


def test_panel_rotation_stands_when_the_choice_names_nobody():
    from ear import ModelBinding, Panel

    binding = ModelBinding(provider="anthropic", model="test")
    binding.lm = ScriptedLM(
        [
            "## speaker\n\nThe Moderator\n\n## rationale\n\nA phantom.\n",
            "## statement\n\nFirst word.\n",
            "## speaker\n\nB\n\n## rationale\n\nB's turn.\n",
            "## statement\n\nSecond word.\n",
            "## decision\n\nSettled.\n",
        ]
    )
    runtime = Runtime(name="Panel-Runtime", model_binding=binding)
    personas = [Persona(name="A"), Persona(name="B")]
    Panel(rounds=1).convene(runtime, personas, Intent(text="anything"))
    turns = runtime.reasoning_log.for_stage("conversation")
    assert [record.inputs["speaker"] for record in turns] == ["A", "B"]
    assert "names no listed persona" in turns[0].inputs["chosen_by"]


def test_panel_persona_uses_tools_inside_a_turn_on_the_record(tmp_path):
    from ear import ModelBinding, Panel

    stack = dict(MINIMAL_STACK)
    stack["memory"] = "## Tools\n\n- credit lookup: fetch the applicant's bureau score\n"
    directory = write_stack(tmp_path / "stack", **stack)
    runtime = load_runtime(directory)
    runtime.tool_binder.bind("credit lookup", lambda applicant_id: f"score for {applicant_id}: 668")

    binding = ModelBinding(provider="anthropic", model="test")
    binding.lm = ScriptedLM(
        [
            "## speaker\n\nAnalyst\n\n## rationale\n\nFacts first.\n",
            # The turn calls the tool, then speaks with the fact in hand.
            "## tool\n\ncredit lookup\n\n## arguments\n\n- applicant_id: 42\n\n## statement\n\n\n",
            "## tool\n\n\n\n## arguments\n\n\n\n## statement\n\nThe bureau score is 668; marginal.\n",
            "## decision\n\nDecline at 668.\n",
        ]
    )
    runtime.model_binding = binding
    personas = [Persona(name="Analyst")]
    decision = Panel(rounds=1).convene(runtime, personas, Intent(text="Underwrite"))

    tool_record = runtime.reasoning_log.for_stage("tool")[0]
    assert tool_record.inputs["tool"] == "credit lookup"
    assert "668" in tool_record.output
    turn = runtime.reasoning_log.for_stage("conversation")[0]
    assert turn.output == "The bureau score is 668; marginal."
    assert decision == "Decline at 668."


@requires_anthropic_key
def test_journey_routes_the_authored_skip_live(tmp_path):
    from ear import Journey

    stack = dict(MINIMAL_STACK)
    stack["workflow"] = ROUTED_WORKFLOW
    stack["memory"] = LIVE_MEMORY
    directory = write_stack(tmp_path / "stack", **stack)
    runtime = load_runtime(directory)
    journey = Journey(directory / "journeys" / "marginal.md")
    status = journey.run(
        runtime,
        Intent(
            text="Underwrite a $9,000 loan for a marginal applicant",
            context={"loan_amount": 9000, "credit_score": 580, "debt_to_income": 0.42},
        ),
    )

    assert status == "completed"
    steps_walked = [leg.step for leg in journey.legs]
    assert steps_walked[0] == 1
    assert 2 not in steps_walked  # the approval paperwork was skipped
    assert 3 in steps_walked
    routing = runtime.reasoning_log.for_stage("routing")[0]
    assert routing.model == "anthropic/claude-haiku-4-5"


@requires_anthropic_key
def test_panel_concludes_early_on_consensus_live(tmp_path):
    stack = dict(PANEL_STACK)
    stack["memory"] = LIVE_MEMORY
    stack["workflow"] = PANEL_STACK["workflow"].replace(
        "Pattern: adversarial debate; the Credit Risk Guru has the last word",
        "Pattern: one short turn each, then conclude immediately once both have spoken -- "
        "do not keep talking past agreement",
    )
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    runtime.panel.rounds = 3  # budget of six turns the consensus should beat
    decision = runtime.reason(
        Intent(
            text="Underwrite a $2,000 loan for a prime applicant with excellent credit",
            context={"loan_amount": 2000, "credit_score": 810, "debt_to_income": 0.05},
        )
    )
    turns = runtime.reasoning_log.for_stage("conversation")
    assert 2 <= len(turns) < 6  # concluded before the budget
    assert decision and "no model bound" not in decision


# ---------------------------------------------------------------------------
# N4: governance & connectivity depth -- approver allow-lists, tool-scoped
# policies, the hash-chained tamper-evident trail, retention + the usage
# ledger, and the native MCP client.
# ---------------------------------------------------------------------------


def test_policy_reads_approvers_and_gates_by_the_allow_list(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["workflow"] = "# Workflows\n\n## Underwriting Workflow\n\n1. Decide. (Credit Risk Guru)\n"
    stack["policy"] = (
        "# Policies\n\n## Loan Amount Cap\nThe loan must not exceed $75,000.\n\n"
        "Fallback: loan_amount <= 75000\nApplies to: runtime\n\n"
        "## Large Loan Approval\nLoans over $50,000 need a senior underwriter.\n\n"
        "Fallback: loan_amount <= 50000\nApplies to: runtime\n"
        "Approval: required\nApprovers: senior@bank.com, Chief Risk Officer\n"
    )
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    policy = {p.name: p for p in runtime.policies}["Large Loan Approval"]
    assert policy.approvers == ["senior@bank.com", "Chief Risk Officer"]
    assert policy.approver_allowed("SENIOR@bank.com") is True
    assert policy.approver_allowed("chief-risk-officer") is True
    assert policy.approver_allowed("intern@bank.com") is False


def test_off_list_approver_is_refused_and_the_gate_stays_parked(tmp_path):
    from ear import Approval, Journey, Journeys

    stack = dict(MINIMAL_STACK)
    stack["workflow"] = "# Workflows\n\n## Underwriting Workflow\n\n1. Decide. (Credit Risk Guru)\n"
    stack["policy"] = (
        "# Policies\n\n## Loan Amount Cap\nThe loan must not exceed $75,000.\n\n"
        "Fallback: loan_amount <= 75000\nApplies to: Underwriting Workflow\n\n"
        "## Large Loan Approval\nLoans over $50,000 need a senior underwriter.\n\n"
        "Fallback: loan_amount <= 50000\nApplies to: Underwriting Workflow\n"
        "Approval: required\nApprovers: senior@bank.com\n"
    )
    directory = write_stack(tmp_path / "stack", **stack)
    runtime = load_runtime(directory)
    journeys = directory / "journeys"

    parked = Journey(journeys / "big.md")
    assert parked.run(runtime, Intent(text="Underwrite big", context={"loan_amount": 60000})) == "PENDING APPROVAL"

    # An off-list approver waives nothing: the gate stays parked, refused
    # on the record.
    off_list = Approval(verdict=True, approver="intern@bank.com")
    assert Journey(journeys / "big.md").run(runtime, approval=off_list) == "PENDING APPROVAL"
    refusal = [r for r in runtime.reasoning_log.for_stage("approval") if r.output.startswith("REFUSED")]
    assert refusal and "intern@bank.com" in refusal[-1].output

    # The listed approver releases it.
    on_list = Approval(verdict=True, approver="senior@bank.com")
    assert Journey(journeys / "big.md").run(runtime, approval=on_list) == "completed"


def test_tool_scoped_policy_blocks_one_call_and_returns_it_to_the_model(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["memory"] = "## Tools\n\n- wire transfer: move money out of the ledger\n"
    stack["policy"] = MINIMAL_STACK["policy"] + (
        "\n## Transfer Cap\nThe wire transfer tool must never move more than $10,000.\n\n"
        "Fallback: amount <= 10000\nApplies to: tools\n"
    )
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    assert [p.name for p in runtime.tool_policies] == ["Transfer Cap"]
    runtime.tool_binder.bind("wire transfer", lambda amount: f"moved {amount}")

    from ear.tool_binder import BoundTool

    tool = BoundTool(name="wire transfer", description="move money", handler=lambda amount: f"moved {amount}")
    invoke = runtime.tool_binder.logged_handler(runtime, tool)

    # Over the cap: blocked, and the refusal comes back as text (not a raise).
    blocked = invoke(amount=50000)
    assert "blocked by policy 'Transfer Cap'" in blocked
    record = runtime.reasoning_log.for_stage("tool")[-1]
    assert record.output.startswith("BLOCKED") and record.inputs["policy"] == "Transfer Cap"

    # Within the cap: it runs.
    assert invoke(amount=5000) == "moved 5000"


def test_the_trail_is_hash_chained_and_verify_catches_a_hand_edit(tmp_path):
    from ear.reasoning_log import ReasoningLog

    for extension in (".md", ".jsonl"):
        path = tmp_path / f"trail{extension}"
        runtime = Runtime(name="Chained-Runtime")
        runtime.reasoning_log.path = str(path)
        runtime.reason(Intent(text="first cycle", context={"amount": 1}))
        runtime.reason(Intent(text="second cycle", context={"amount": 2}))

        ok, message = ReasoningLog.verify(str(path))
        assert ok is True and "intact" in message

        # Hand-edit one record's content; the chain breaks at that record.
        text = path.read_text(encoding="utf-8")
        tampered = text.replace("second cycle", "smuggled cycle", 1)
        assert tampered != text
        path.write_text(tampered, encoding="utf-8")
        broken, why = ReasoningLog.verify(str(path))
        assert broken is False and "broken chain" in why


def test_the_chain_continues_across_a_resumed_session(tmp_path):
    from ear.reasoning_log import ReasoningLog

    path = tmp_path / "trail.md"
    first = Runtime(name="Session-One")
    first.reasoning_log.path = str(path)
    first.reasoning_log.resume()
    first.reason(Intent(text="cycle in session one"))

    second = Runtime(name="Session-Two")
    second.reasoning_log.path = str(path)
    second.reasoning_log.resume()  # picks up the chain tip and cycle number
    second.reason(Intent(text="cycle in session two"))

    ok, message = ReasoningLog.verify(str(path))
    assert ok is True, message
    assert "## Cycle 2" in path.read_text(encoding="utf-8")  # numbering continued


def test_retention_rotates_old_cycles_with_a_note_and_stays_verifiable(tmp_path):
    from datetime import datetime, timedelta, timezone

    from ear.reasoning_log import ReasoningLog

    path = tmp_path / "trail.md"
    runtime = Runtime(name="Retained-Runtime")
    runtime.reasoning_log.path = str(path)
    runtime.reason(Intent(text="ancient cycle"))
    for record in runtime.reasoning_log.records:
        record.timestamp = datetime.now(timezone.utc) - timedelta(days=100)
    runtime.reason(Intent(text="fresh cycle"))

    rotated = runtime.reasoning_log.rotate(90.0)
    assert rotated > 0
    stages = [r.stage for r in runtime.reasoning_log.records]
    assert stages[0] == "retention"  # the note stands in for what was rotated
    assert not runtime.reasoning_log.for_cycle(1)  # the ancient cycle is gone
    note = runtime.reasoning_log.records[0]
    assert "rotated" in note.output and note.inputs["rotated_records"] == rotated

    ok, message = ReasoningLog.verify(str(path))  # the rewrite re-chained cleanly
    assert ok is True, message


def test_usage_ledger_renders_from_the_trail_with_dollars_when_priced():
    from ear import ModelBinding

    binding = ModelBinding(provider="anthropic", model="test")
    binding.lm = ScriptedLM(["## complies\n\nyes\n\n## rationale\n\nfine\n"])
    runtime = Runtime(name="Ledger-Runtime", model_binding=binding)
    runtime.strategy = Strategy.from_markdown(
        "## Pricing\n\nInput tokens cost $3 per million; output tokens cost $15 per million.\n"
    )
    from ear import Policy

    runtime.add_policy(Policy(name="Cap", statement="Stay under the cap."))
    runtime.reason(Intent(text="a priced cycle", context={"amount": 1}))

    report = runtime.reasoning_log.usage_report(strategy=runtime.strategy)
    assert "# Usage Report" in report
    assert "| Cycle |" in report and "**total**" in report
    assert "$0." in report  # a dollar figure, because Pricing was declared

    offline = Runtime(name="Unpriced").reasoning_log.usage_report()
    assert "—" in offline or "total" in offline  # no pricing -> no invented dollars


# -- the native MCP client ---------------------------------------------------

FAKE_MCP_SERVER = str(Path(__file__).resolve().parent / "fixtures" / "fake_mcp_server.py")


def mcp_command() -> str:
    import sys

    return f"{sys.executable} {FAKE_MCP_SERVER}"


def test_native_mcp_client_handshakes_lists_and_calls_over_stdio():
    from ear.mcp_client import McpClient, McpError, command_words

    with McpClient(command_words(mcp_command())) as client:
        tools = client.list_tools()
        assert [(tool.name, tool.parameters) for tool in tools] == [("add", ["a", "b"])]
        assert client.call_tool("add", {"a": 3, "b": 4}) == "sum is 7"
        with pytest.raises(McpError, match="error"):
            client.call_tool("nonexistent", {})


def test_mcp_client_fails_loudly_on_a_bad_command():
    from ear.mcp_client import McpClient, McpError

    with pytest.raises(McpError, match="could not launch"):
        McpClient(["definitely-not-a-real-binary-xyz"]).connect()


def test_declared_mcp_server_binds_tools_into_a_cycle_on_the_record(tmp_path):
    stack = dict(MINIMAL_STACK)
    stack["memory"] = f"## MCP\n\n- calc: arithmetic over stdio `{mcp_command()}`\n"
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    assert [server.name for server in runtime.strategy.mcp_servers] == ["calc"]

    bound = runtime.connect_mcp()
    try:
        assert [(tool.name, tool.parameters) for tool in bound] == [("calc: add", ["a", "b"])]
        # The bound MCP tool joins the cycle's toolset and runs through the
        # same logged handler as any native tool.
        tool = runtime.tool_binder.bound_tools(runtime)[0]
        assert tool.name == "calc: add"
        result = runtime.tool_binder.logged_handler(runtime, tool)(a=8, b=9)
        assert result == "sum is 17"
        record = runtime.reasoning_log.for_stage("tool")[-1]
        assert record.inputs["tool"] == "calc: add" and record.output == "sum is 17"
    finally:
        runtime.disconnect_mcp()
    assert runtime.tool_binder.mcp_tools == []


def test_tool_scoped_policy_governs_an_mcp_call(tmp_path):
    from ear import Policy

    stack = dict(MINIMAL_STACK)
    stack["memory"] = f"## MCP\n\n- calc: arithmetic over stdio `{mcp_command()}`\n"
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    # A tool-scoped policy that forbids the MCP tool by name.
    runtime.tool_policies.append(
        Policy(name="No Adding", statement="never", fallback_expression="tool != 'calc: add'")
    )
    bound = runtime.connect_mcp()
    try:
        blocked = runtime.tool_binder.logged_handler(runtime, bound[0])(a=1, b=2)
        assert "blocked by policy 'No Adding'" in blocked
    finally:
        runtime.disconnect_mcp()


# ---------------------------------------------------------------------------
# The runtime dashboard: a self-contained HTML view of the reasoning trail,
# the TensorBoard-equivalent, rendered from the log with zero dependencies.
# ---------------------------------------------------------------------------


def test_dashboard_renders_self_contained_html_from_a_runtime(tmp_path):
    from ear import Dashboard

    runtime = load_runtime(write_stack(tmp_path / "stack", **MINIMAL_STACK))
    runtime.reason(Intent(text="Underwrite a consumer loan", context={"loan_amount": 5000}))
    runtime.reason(Intent(text="Underwrite another loan", context={"loan_amount": 6000}))

    html = Dashboard().render(runtime)
    # Self-contained: a real document, no external fetches of any kind.
    assert html.startswith("<!doctype html>")
    assert "<style>" in html and "<svg" in html
    assert "http://" not in html and "https://" not in html
    assert "cdn" not in html.lower() and "<script src" not in html
    # It shows the runtime and its cycles.
    assert "Credit Risk Runtime" in html
    assert html.count('class="cycle"') == 2
    assert 'class="tile"' in html and "Cycles" in html
    # And it writes the same document to disk.
    path = tmp_path / "board.html"
    written = Dashboard().write(runtime, path)
    assert path.read_text(encoding="utf-8") == written


def test_dashboard_integrity_badge_reflects_the_hash_chain(tmp_path):
    from ear import Dashboard

    trail = tmp_path / "trail.jsonl"
    runtime = load_runtime(write_stack(tmp_path / "stack", **MINIMAL_STACK))
    runtime.reasoning_log.path = str(trail)
    runtime.reason(Intent(text="Underwrite a loan", context={"loan_amount": 5000}))

    assert "badge-good" in Dashboard().render(runtime)  # chain intact

    tampered = trail.read_text(encoding="utf-8").replace("Underwrite", "Tampered", 1)
    trail.write_text(tampered, encoding="utf-8")
    assert "badge-bad" in Dashboard().render(runtime)  # verify catches the edit


def test_dashboard_surfaces_governance_and_tool_calls(tmp_path):
    from ear import Dashboard

    stack = dict(MINIMAL_STACK)
    stack["memory"] = "## Tools\n\n- credit lookup: fetch the bureau score\n"
    runtime = load_runtime(write_stack(tmp_path / "stack", **stack))
    runtime.tool_binder.bind("credit lookup", lambda applicant: "score 700")

    from ear.tool_binder import BoundTool

    runtime.tool_binder.logged_handler(
        runtime, BoundTool(name="credit lookup", description="x", handler=lambda applicant: "score 700")
    )(applicant="a1")
    runtime.reason(Intent(text="Underwrite a loan", context={"loan_amount": 5000}))
    with pytest.raises(PermissionError):
        runtime.reason(Intent(text="Underwrite oversized", context={"loan_amount": 90000}))

    html = Dashboard().render(runtime)
    assert "Governance" in html and "Loan Amount Cap" in html
    assert "Tool calls" in html and "credit lookup" in html
    assert "v-bad" in html  # the blocked policy is flagged


def test_dashboard_rebuilds_from_a_persisted_jsonl_trail(tmp_path):
    from ear import Dashboard
    from ear.reasoning_log import ReasoningLog

    trail = tmp_path / "trail.jsonl"
    runtime = load_runtime(write_stack(tmp_path / "stack", **MINIMAL_STACK))
    runtime.reasoning_log.path = str(trail)
    runtime.reason(Intent(text="Underwrite a loan", context={"loan_amount": 5000}))
    original = len(runtime.reasoning_log.records)

    # Lossless reconstruction from disk, and a dashboard straight from the path.
    rebuilt = ReasoningLog.from_trail(str(trail))
    assert len(rebuilt.records) == original
    assert rebuilt.records[0].stage == "intent"
    html = Dashboard().render(str(trail))
    assert html.startswith("<!doctype html>") and 'class="cycle"' in html
