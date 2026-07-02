"""Tests for the natural-language markdown authoring layer.

Everything here runs offline: the loader is structural, and the loaded
runtime exercises its deterministic fallbacks, so the whole stacked
authoring model -- skills.md -> persona.md -> workflow.md -> process.md ->
policy.md -> memory.md -> Runtime -- is testable with no LLM configured.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ear import (
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
