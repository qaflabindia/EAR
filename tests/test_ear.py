"""Tests for the `ear` package.

Two tiers:

* Offline tests exercise every stage's deterministic fallback path (no
  ModelBinding active), so the package is fully testable without any
  credentials.
* Live tests exercise the natural-language reasoning paths (Policy
  judgment, Discoverer relevance ranking, the Reasoner's decision,
  Explainer's prose) against a real Claude model. They are skipped
  automatically when no `ANTHROPIC_API_KEY` is set in the environment --
  the key is never hardcoded here, only read from the environment at test
  time -- and use a small, fast model by default to keep cost and latency
  down; override with `ANTHROPIC_TEST_MODEL`.
"""

from __future__ import annotations

import os

import pytest

from ear import (
    Adaptation,
    AdaptationBank,
    Adapter,
    Auditor,
    Composer,
    Decider,
    Deliberator,
    Discoverer,
    Evidence,
    Executor,
    Experience,
    Explainer,
    Governor,
    Initializer,
    Intent,
    Learner,
    Memory,
    ModelBinding,
    Orchestrator,
    Performer,
    Persona,
    Policy,
    Process,
    Reasoner,
    Recaller,
    Runtime,
    Scheduler,
    Selector,
    Skill,
    Validator,
    Workflow,
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_TEST_MODEL = os.environ.get("ANTHROPIC_TEST_MODEL", "claude-haiku-4-5")

requires_anthropic_key = pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY is not set in the environment -- live-LLM tests are skipped",
)


def claude_binding() -> ModelBinding:
    return ModelBinding(provider="anthropic", model=ANTHROPIC_TEST_MODEL)


# ---------------------------------------------------------------------------
# Offline: deterministic-fallback paths, no ModelBinding active.
# ---------------------------------------------------------------------------


def test_skill_invoke():
    skill = Skill(name="add", handler=lambda a, b: a + b)
    assert skill.invoke(2, 3) == 5


def test_skill_invoke_without_handler_raises():
    skill = Skill(name="noop")
    with pytest.raises(NotImplementedError):
        skill.invoke()


def test_persona_skill_lookup():
    persona = Persona(name="Underwriter")
    persona.add_skill(Skill(name="add"))
    assert persona.get_skill("add") is not None
    assert persona.get_skill("missing") is None


def test_workflow_holds_personas():
    workflow = Workflow(name="Credit Review Workflow")
    workflow.add_persona(Persona(name="Underwriter"))
    assert len(workflow.personas) == 1


def test_process_holds_workflows():
    process = Process(name="Evaluate Credit Application")
    process.add_workflow(Workflow(name="Credit Review Workflow"))
    assert len(process.workflows) == 1


def test_policy_fallback_passes_when_expression_holds():
    policy = Policy(name="Debt-to-Income Policy", fallback_expression="debt_to_income <= 0.43")
    assert policy.evaluate(debt_to_income=0.30) is True


def test_policy_fallback_fails_when_expression_violated():
    policy = Policy(name="Debt-to-Income Policy", fallback_expression="debt_to_income <= 0.43")
    assert policy.evaluate(debt_to_income=0.60) is False


def test_policy_without_fallback_or_llm_is_not_applicable():
    policy = Policy(name="Debt-to-Income Policy", statement="Debt-to-income must stay reasonable")
    assert policy.evaluate() is True


def test_policy_fallback_rejects_unsafe_expressions():
    policy = Policy(name="Malicious", fallback_expression="__import__('os').system('echo hi')")
    with pytest.raises(ValueError):
        policy.evaluate()


def test_governor_returns_violated_policies():
    policy = Policy(name="DTI Policy", fallback_expression="debt_to_income <= 0.43")
    runtime = Runtime(name="Credit-Risk-Runtime")
    runtime.add_policy(policy)

    intent = Intent(text="high DTI application", context={"debt_to_income": 0.60})
    assert Governor().govern(runtime, intent) == [policy]


def test_initializer_without_model_binding_returns_none():
    runtime = Runtime(name="Credit-Risk-Runtime")
    assert Initializer().initialize(runtime) is None


def test_discoverer_matches_processes_by_keyword():
    runtime = Runtime(name="Credit-Risk-Runtime")
    runtime.add_process(Process(name="Evaluate Credit Application"))
    runtime.add_process(Process(name="Cancel Lunch Reservation"))

    matches = Discoverer().discover(runtime, Intent(text="Evaluate a new credit application"))
    assert [p.name for p in matches] == ["Evaluate Credit Application"]


def test_discoverer_falls_back_to_all_processes_without_matches():
    runtime = Runtime(name="Credit-Risk-Runtime")
    process = Process(name="Evaluate Credit Application")
    runtime.add_process(process)

    matches = Discoverer().discover(runtime, Intent(text="zzz"))
    assert matches == [process]


def test_selector_deduplicates_by_name():
    process_a = Process(name="Evaluate Credit Application")
    process_b = Process(name="Evaluate Credit Application")
    process_c = Process(name="Cancel Application")

    selected = Selector().select(None, [process_a, process_b, process_c])
    assert selected == [process_a, process_c]


def test_composer_flattens_workflows_from_selected_processes():
    workflow = Workflow(name="Credit Review Workflow")
    process = Process(name="Evaluate Credit Application")
    process.add_workflow(workflow)

    plan = Composer().compose([process])
    assert plan == [workflow]


def test_scheduler_returns_defensive_copy():
    workflow = Workflow(name="Credit Review Workflow")
    plan = [workflow]

    scheduled = Scheduler().schedule(plan)
    assert scheduled == plan
    assert scheduled is not plan


def test_validator_rejects_blank_decision():
    with pytest.raises(ValueError):
        Validator().validate("   ")


def test_validator_validate_candidates_rejects_wrong_item_type():
    with pytest.raises(TypeError):
        Validator().validate_candidates([Workflow(name="Credit Review Workflow")])


def test_decider_rejects_none_deliberation():
    with pytest.raises(ValueError):
        Decider().decide(None)


def test_decider_passes_through_deliberation():
    assert Decider().decide("approved") == "approved"


def test_deliberator_deliberates_via_runtime_reasoner():
    runtime = Runtime(name="Credit-Risk-Runtime")
    result = Deliberator().deliberate(runtime, Intent(text="Evaluate application"))
    assert "Credit-Risk-Runtime" in result


def test_performer_chains_deliberate_decide_validate():
    runtime = Runtime(name="Credit-Risk-Runtime")
    result = Performer().perform(runtime, Intent(text="Evaluate application"))
    assert "Credit-Risk-Runtime" in result


def test_executor_executes_via_performer():
    runtime = Runtime(name="Credit-Risk-Runtime")
    result = Executor().execute(runtime, Intent(text="Evaluate application"))
    assert "Credit-Risk-Runtime" in result


def test_orchestrator_orchestrates_via_executor():
    runtime = Runtime(name="Credit-Risk-Runtime")
    result = Orchestrator().orchestrate(runtime, Intent(text="Evaluate application"))
    assert "Credit-Risk-Runtime" in result


def test_recaller_recalls_memory_context_window():
    memory = Memory(capacity=10)
    memory.record("past application", decision="approved")

    recalled = Recaller().recall(memory, Intent(text="anything"))
    assert "past application" in recalled


def test_explainer_falls_back_to_f_string_without_model_binding():
    explanation = Explainer().explain(Evidence(basis="Cleared DTI Policy"), "approved")
    assert explanation == "Cleared DTI Policy -> approved"


def test_auditor_marks_evidence_audited():
    evidence = Evidence(basis="Cleared DTI Policy")
    Auditor().audit(evidence)
    assert evidence.sources["audited"] is True


def test_learner_learns_memory_entry_into_experience():
    memory = Memory(capacity=10)
    entry = memory.record("a", decision="approved")
    experience = Experience()

    Learner().learn(experience, entry)
    assert experience.observations == 1
    assert experience.decision_counts == {"approved": 1}


def test_adapter_throttles_adaptation_to_every_n_observations():
    bank = AdaptationBank()
    experience = Experience()
    adapter = Adapter(adapt_every=2)
    memory = Memory(capacity=10)

    experience.observe_entry(memory.record("a", decision="approved"))
    assert adapter.adapt(bank, experience) is None
    assert bank.impressions == []

    experience.observe_entry(memory.record("b", decision="approved"))
    learned = adapter.adapt(bank, experience)
    assert learned is not None
    assert bank.impressions == [learned]


def test_memory_records_and_compresses_on_overflow():
    memory = Memory(capacity=3)
    for i in range(5):
        memory.record(f"intent {i}", decision=f"decision {i}")

    assert len(memory.working) == 3
    assert memory.working[0].intent_text == "intent 2"
    assert len(memory.compressed) == 2


def test_adaptation_bank_relevant_to_keyword_overlap():
    bank = AdaptationBank()
    bank.add(Adaptation(name="dti-rule", insight="High debt-to-income applications get declined"))
    bank.add(Adaptation(name="unrelated", insight="Lunch orders are auto-approved"))

    matches = bank.relevant_to("Evaluate a high debt-to-income application")
    assert [a.name for a in matches] == ["dti-rule"]


def test_skill_is_prompt_first_without_a_handler():
    skill = Skill(name="assess-dti", prompt="Decline any application whose debt-to-income exceeds 0.43")
    assert skill.instruction() == "Decline any application whose debt-to-income exceeds 0.43"
    # A prompt-only skill needs no Python code -- the user just stacks a prompt.
    assert skill.handler is None


def test_skill_instruction_falls_back_to_description_then_name():
    assert Skill(name="x", description="does x").instruction() == "does x"
    assert Skill(name="x").instruction() == "x"


def test_reasoner_renders_stacked_capabilities_into_the_decision():
    persona = Persona(name="Underwriter", instructions="Be conservative on risk")
    persona.add_skill(Skill(name="assess-dti", prompt="Decline if DTI exceeds 0.43"))
    workflow = Workflow(name="Credit Review Workflow")
    workflow.add_persona(persona)

    decision = Reasoner().reason(Intent(text="Evaluate application"), plan=[workflow])

    # The stacked persona instruction and skill name reach the reasoning output,
    # proving the stack is no longer dropped before reasoning.
    assert "Underwriter" in decision
    assert "assess-dti" in decision


def test_runtime_threads_scheduled_stack_into_reasoning():
    runtime = Runtime(name="Credit-Risk-Runtime")
    persona = Persona(name="Underwriter", instructions="Be conservative on risk")
    persona.add_skill(Skill(name="assess-dti", prompt="Decline if DTI exceeds 0.43"))
    workflow = Workflow(name="Credit Review Workflow")
    workflow.add_persona(persona)
    process = Process(name="Evaluate Credit Application")
    process.add_workflow(workflow)
    runtime.add_process(process)

    decision = runtime.reason(Intent(text="Evaluate a credit application"))
    assert "Underwriter" in decision


def test_runtime_reason_default_path():
    runtime = Runtime(name="Credit-Risk-Runtime")
    runtime.add_policy(Policy(name="DTI Policy", fallback_expression="debt_to_income <= 0.43"))
    process = Process(name="Evaluate Credit Application")
    process.add_workflow(Workflow(name="Credit Review Workflow"))
    runtime.add_process(process)

    result = runtime.reason(Intent(text="Evaluate a credit application within policy", context={"debt_to_income": 0.30}))
    assert "Credit-Risk-Runtime" in result

    entry = runtime.memory.working[-1]
    assert entry.evidence.basis == "Resolved via the Reasoner's dependency-free default"
    assert entry.evidence.sources["plan"] == ["Credit Review Workflow"]
    assert "audited" in entry.evidence.sources
    assert "explanation" in entry.evidence.sources
    assert runtime.experience.observations == 1


def test_runtime_reason_raises_on_policy_violation():
    runtime = Runtime(name="Credit-Risk-Runtime")
    runtime.add_policy(Policy(name="DTI Policy", fallback_expression="debt_to_income <= 0.43"))

    intent = Intent(text="Evaluate an over-leveraged application", context={"debt_to_income": 0.60})
    with pytest.raises(PermissionError, match="DTI Policy"):
        runtime.reason(intent)


def test_runtime_reason_rejects_malformed_discoverer_output_via_validator():
    runtime = Runtime(name="Credit-Risk-Runtime")
    runtime.discoverer.discover = lambda *_args, **_kwargs: [Workflow(name="not-a-process")]

    with pytest.raises(TypeError, match="Discoverer candidates"):
        runtime.reason(Intent(text="Evaluate application"))


# ---------------------------------------------------------------------------
# Live: natural-language reasoning against a real Claude model.
# ---------------------------------------------------------------------------


@requires_anthropic_key
def test_policy_judges_natural_language_statement_with_llm():
    binding = claude_binding()
    binding.activate()
    policy = Policy(
        name="DTI Policy",
        statement="The applicant's debt-to-income ratio must not exceed 0.43.",
    )

    assert policy.evaluate(model_binding=binding, debt_to_income=0.25) is True
    assert policy.evaluate(model_binding=binding, debt_to_income=0.85) is False


@requires_anthropic_key
def test_discoverer_ranks_processes_by_relevance_with_llm():
    binding = claude_binding()
    binding.activate()
    runtime = Runtime(name="Credit-Risk-Runtime", model_binding=binding)
    runtime.add_process(
        Process(
            name="Evaluate Credit Application",
            description="Reviews a loan applicant's credit history, income and debts to decide approval.",
        )
    )
    runtime.add_process(
        Process(
            name="Cancel Lunch Reservation",
            description="Cancels a previously booked lunch reservation in the office cafeteria.",
        )
    )

    matches = Discoverer().discover(runtime, Intent(text="Review a new loan applicant's debt and income"))
    assert "Evaluate Credit Application" in [p.name for p in matches]
    assert "Cancel Lunch Reservation" not in [p.name for p in matches]


@requires_anthropic_key
def test_explainer_writes_explanation_with_llm():
    binding = claude_binding()
    binding.activate()
    explanation = Explainer().explain(
        Evidence(basis="Cleared the debt-to-income policy at 0.30"),
        "approved",
        model_binding=binding,
    )
    assert isinstance(explanation, str) and explanation.strip()


@requires_anthropic_key
def test_runtime_reason_resolves_intent_with_llm_and_no_attached_program():
    binding = claude_binding()
    runtime = Runtime(name="Credit-Risk-Runtime", model_binding=binding)
    runtime.add_policy(Policy(name="DTI Policy", statement="Debt-to-income ratio must not exceed 0.43."))
    process = Process(
        name="Evaluate Credit Application",
        description="Reviews a loan applicant's credit history, income and debts to decide approval.",
    )
    process.add_workflow(Workflow(name="Credit Review Workflow"))
    runtime.add_process(process)

    decision = runtime.reason(
        Intent(
            text="Evaluate a credit application for a $10,000 personal loan",
            context={"debt_to_income": 0.28, "credit_score": 720},
        )
    )

    assert isinstance(decision, str) and decision.strip()
    entry = runtime.memory.working[-1]
    assert entry.evidence.basis.startswith("Resolved via ModelBinding LM")


@requires_anthropic_key
def test_runtime_reason_blocks_on_llm_judged_policy_violation():
    binding = claude_binding()
    runtime = Runtime(name="Credit-Risk-Runtime", model_binding=binding)
    runtime.add_policy(Policy(name="DTI Policy", statement="Debt-to-income ratio must not exceed 0.43."))

    intent = Intent(
        text="Evaluate a credit application from a heavily indebted applicant",
        context={"debt_to_income": 0.91},
    )
    with pytest.raises(PermissionError, match="DTI Policy"):
        runtime.reason(intent)


@requires_anthropic_key
def test_reasoner_compiles_default_dspy_program_with_llm():
    binding = claude_binding()
    binding.activate()
    reasoner = Reasoner().compile_with_dspy()

    decision = reasoner.reason(Intent(text="Approve a $5,000 loan for an applicant with credit score 780"))
    assert decision is not None
