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
    AllProvidersFailed,
    Assessor,
    Auditor,
    Composer,
    Decider,
    Deliberator,
    Discoverer,
    Evidence,
    Executor,
    Experience,
    Explainer,
    Goal,
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
    Router,
    RoutingStrategy,
    Runtime,
    Scheduler,
    Selector,
    Skill,
    Step,
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


def test_workflow_add_step_delegates_to_persona():
    persona = Persona(name="Underwriter")
    workflow = Workflow(name="Underwriting Workflow")
    workflow.add_step("Band the applicant's credit profile", persona=persona)
    workflow.add_step("Decide approval within policy", persona=persona)

    assert [s.instruction for s in workflow.steps] == [
        "Band the applicant's credit profile",
        "Decide approval within policy",
    ]
    assert all(isinstance(s, Step) for s in workflow.steps)
    assert workflow.steps[0].persona is persona
    # Same persona delegated twice -> de-duplicated by identity.
    assert workflow.delegated_personas() == [persona]


def test_workflow_add_policy_attaches_governing_policy():
    workflow = Workflow(name="Underwriting Workflow")
    policy = Policy(name="DTI Ceiling", fallback_expression="debt_to_income <= 0.45")
    workflow.add_policy(policy)
    assert workflow.policies == [policy]


def test_governor_returns_violated_workflow_policies():
    workflow = Workflow(name="Underwriting Workflow")
    policy = Policy(name="DTI Ceiling", fallback_expression="debt_to_income <= 0.45")
    workflow.add_policy(policy)
    runtime = Runtime(name="Credit-Risk-Runtime")

    intent = Intent(text="high DTI application", context={"debt_to_income": 0.60})
    assert Governor().govern_workflows(runtime, intent, [workflow]) == [policy]


def test_runtime_reason_blocks_on_workflow_policy_violation():
    runtime = Runtime(name="Credit-Risk-Runtime")
    workflow = Workflow(name="Underwriting Workflow")
    workflow.add_policy(Policy(name="DTI Ceiling", fallback_expression="debt_to_income <= 0.45"))
    process = Process(name="Evaluate Credit Application")
    process.add_workflow(workflow)
    runtime.add_process(process)

    with pytest.raises(PermissionError, match="Workflow policy violated: DTI Ceiling"):
        runtime.reason(Intent(text="Evaluate application", context={"debt_to_income": 0.60}))


def test_reasoner_renders_workflow_steps_and_delegated_persona():
    persona = Persona(name="Underwriter", instructions="Be conservative on risk")
    persona.add_skill(Skill(name="assess-dti", prompt="Decline if DTI exceeds 0.45"))
    workflow = Workflow(name="Underwriting Workflow")
    workflow.add_step("Band the applicant's credit profile", persona=persona)

    decision = Reasoner().reason(Intent(text="Evaluate application"), plan=[workflow])
    # Offline default reasoning echoes the stacked capability names, proving the
    # narrated step and its delegated persona reached the reasoner.
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
# Offline: the omni-route Router -- selection order, fallback and the
# circuit-breaker cooldown are plain Python, exercised with fake per-provider
# callables and no LLM configured at all.
# ---------------------------------------------------------------------------


class _Clock:
    """A hand-cranked clock so cooldown windows are deterministic in tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def fake_provider(name: str, result: str = None, fail: bool = False, **meta) -> ModelBinding:
    """A ModelBinding whose `lm` is a fake callable -- so `build()` returns
    it without importing dspy, and `dispatch` can drive it offline."""
    binding = ModelBinding(provider=name, model=name, **meta)

    def lm(**kwargs):
        if fail:
            raise RuntimeError(f"{name} failed")
        return result if result is not None else name

    binding.lm = lm
    return binding


def test_router_is_a_drop_in_model_binding():
    router = Router(providers=[fake_provider("a"), fake_provider("b")])
    assert "omni-route" in router.model_id
    assert hasattr(router, "activate") and hasattr(router, "lm")


def test_router_priority_orders_by_priority_number():
    router = Router(
        providers=[
            fake_provider("a", priority=30),
            fake_provider("b", priority=10),
            fake_provider("c", priority=20),
        ]
    )
    assert [p.provider for p in router.order()] == ["b", "c", "a"]


def test_router_cheapest_orders_by_cost():
    router = Router(
        providers=[
            fake_provider("a", cost_per_1k=5.0),
            fake_provider("b", cost_per_1k=1.0),
            fake_provider("c", cost_per_1k=3.0),
        ],
        strategy=RoutingStrategy.CHEAPEST,
    )
    assert [p.provider for p in router.order()] == ["b", "c", "a"]


def test_router_free_first_prefers_free_providers():
    router = Router(
        providers=[
            fake_provider("paid", priority=1, is_free=False),
            fake_provider("free", priority=9, is_free=True),
        ],
        strategy=RoutingStrategy.FREE_FIRST,
    )
    assert [p.provider for p in router.order()] == ["free", "paid"]


def test_router_round_robin_rotates_the_starting_provider():
    router = Router(
        providers=[fake_provider("a"), fake_provider("b"), fake_provider("c")],
        strategy=RoutingStrategy.ROUND_ROBIN,
    )
    assert [p.provider for p in router.order()] == ["a", "b", "c"]
    assert [p.provider for p in router.order()] == ["b", "c", "a"]
    assert [p.provider for p in router.order()] == ["c", "a", "b"]


def test_router_weighted_order_is_deterministic_and_complete():
    import random

    router = Router(
        providers=[fake_provider("a", weight=1.0), fake_provider("b", weight=50.0)],
        strategy=RoutingStrategy.WEIGHTED,
        rng=random.Random(0),
    )
    order = [p.provider for p in router.order()]
    assert set(order) == {"a", "b"} and len(order) == 2


def test_router_dispatch_returns_first_successful_provider():
    router = Router(providers=[fake_provider("a", result="from-a"), fake_provider("b", result="from-b")])
    assert router.dispatch(lambda lm: lm()) == "from-a"
    assert router.last_served.provider == "a"


def test_router_falls_back_on_failure_and_benches_provider():
    clock = _Clock()
    a = fake_provider("a", fail=True, priority=10)
    b = fake_provider("b", result="from-b", priority=20)
    router = Router(providers=[a, b], cooldown_seconds=30.0, clock=clock)

    # a errors, so the call falls back to b.
    assert router.dispatch(lambda lm: lm()) == "from-b"
    assert router.last_served is b
    # a is now benched by the circuit breaker; only b is offered next.
    assert [p.model_id for p in router.order()] == ["b/b"]
    # once the cooldown elapses, a returns to the rotation.
    clock.now = 30.0
    assert [p.model_id for p in router.order()] == ["a/a", "b/b"]


def test_router_success_resets_a_providers_failure_state():
    clock = _Clock()
    flaky = fake_provider("flaky", fail=True, priority=10)
    backup = fake_provider("backup", result="ok", priority=20)
    router = Router(providers=[flaky, backup], max_failures=2, cooldown_seconds=30.0, clock=clock)

    # First failure: with max_failures=2 the breaker has not tripped yet, so
    # flaky is still offered (it just isn't the one that served the call).
    router.dispatch(lambda lm: lm())
    assert flaky in router.order()

    # Recover flaky, then a success clears its accumulated failure count.
    flaky.lm = lambda **kwargs: "flaky-back"
    assert router.dispatch(lambda lm: lm()) == "flaky-back"
    assert router._failures.get("flaky/flaky", 0) == 0


def test_router_raises_when_every_provider_fails():
    router = Router(providers=[fake_provider("a", fail=True), fake_provider("b", fail=True)])
    with pytest.raises(AllProvidersFailed):
        router.dispatch(lambda lm: lm())


def test_router_from_spec_parses_shorthand():
    router = Router.from_spec("anthropic/claude-opus-4-8, openai/gpt-4o; groq/llama-3.3-70b")
    assert [(p.provider, p.model) for p in router.providers] == [
        ("anthropic", "claude-opus-4-8"),
        ("openai", "gpt-4o"),
        ("groq", "llama-3.3-70b"),
    ]
    assert [p.priority for p in router.providers] == [0, 1, 2]


def test_router_from_spec_parses_json():
    router = Router.from_spec(
        '[{"provider": "anthropic", "model": "claude-opus-4-8", "priority": 10},'
        ' {"provider": "groq", "model": "llama-3.3-70b", "free": true}]'
    )
    assert router.providers[0].priority == 10
    assert router.providers[1].is_free is True
    assert router.providers[1].priority == 1


def test_router_from_env_reads_the_environment(monkeypatch):
    monkeypatch.setenv("EAR_ROUTER", "anthropic/claude-opus-4-8, openai/gpt-4o")
    router = Router.from_env("EAR_ROUTER")
    assert [p.provider for p in router.providers] == ["anthropic", "openai"]


def test_router_from_env_raises_when_unset(monkeypatch):
    monkeypatch.delenv("EAR_ROUTER", raising=False)
    with pytest.raises(ValueError):
        Router.from_env("EAR_ROUTER")


def test_router_activate_builds_a_routing_lm():
    router = Router(providers=[fake_provider("a")])
    lm = router.activate()
    assert lm is router.lm and lm is not None


# ---------------------------------------------------------------------------
# Offline: the Goal loop -- completion is judged by the Assessor's safe-eval
# fallback (no ModelBinding), so goal-driven iteration is deterministic.
# ---------------------------------------------------------------------------


def test_assessor_fallback_completes_when_expression_holds():
    goal = Goal(fallback_expression="'approved' in decision")
    done, blocker = Assessor().assess(Runtime(name="R"), Intent(text="x"), goal, "approved")
    assert done is True and blocker == ""


def test_assessor_fallback_is_not_done_when_expression_fails():
    goal = Goal(fallback_expression="'approved' in decision")
    done, _ = Assessor().assess(Runtime(name="R"), Intent(text="x"), goal, "declined")
    assert done is False


def test_assessor_fallback_missing_variable_is_not_done():
    # A variable the goal expects isn't set yet -> keep iterating, don't crash.
    goal = Goal(fallback_expression="risk_grade in ('A', 'B')")
    done, _ = Assessor().assess(Runtime(name="R"), Intent(text="x"), goal, "ungraded text")
    assert done is False


def test_assessor_without_expression_completes_after_one_cycle():
    # No fallback expression and no LLM -> nothing to iterate on, so done.
    goal = Goal(statement="graded and decided")
    done, _ = Assessor().assess(Runtime(name="R"), Intent(text="x"), goal, "anything")
    assert done is True


def test_assessor_fallback_rejects_unsafe_expression():
    goal = Goal(fallback_expression="__import__('os').system('echo hi')")
    with pytest.raises(ValueError):
        Assessor().assess(Runtime(name="R"), Intent(text="x"), goal, "d")


def test_intent_continued_with_threads_prior_decisions():
    first = Intent(text="Evaluate").continued_with("cycle-1")
    assert first.context["_prior_decision"] == "cycle-1"
    assert first.context["_prior_decisions"] == ["cycle-1"]
    second = first.continued_with("cycle-2")
    assert second.context["_prior_decisions"] == ["cycle-1", "cycle-2"]
    assert second.text == "Evaluate"


def test_runtime_reason_without_a_goal_runs_a_single_cycle():
    runtime = Runtime(name="Credit-Risk-Runtime")
    runtime.add_process(Process(name="Evaluate Credit Application"))
    runtime.reason(Intent(text="Evaluate a credit application"))
    assert len(runtime.memory.working) == 1
    assert runtime.experience.observations == 1


def test_runtime_reason_with_a_met_goal_stops_after_one_cycle():
    runtime = Runtime(name="Credit-Risk-Runtime")
    runtime.add_process(Process(name="Evaluate Credit Application"))
    # The offline reasoner's output contains the word "resolved".
    goal = Goal(fallback_expression="'resolved' in decision", max_cycles=5)
    runtime.reason(Intent(text="Evaluate a credit application"), goal=goal)
    assert len(runtime.memory.working) == 1


def test_runtime_reason_with_an_unmet_goal_iterates_to_the_cap():
    runtime = Runtime(name="Credit-Risk-Runtime")
    runtime.add_process(Process(name="Evaluate Credit Application"))
    goal = Goal(fallback_expression="'never-appears' in decision", max_cycles=3)
    runtime.reason(Intent(text="Evaluate a credit application"), goal=goal)
    assert len(runtime.memory.working) == 3
    assert runtime.experience.observations == 3


def test_runtime_reason_stops_when_the_assessor_reports_a_blocker():
    runtime = Runtime(name="Credit-Risk-Runtime")
    runtime.add_process(Process(name="Evaluate Credit Application"))
    runtime.assessor.assess = lambda *args, **kwargs: (False, "needs_input")
    goal = Goal(fallback_expression="'x' in decision", max_cycles=5)
    runtime.reason(Intent(text="Evaluate application"), goal=goal)
    assert len(runtime.memory.working) == 1


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


@requires_anthropic_key
def test_router_falls_back_across_providers_in_a_real_runtime_cycle():
    # A broken provider (a model that does not exist) sits ahead of a working
    # one by priority. Every judgment-laden stage should route past the broken
    # provider to the working model, proving cross-provider fallback end to end.
    broken = ModelBinding(provider="anthropic", model="claude-does-not-exist-9999", priority=10)
    working = ModelBinding(provider="anthropic", model=ANTHROPIC_TEST_MODEL, priority=20)
    router = Router.across(broken, working, strategy=RoutingStrategy.PRIORITY)

    runtime = Runtime(name="Credit-Risk-Runtime", model_binding=router)
    process = Process(
        name="Evaluate Credit Application",
        description="Reviews a loan applicant's credit history, income and debts to decide approval.",
    )
    process.add_workflow(Workflow(name="Credit Review Workflow"))
    runtime.add_process(process)

    decision = runtime.reason(
        Intent(text="Evaluate a $5,000 loan for an applicant with credit score 780")
    )
    assert isinstance(decision, str) and decision.strip()
    # The working provider is the one that actually served the reasoning.
    assert router.last_served is working


@requires_anthropic_key
def test_assessor_judges_goal_completion_with_llm():
    binding = claude_binding()
    binding.activate()
    runtime = Runtime(name="Credit-Risk-Runtime", model_binding=binding)
    goal = Goal(statement="The applicant has been given a final approve-or-decline decision.")

    done, _ = Assessor().assess(
        runtime, Intent(text="Evaluate a loan"), goal, "Decision: approve the $5,000 loan."
    )
    assert done is True

    not_done, _ = Assessor().assess(
        runtime, Intent(text="Evaluate a loan"), goal, "I still need the applicant's income before deciding."
    )
    assert not_done is False


@requires_anthropic_key
def test_runtime_reason_iterates_toward_an_llm_judged_goal():
    binding = claude_binding()
    runtime = Runtime(name="Credit-Risk-Runtime", model_binding=binding)
    process = Process(
        name="Evaluate Credit Application",
        description="Reviews a loan applicant's credit history, income and debts to decide approval.",
    )
    process.add_workflow(Workflow(name="Credit Review Workflow"))
    runtime.add_process(process)

    decision = runtime.reason(
        Intent(
            text="Evaluate a $5,000 loan for an applicant with credit score 780 and low debt",
            context={"credit_score": 780, "debt_to_income": 0.2},
        ),
        goal=Goal(
            statement="A clear approve or decline decision has been stated.",
            max_cycles=3,
        ),
    )
    assert isinstance(decision, str) and decision.strip()
    # It reached a decision within the cap.
    assert 1 <= len(runtime.memory.working) <= 3
