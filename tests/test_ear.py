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
    Router,
    RouterProvider,
    RoutingStrategy,
    Runtime,
    Scheduler,
    Selector,
    Skill,
    SkillSelector,
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
# LMs (matching the native `LM.complete()`/`.history` shape) and no network.
# ---------------------------------------------------------------------------


class _Clock:
    """A hand-cranked clock so cooldown windows are deterministic in tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FakeLM:
    """A stand-in for `ear.llm.LM`: same `.complete()`/`.history` shape, no
    network, so Router can be tested without a real provider."""

    def __init__(self, name: str, fail: bool = False, reply: str = "") -> None:
        self.name = name
        self.fail = fail
        self.reply = reply
        self.history: list[dict] = []

    def complete(self, prompt: str, system: str = "") -> str:
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        self.history.append({"usage": {"input_tokens": 1, "output_tokens": 1}, "latency_ms": 1, "retries": 0})
        return self.reply or self.name


def fake_provider(name: str, **kwargs) -> RouterProvider:
    """A RouterProvider whose binding's `lm` is a fake -- so `activate()`
    returns it without any network, and `dispatch` can drive it offline."""
    binding = ModelBinding(provider=name, model=name)
    priority = kwargs.pop("priority", 100)
    cost_per_1k = kwargs.pop("cost_per_1k", 0.0)
    weight = kwargs.pop("weight", 1.0)
    is_free = kwargs.pop("is_free", False)
    binding.lm = _FakeLM(name, **kwargs)
    return RouterProvider(binding=binding, priority=priority, cost_per_1k=cost_per_1k, weight=weight, is_free=is_free)


def test_router_is_a_drop_in_model_binding():
    router = Router(providers=[fake_provider("a"), fake_provider("b")])
    assert "omni-route" in router.model_id
    assert hasattr(router, "activate") and hasattr(router, "lm")


def test_router_priority_orders_by_priority_number():
    router = Router(providers=[fake_provider("a", priority=30), fake_provider("b", priority=10), fake_provider("c", priority=20)])
    assert [p.binding.provider for p in router.order()] == ["b", "c", "a"]


def test_router_cheapest_orders_by_cost():
    router = Router(
        providers=[fake_provider("a", cost_per_1k=5.0), fake_provider("b", cost_per_1k=1.0), fake_provider("c", cost_per_1k=3.0)],
        strategy=RoutingStrategy.CHEAPEST,
    )
    assert [p.binding.provider for p in router.order()] == ["b", "c", "a"]


def test_router_free_first_prefers_free_providers():
    router = Router(
        providers=[fake_provider("paid", priority=1, is_free=False), fake_provider("free", priority=9, is_free=True)],
        strategy=RoutingStrategy.FREE_FIRST,
    )
    assert [p.binding.provider for p in router.order()] == ["free", "paid"]


def test_router_round_robin_rotates_the_starting_provider():
    router = Router(providers=[fake_provider("a"), fake_provider("b"), fake_provider("c")], strategy=RoutingStrategy.ROUND_ROBIN)
    assert [p.binding.provider for p in router.order()] == ["a", "b", "c"]
    assert [p.binding.provider for p in router.order()] == ["b", "c", "a"]
    assert [p.binding.provider for p in router.order()] == ["c", "a", "b"]


def test_router_weighted_order_is_deterministic_and_complete():
    import random

    router = Router(
        providers=[fake_provider("a", weight=1.0), fake_provider("b", weight=50.0)],
        strategy=RoutingStrategy.WEIGHTED,
        rng=random.Random(0),
    )
    order = [p.binding.provider for p in router.order()]
    assert set(order) == {"a", "b"} and len(order) == 2


def test_router_dispatch_returns_first_successful_provider():
    router = Router(providers=[fake_provider("a", reply="from-a"), fake_provider("b", reply="from-b")])
    assert router.dispatch(lambda lm: lm.complete("x")) == "from-a"
    assert router.last_served.binding.provider == "a"


def test_router_falls_back_on_failure_and_benches_provider():
    clock = _Clock()
    a = fake_provider("a", fail=True, priority=10)
    b = fake_provider("b", reply="from-b", priority=20)
    router = Router(providers=[a, b], cooldown_seconds=30.0, clock=clock)

    assert router.dispatch(lambda lm: lm.complete("x")) == "from-b"
    assert router.last_served is b
    # a is now benched by the circuit breaker; only b is offered next.
    assert [p.model_id for p in router.order()] == ["b/b"]
    # once the cooldown elapses, a returns to the rotation.
    clock.now = 30.0
    assert [p.model_id for p in router.order()] == ["a/a", "b/b"]


def test_router_success_resets_a_providers_failure_state():
    clock = _Clock()
    flaky = fake_provider("flaky", fail=True, priority=10)
    backup = fake_provider("backup", reply="ok", priority=20)
    router = Router(providers=[flaky, backup], max_failures=2, cooldown_seconds=30.0, clock=clock)

    router.dispatch(lambda lm: lm.complete("x"))
    assert flaky in router.order()

    flaky.binding.lm.fail = False
    flaky.binding.lm.reply = "flaky-back"
    assert router.dispatch(lambda lm: lm.complete("x")) == "flaky-back"
    assert router._failures.get("flaky/flaky", 0) == 0


def test_router_raises_when_every_provider_fails():
    router = Router(providers=[fake_provider("a", fail=True), fake_provider("b", fail=True)])
    with pytest.raises(AllProvidersFailed):
        router.dispatch(lambda lm: lm.complete("x"))


def test_router_from_spec_parses_shorthand():
    router = Router.from_spec("anthropic/claude-opus-4-8, openai/gpt-4o; groq/llama-3.3-70b")
    assert [(p.binding.provider, p.binding.model) for p in router.providers] == [
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


def test_router_from_env_reads_the_environment(monkeypatch):
    monkeypatch.setenv("EAR_ROUTER", "anthropic/claude-opus-4-8, openai/gpt-4o")
    router = Router.from_env("EAR_ROUTER")
    assert [p.binding.provider for p in router.providers] == ["anthropic", "openai"]


def test_router_from_env_raises_when_unset(monkeypatch):
    monkeypatch.delenv("EAR_ROUTER", raising=False)
    with pytest.raises(ValueError):
        Router.from_env("EAR_ROUTER")


def test_router_activate_builds_a_routing_lm_facade():
    router = Router(providers=[fake_provider("a")])
    lm = router.activate()
    assert lm is router.lm and lm is not None


def test_router_routes_a_real_native_judgment_with_fallback():
    """The full, real stack: a native Judgment's `.run()` call is served
    through a Router's fallback path, proving Router is a genuine drop-in
    for anything that calls `lm.complete()` and parses a markdown reply --
    not just a synthetic string-in/string-out fake."""
    from ear.signatures import JudgePolicyCompliance

    reply = "## Complies\n\nyes\n\n## Rationale\n\nWithin the stated limit.\n"
    router = Router.across(
        ModelBinding(provider="broken", model="broken"),
        ModelBinding(provider="working", model="working"),
    )
    router.providers[0].binding.lm = _FakeLM("broken", fail=True)
    router.providers[1].binding.lm = _FakeLM("working", reply=reply)

    result = JudgePolicyCompliance.run(router.activate(), policy_statement="cap at 100", context={"amount": 50})
    assert result.complies is True
    assert "stated limit" in result.rationale
    assert router.last_served.binding.provider == "working"


# ---------------------------------------------------------------------------
# Offline: progressive skill selection -- only the skills relevant to the
# intent are stacked, ranked by keyword overlap when no LLM is active.
# ---------------------------------------------------------------------------


def _persona_with_skills(*specs) -> Persona:
    persona = Persona(name="Underwriter")
    for name, prompt in specs:
        persona.add_skill(Skill(name=name, prompt=prompt))
    return persona


def test_skill_carries_provenance_metadata():
    skill = Skill(name="risk_grade", prompt="Grade A-E", version="1.2", author="risk-team")
    assert skill.version == "1.2" and skill.author == "risk-team"


def test_skill_selector_returns_all_when_within_top_k():
    persona = _persona_with_skills(("a", "x"), ("b", "y"), ("c", "z"))
    picked = SkillSelector(top_k=5).select(persona, Intent(text="anything"))
    assert [s.name for s in picked] == ["a", "b", "c"]


def test_skill_selector_keyword_ranks_relevant_skills_first():
    persona = _persona_with_skills(
        ("lunch", "Cancel a lunch reservation"),
        ("credit", "Assess a credit application's risk"),
        ("weather", "Report today's weather"),
    )
    picked = SkillSelector(top_k=1).select(persona, Intent(text="Assess the applicant's credit risk"))
    assert [s.name for s in picked] == ["credit"]


def test_skill_selector_caps_at_top_k():
    persona = _persona_with_skills(*[(f"s{i}", "generic task") for i in range(10)])
    picked = SkillSelector(top_k=4).select(persona, Intent(text="something"))
    assert len(picked) == 4


def test_reasoner_stacks_only_selected_skills():
    persona = Persona(name="Underwriter", instructions="Be conservative")
    persona.add_skill(Skill(name="relevant-credit", prompt="assess credit risk"))
    for i in range(9):
        persona.add_skill(Skill(name=f"filler{i}", prompt="unrelated clerical task"))
    workflow = Workflow(name="Credit Review Workflow")
    workflow.add_persona(persona)

    reasoner = Reasoner(skill_selector=SkillSelector(top_k=2))
    decision = reasoner.reason(Intent(text="assess credit risk"), plan=[workflow])
    assert "relevant-credit" in decision
    assert "filler8" not in decision


def test_reasoner_without_selector_stacks_every_skill():
    persona = Persona(name="Underwriter")
    for i in range(10):
        persona.add_skill(Skill(name=f"s{i}", prompt="generic"))
    workflow = Workflow(name="WF")
    workflow.add_persona(persona)

    decision = Reasoner(skill_selector=None).reason(Intent(text="anything"), plan=[workflow])
    assert "s9" in decision


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
def test_native_lm_completes_and_accounts_usage():
    binding = claude_binding()
    lm = binding.activate()

    reply = lm.complete("Answer with one word: what colour is a clear daytime sky?")
    assert reply.strip()
    assert lm.history[-1]["usage"]["prompt_tokens"] > 0
    assert lm.history[-1]["usage"]["completion_tokens"] > 0


@requires_anthropic_key
def test_router_falls_back_across_real_providers_in_a_real_runtime_cycle():
    # A broken provider (a model that does not exist) sits ahead of a working
    # one by priority. Every judgment-laden stage should route past the
    # broken provider to the working model, proving cross-provider fallback
    # end to end against the real native LM client.
    broken = ModelBinding(provider="anthropic", model="claude-does-not-exist-9999")
    working = ModelBinding(provider="anthropic", model=ANTHROPIC_TEST_MODEL)
    router = Router.across(broken, working, strategy=RoutingStrategy.PRIORITY)

    runtime = Runtime(name="Credit-Risk-Runtime", model_binding=router)
    process = Process(
        name="Evaluate Credit Application",
        description="Reviews a loan applicant's credit history, income and debts to decide approval.",
    )
    process.add_workflow(Workflow(name="Credit Review Workflow"))
    runtime.add_process(process)

    decision = runtime.reason(Intent(text="Evaluate a $5,000 loan for an applicant with credit score 780"))
    assert isinstance(decision, str) and decision.strip()
    # The working provider is the one that actually served the reasoning.
    assert router.last_served.binding is working


@requires_anthropic_key
def test_skill_selector_ranks_by_relevance_with_llm():
    binding = claude_binding()
    lm = binding.activate()
    persona = Persona(name="Underwriter")
    persona.add_skill(Skill(name="cancel_lunch", prompt="Cancel a cafeteria lunch reservation."))
    persona.add_skill(Skill(name="assess_dti", prompt="Assess a loan applicant's debt-to-income ratio."))
    persona.add_skill(Skill(name="book_travel", prompt="Book flights and hotels for a business trip."))

    picked = SkillSelector(top_k=1).select(
        persona, Intent(text="Review the applicant's debt and income for a loan"), lm=lm
    )
    assert [s.name for s in picked] == ["assess_dti"]
