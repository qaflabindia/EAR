from ear import (
    Adhyayana,
    Anubhava,
    Anukulana,
    Anushthana,
    Anveshana,
    Arambha,
    Dharma,
    Guna,
    Karma,
    Kriya,
    Ksetra,
    Manas,
    Nirnaya,
    Niyamana,
    Niyojana,
    Pariksha,
    Pramana,
    Samanvaya,
    Samskara,
    SamskaraBank,
    Samyojana,
    Sankalpa,
    Smarana,
    Smriti,
    Varana,
    Varna,
    Vicara,
    Vidya,
    Vyakhya,
)


def test_vidya_invoke():
    skill = Vidya(name="add", handler=lambda a, b: a + b)
    assert skill.invoke(2, 3) == 5


def test_vidya_invoke_without_handler_raises():
    skill = Vidya(name="noop")
    try:
        skill.invoke()
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError")


def test_guna_skill_lookup():
    persona = Guna(name="Buyer")
    persona.add_skill(Vidya(name="add"))
    assert persona.get_skill("add") is not None
    assert persona.get_skill("missing") is None


def test_varna_holds_personas():
    workflow = Varna(name="Procurement Workflow")
    workflow.add_persona(Guna(name="Buyer"))
    assert len(workflow.personas) == 1


def test_karma_holds_workflows():
    process = Karma(name="Create PO")
    process.add_workflow(Varna(name="Procurement Workflow"))
    assert len(process.workflows) == 1


def test_dharma_passes_when_rule_holds():
    policy = Dharma(name="PO Approval Policy", rule="purchase_amount <= approval_limit")
    assert policy.evaluate(purchase_amount=100, approval_limit=500) is True


def test_dharma_fails_when_rule_violated():
    policy = Dharma(name="PO Approval Policy", rule="purchase_amount <= approval_limit")
    assert policy.evaluate(purchase_amount=999, approval_limit=500) is False


def test_dharma_not_applicable_without_context():
    policy = Dharma(name="PO Approval Policy", rule="purchase_amount <= approval_limit")
    assert policy.evaluate() is True


def test_dharma_rejects_unsafe_expressions():
    policy = Dharma(name="Malicious", rule="__import__('os').system('echo hi')")
    try:
        policy.evaluate()
    except ValueError:
        pass
    else:
        raise AssertionError("expected unsafe expression to be rejected")


def test_ksetra_reason_default_path():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    runtime.add_policy(Dharma(name="PO Approval Policy", rule="purchase_amount <= approval_limit"))
    process = Karma(name="Create Purchase Order")
    process.add_workflow(Varna(name="Procurement Workflow"))
    runtime.add_process(process)

    result = runtime.reason(Sankalpa(text="Create PO for laptops under approved budget"))
    assert "Procurement-Kurukshetra" in result
    assert "Create Purchase Order" in result


def test_ksetra_reason_raises_on_policy_violation():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    runtime.add_policy(Dharma(name="PO Approval Policy", rule="purchase_amount <= approval_limit"))

    sankalpa = Sankalpa(
        text="Create PO over budget",
        context={"purchase_amount": 9999, "approval_limit": 500},
    )
    try:
        runtime.reason(sankalpa)
    except PermissionError as exc:
        assert "PO Approval Policy" in str(exc)
    else:
        raise AssertionError("expected PermissionError")


class _StubLM:
    """Stands in for a dspy.LM so tests never make a network call."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        self.calls.append(prompt)
        return ["stubbed reasoning"]


def test_manas_model_id_defaults_to_provider_prefix():
    manas = Manas(provider="openai", model="gpt-4o-mini")
    assert manas.model_id == "openai/gpt-4o-mini"


def test_manas_model_id_passthrough_when_already_qualified():
    manas = Manas(provider="openai", model="anthropic/claude-3-5-sonnet")
    assert manas.model_id == "anthropic/claude-3-5-sonnet"


def test_ksetra_reason_uses_activated_manas_lm_when_no_dspy_program():
    stub = _StubLM()
    manas = Manas(provider="openai", model="gpt-4o-mini", lm=stub)
    runtime = Ksetra(name="Procurement-Kurukshetra", manas=manas)
    process = Karma(name="Create Purchase Order")
    runtime.add_process(process)

    result = runtime.reason(Sankalpa(text="Create PO for laptops"))

    assert result == "stubbed reasoning"
    assert len(stub.calls) == 1
    assert "Create PO for laptops" in stub.calls[0]


def test_bhuddi_falls_back_to_default_without_manas_or_program():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    result = runtime.reason(Sankalpa(text="Create PO for laptops"))
    assert result.startswith("[Procurement-Kurukshetra]")


def test_smriti_records_and_compresses_on_overflow():
    smriti = Smriti(capacity=3)
    for i in range(5):
        smriti.record(f"intent {i}", decision=f"decision {i}")

    # Capacity 3 with 5 records overflows twice (at the 4th and 5th record),
    # producing one compressed summary per overflow event.
    assert len(smriti.working) == 3
    assert smriti.working[0].sankalpa_text == "intent 2"
    assert len(smriti.compressed) == 2
    assert "decision 0" in smriti.compressed[0]
    assert "decision 1" in smriti.compressed[1]


def test_smriti_context_window_includes_both_layers():
    smriti = Smriti(capacity=1)
    smriti.record("first", decision="approved")
    smriti.record("second", decision="rejected")

    window = smriti.context_window()
    assert "Earlier history (compressed)" in window
    assert "Recent history" in window
    assert "second" in window


def test_smriti_len_counts_working_and_compressed():
    smriti = Smriti(capacity=2)
    for i in range(3):
        smriti.record(f"intent {i}", decision="ok")
    assert len(smriti) == 3  # 2 working + 1 compressed summary


def test_samskara_bank_learn_from_picks_most_common_decision():
    smriti = Smriti(capacity=10)
    smriti.record("a", decision="approved")
    smriti.record("b", decision="approved")
    smriti.record("c", decision="rejected")
    anubhava = Anubhava().observe(smriti)

    bank = SamskaraBank()
    learned = bank.learn_from(anubhava)

    assert learned is not None
    assert "approved" in learned.insight
    assert bank.impressions == [learned]


def test_samskara_bank_learn_from_empty_memory_returns_none():
    assert SamskaraBank().learn_from(Anubhava()) is None


def test_samskara_bank_relevant_to_keyword_overlap():
    bank = SamskaraBank()
    bank.add(Samskara(name="escalation-rule", insight="Purchases over budget get escalated"))
    bank.add(Samskara(name="unrelated", insight="Lunch orders are auto-approved"))

    matches = bank.relevant_to("Create PO over budget for laptops")
    assert [s.name for s in matches] == ["escalation-rule"]


def test_ksetra_records_each_reason_call_into_smriti():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    runtime.reason(Sankalpa(text="first request"))
    runtime.reason(Sankalpa(text="second request"))

    assert len(runtime.smriti.working) == 2
    assert runtime.smriti.working[0].sankalpa_text == "first request"
    assert runtime.smriti.working[1].sankalpa_text == "second request"


def test_ksetra_default_reasoning_mentions_remembered_cycles():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    runtime.reason(Sankalpa(text="first request"))
    result = runtime.reason(Sankalpa(text="second request"))
    assert "1 remembered cycles" in result


def test_manas_reasoning_includes_memory_and_samskara_in_prompt():
    stub = _StubLM()
    manas = Manas(provider="openai", model="gpt-4o-mini", lm=stub)
    runtime = Ksetra(name="Procurement-Kurukshetra", manas=manas)
    runtime.smriti.record("past request", decision="approved")
    runtime.samskara.add(Samskara(name="rule", insight="past requests get approved"))

    runtime.reason(Sankalpa(text="new past requests"))

    prompt = stub.calls[0]
    assert "Memory (Smriti)" in prompt
    assert "Learned adaptations (Samskara)" in prompt
    assert "past requests get approved" in prompt


def test_pramana_str_returns_basis():
    pramana = Pramana(basis="Cleared PO Approval Policy", sources={"policy": "PO Approval Policy"})
    assert str(pramana) == "Cleared PO Approval Policy"


def test_anubhava_observe_entry_counts_decisions_and_evidence():
    anubhava = Anubhava()
    smriti = Smriti(capacity=10)
    entry = smriti.record("a", decision="approved", evidence=Pramana(basis="cleared"))

    anubhava.observe_entry(entry)

    assert anubhava.observations == 1
    assert anubhava.decision_counts == {"approved": 1}
    assert anubhava.evidence_seen == [entry.evidence]


def test_anubhava_observe_rebuilds_from_smriti_working():
    smriti = Smriti(capacity=10)
    smriti.record("a", decision="approved")
    smriti.record("b", decision="approved")
    smriti.record("c", decision="rejected")

    anubhava = Anubhava().observe(smriti)

    assert anubhava.observations == 3
    assert anubhava.most_common_decision() == ("approved", 2)


def test_anubhava_summary_ranks_by_count():
    anubhava = Anubhava()
    smriti = Smriti(capacity=10)
    anubhava.observe_entry(smriti.record("a", decision="approved"))
    anubhava.observe_entry(smriti.record("b", decision="approved"))
    anubhava.observe_entry(smriti.record("c", decision="rejected"))

    summary = anubhava.summary()
    assert "'approved': 2/3 cycles" in summary
    assert summary.index("approved") < summary.index("rejected")


def test_anubhava_len_reflects_observations():
    anubhava = Anubhava()
    smriti = Smriti(capacity=10)
    anubhava.observe_entry(smriti.record("a", decision="approved"))
    assert len(anubhava) == 1


def test_ksetra_reason_builds_pramana_and_updates_anubhava_on_default_path():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    runtime.reason(Sankalpa(text="Create PO for laptops"))

    entry = runtime.smriti.working[-1]
    assert isinstance(entry.evidence, Pramana)
    assert entry.evidence.basis == "Resolved via Bhuddi's dependency-free default"
    assert runtime.anubhava.observations == 1


def test_ksetra_reason_builds_pramana_for_manas_path():
    stub = _StubLM()
    manas = Manas(provider="openai", model="gpt-4o-mini", lm=stub)
    runtime = Ksetra(name="Procurement-Kurukshetra", manas=manas)

    runtime.reason(Sankalpa(text="Create PO for laptops"))

    entry = runtime.smriti.working[-1]
    assert entry.evidence.basis == "Resolved via Manas LM 'openai/gpt-4o-mini'"
    assert runtime.anubhava.observations == 1


def test_ksetra_reason_builds_pramana_for_dspy_program_path():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    runtime.reasoner.program = lambda sankalpa, context: "programmatic decision"

    runtime.reason(Sankalpa(text="Create PO for laptops"))

    entry = runtime.smriti.working[-1]
    assert entry.evidence.basis == "Resolved via a compiled DSPy program"
    assert runtime.anubhava.observations == 1


def test_ksetra_default_reasoning_surfaces_anubhava_experience_in_prompt():
    stub = _StubLM()
    manas = Manas(provider="openai", model="gpt-4o-mini", lm=stub)
    runtime = Ksetra(name="Procurement-Kurukshetra", manas=manas)
    runtime.reason(Sankalpa(text="first request"))

    runtime.reason(Sankalpa(text="second request"))

    prompt = stub.calls[-1]
    assert "Experience (Anubhava)" in prompt


def test_niyamana_governs_returns_violated_policies():
    policy = Dharma(name="PO Approval Policy", rule="purchase_amount <= approval_limit")
    runtime = Ksetra(name="Procurement-Kurukshetra")
    runtime.add_policy(policy)

    sankalpa = Sankalpa(text="over budget", context={"purchase_amount": 999, "approval_limit": 500})
    assert Niyamana().govern(runtime, sankalpa) == [policy]


def test_arambha_initializes_manas_when_present():
    stub = _StubLM()
    manas = Manas(provider="openai", model="gpt-4o-mini", lm=stub)
    runtime = Ksetra(name="Procurement-Kurukshetra", manas=manas)

    assert Arambha().initialize(runtime) is stub


def test_arambha_initialize_without_manas_returns_none():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    assert Arambha().initialize(runtime) is None


def test_anveshana_discovers_processes_matching_sankalpa_text():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    runtime.add_process(Karma(name="Create Purchase Order"))
    runtime.add_process(Karma(name="Cancel Lunch Reservation"))

    matches = Anveshana().discover(runtime, Sankalpa(text="Purchase some laptops"))
    assert [p.name for p in matches] == ["Create Purchase Order"]


def test_anveshana_falls_back_to_all_processes_without_matches():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    process = Karma(name="Create Purchase Order")
    runtime.add_process(process)

    matches = Anveshana().discover(runtime, Sankalpa(text="zzz"))
    assert matches == [process]


def test_varana_selects_deduplicating_by_name():
    process_a = Karma(name="Create Purchase Order")
    process_b = Karma(name="Create Purchase Order")
    process_c = Karma(name="Cancel Order")

    selected = Varana().select(None, [process_a, process_b, process_c])
    assert selected == [process_a, process_c]


def test_samyojana_composes_workflows_from_selected_processes():
    workflow = Varna(name="Procurement Workflow")
    process = Karma(name="Create Purchase Order")
    process.add_workflow(workflow)

    plan = Samyojana().compose([process])
    assert plan == [workflow]


def test_niyojana_schedule_returns_defensive_copy():
    workflow = Varna(name="Procurement Workflow")
    plan = [workflow]

    scheduled = Niyojana().schedule(plan)
    assert scheduled == plan
    assert scheduled is not plan


def test_vicara_deliberates_via_runtime_reasoner():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    result = Vicara().deliberate(runtime, Sankalpa(text="Create PO for laptops"))
    assert "Procurement-Kurukshetra" in result


def test_nirnaya_decide_rejects_none_deliberation():
    try:
        Nirnaya().decide(None)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_nirnaya_decide_passes_through_deliberation():
    assert Nirnaya().decide("approved") == "approved"


def test_pariksha_rejects_blank_decision():
    try:
        Pariksha().validate("   ")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_pariksha_validate_passes_through_decision():
    assert Pariksha().validate("approved") == "approved"


def test_pariksha_validate_candidates_passes_through_karma_list():
    candidates = [Karma(name="Create Purchase Order")]
    assert Pariksha().validate_candidates(candidates) == candidates


def test_pariksha_validate_candidates_rejects_non_list():
    try:
        Pariksha().validate_candidates("not-a-list")
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


def test_pariksha_validate_candidates_rejects_wrong_item_type():
    try:
        Pariksha().validate_candidates([Varna(name="Procurement Workflow")])
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


def test_pariksha_validate_selection_passes_through_karma_list():
    selected = [Karma(name="Create Purchase Order")]
    assert Pariksha().validate_selection(selected) == selected


def test_pariksha_validate_selection_rejects_wrong_item_type():
    try:
        Pariksha().validate_selection([object()])
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


def test_pariksha_validate_plan_passes_through_varna_list():
    plan = [Varna(name="Procurement Workflow")]
    assert Pariksha().validate_plan(plan) == plan


def test_pariksha_validate_plan_rejects_wrong_item_type():
    try:
        Pariksha().validate_plan([Karma(name="Create Purchase Order")])
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


def test_pariksha_validate_schedule_passes_through_varna_list():
    scheduled = [Varna(name="Procurement Workflow")]
    assert Pariksha().validate_schedule(scheduled) == scheduled


def test_pariksha_validate_schedule_rejects_non_list():
    try:
        Pariksha().validate_schedule(None)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


def test_pariksha_validate_lists_allow_empty():
    assert Pariksha().validate_candidates([]) == []
    assert Pariksha().validate_selection([]) == []
    assert Pariksha().validate_plan([]) == []
    assert Pariksha().validate_schedule([]) == []


def test_kriya_performs_deliberate_decide_validate_chain():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    result = Kriya().perform(runtime, Sankalpa(text="Create PO for laptops"))
    assert "Procurement-Kurukshetra" in result


def test_anushthana_executes_via_kriya():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    result = Anushthana().execute(runtime, Sankalpa(text="Create PO for laptops"))
    assert "Procurement-Kurukshetra" in result


def test_samanvaya_orchestrates_via_anushthana():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    result = Samanvaya().orchestrate(runtime, Sankalpa(text="Create PO for laptops"))
    assert "Procurement-Kurukshetra" in result


def test_smarana_recalls_smriti_context_window():
    smriti = Smriti(capacity=10)
    smriti.record("past request", decision="approved")

    recalled = Smarana().recall(smriti, Sankalpa(text="anything"))
    assert "past request" in recalled


def test_vyakhya_explains_decision_from_pramana():
    explanation = Vyakhya().explain(Pramana(basis="Cleared PO Approval Policy"), "approved")
    assert explanation == "Cleared PO Approval Policy -> approved"


def test_adhyayana_learns_smriti_entry_into_anubhava():
    smriti = Smriti(capacity=10)
    entry = smriti.record("a", decision="approved")
    anubhava = Anubhava()

    Adhyayana().learn(anubhava, entry)
    assert anubhava.observations == 1
    assert anubhava.decision_counts == {"approved": 1}


def test_anukulana_throttles_adaptation_to_every_n_observations():
    bank = SamskaraBank()
    anubhava = Anubhava()
    anukulana = Anukulana(adapt_every=2)
    smriti = Smriti(capacity=10)

    anubhava.observe_entry(smriti.record("a", decision="approved"))
    assert anukulana.adapt(bank, anubhava) is None
    assert bank.impressions == []

    anubhava.observe_entry(smriti.record("b", decision="approved"))
    learned = anukulana.adapt(bank, anubhava)
    assert learned is not None
    assert bank.impressions == [learned]


def test_ksetra_reason_runs_the_full_pipeline_and_records_plan_and_recall():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    runtime.add_policy(Dharma(name="PO Approval Policy", rule="purchase_amount <= approval_limit"))
    process = Karma(name="Create Purchase Order")
    process.add_workflow(Varna(name="Procurement Workflow"))
    runtime.add_process(process)

    runtime.reason(Sankalpa(text="Create Purchase Order for laptops under approved budget"))

    entry = runtime.smriti.working[-1]
    assert entry.evidence.sources["plan"] == ["Procurement Workflow"]
    assert "audited" in entry.evidence.sources
    assert "explanation" in entry.evidence.sources
    assert runtime.anubhava.observations == 1


def test_ksetra_reason_rejects_malformed_anveshana_output_via_pariksha():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    runtime.anveshana.discover = lambda *_args, **_kwargs: [Varna(name="not-a-karma")]

    try:
        runtime.reason(Sankalpa(text="Create PO for laptops"))
    except TypeError as exc:
        assert "Anveshana candidates" in str(exc)
    else:
        raise AssertionError("expected TypeError")


def test_ksetra_reason_adapts_samskara_every_adapt_every_cycles():
    runtime = Ksetra(name="Procurement-Kurukshetra")
    runtime.anukulana = Anukulana(adapt_every=2)

    runtime.reason(Sankalpa(text="first request"))
    assert runtime.samskara.impressions == []

    runtime.reason(Sankalpa(text="second request"))
    assert len(runtime.samskara.impressions) == 1
