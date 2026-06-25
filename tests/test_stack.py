from ear import Dharma, Guna, Karma, Ksetra, Sankalpa, Varna, Vidya


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
