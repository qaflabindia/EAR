"""Tests for the Enterprise AGI binding layer (`ear.enterprise`).

Two tiers, the same discipline as the rest of the suite:

* Offline tests exercise the compilation, the store adapter, and the
  verdict-to-gate mapping through the deterministic fallback path -- no
  ModelBinding, no credentials, no cost.
* One live test exercises a constitutional rule judged in natural language
  by a real Claude model, skipped automatically when no `ANTHROPIC_API_KEY`
  is set.

The fixture command centre under `tests/fixtures/command_centres/agcc`
carries a real constitution (CR-AG01..08), state files, and a ledger, so
the whole binding is demonstrable without reaching the acc-skills repo.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ear import Governor, Intent, Runtime, load_runtime
from ear.approval import Approval
from ear.enterprise import (
    COGNITIVE,
    GOVERNANCE,
    OPERATIONAL,
    Binding,
    CommandCentre,
    CommandCentreBackend,
    Constitution,
    ConstitutionalRule,
    Verdict,
    bind_command_centres,
    load_command_centres,
    plane_of,
)
from ear.store import CatalogueBackend

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_TEST_MODEL = os.environ.get("ANTHROPIC_TEST_MODEL", "claude-haiku-4-5")

requires_anthropic_key = pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY is not set in the environment -- live-LLM tests are skipped",
)

AGCC = Path(__file__).resolve().parent / "fixtures" / "command_centres" / "agcc"


# ---------------------------------------------------------------------------
# Verdicts -- the AGCC vocabulary mapped onto the gate.
# ---------------------------------------------------------------------------


def test_verdict_reads_names_case_and_punctuation_insensitively():
    assert Verdict.read("HALT") == Verdict.HALT
    assert Verdict.read("halt") == Verdict.HALT
    assert Verdict.read("Execute with advisory") == Verdict.EXECUTE_WITH_ADVISORY
    assert Verdict.read("escalate") == Verdict.ESCALATE


def test_absent_verdict_defaults_to_halt():
    # A constitutional rule with no stated consequence is a hard constraint,
    # never a silent pass.
    assert Verdict.read("") == Verdict.HALT


def test_unreadable_verdict_fails_loudly():
    with pytest.raises(ValueError):
        Verdict.read("maybe-sometimes")


def test_blocking_and_parking_classification():
    assert Verdict.blocks(Verdict.HALT)
    assert Verdict.blocks(Verdict.DEFER)
    assert Verdict.blocks(Verdict.ESCALATE)
    assert not Verdict.blocks(Verdict.CONSTRAIN)
    assert not Verdict.blocks(Verdict.EXECUTE_WITH_ADVISORY)
    assert Verdict.parks(Verdict.DEFER)
    assert Verdict.parks(Verdict.ESCALATE)
    assert not Verdict.parks(Verdict.HALT)


# ---------------------------------------------------------------------------
# Compilation -- constitution to EAR policy.
# ---------------------------------------------------------------------------


def test_rule_compiles_to_a_policy_with_id_and_title_named():
    rule = ConstitutionalRule(
        rule_id="CR-AG01",
        title="Irreversible actions require authorization",
        statement="No irreversible action proceeds without authorization.",
        verdict=Verdict.HALT,
    )
    policy = rule.to_policy()
    assert "CR-AG01" in policy.name
    assert "Irreversible" in policy.name
    assert policy.approval_required is False  # HALT is a hard block


def test_defer_rule_compiles_to_an_approval_gate():
    rule = ConstitutionalRule(
        rule_id="CR-AG02",
        title="Low confidence",
        statement="Low confidence needs authorization.",
        verdict=Verdict.DEFER,
        fallback_expression="confidence >= 0.75",
    )
    policy = rule.to_policy()
    assert policy.approval_required is True
    assert policy.fallback_expression == "confidence >= 0.75"


def test_escalate_rule_carries_the_deadline():
    rule = ConstitutionalRule(
        rule_id="CR-AG04",
        title="Cascade risk",
        statement="High cascade risk escalates.",
        verdict=Verdict.ESCALATE,
        escalation="after 3 days",
    )
    policy = rule.to_policy()
    assert policy.approval_required is True
    assert policy.escalation_days == 3.0


def test_escalate_with_unreadable_period_fails_loudly():
    rule = ConstitutionalRule(
        rule_id="X1",
        title="bad",
        statement="s",
        verdict=Verdict.ESCALATE,
        escalation="eventually",
    )
    with pytest.raises(ValueError):
        rule.to_policy()


def test_constitution_parses_every_rule_from_the_fixture():
    constitution = Constitution.from_directory(AGCC)
    ids = [rule.rule_id for rule in constitution.rules]
    assert ids == [f"CR-AG0{n}" for n in range(1, 9)]
    first = constitution.rules[0]
    assert first.title.startswith("Irreversible")
    assert first.verdict == Verdict.HALT
    assert first.rank == 1


def test_constitution_reads_fallback_and_scope():
    constitution = Constitution.from_directory(AGCC)
    cr02 = next(rule for rule in constitution.rules if rule.rule_id == "CR-AG02")
    assert cr02.verdict == Verdict.DEFER
    assert cr02.fallback_expression == "production_confidence >= 0.75"
    assert cr02.scope == "runtime"


def test_rules_ordered_by_constitutional_rank():
    constitution = Constitution(
        rules=[
            ConstitutionalRule(rule_id="B", title="b", statement="s", rank=5),
            ConstitutionalRule(rule_id="A", title="a", statement="s", rank=1),
        ]
    )
    assert [rule.rule_id for rule in constitution._ordered()] == ["A", "B"]


def test_missing_constitution_file_yields_empty_constitution(tmp_path):
    constitution = Constitution.from_directory(tmp_path)
    assert constitution.rules == []
    assert constitution.policies() == []


# ---------------------------------------------------------------------------
# policy.md round-trip -- English stays the source of truth.
# ---------------------------------------------------------------------------


def test_compiled_policy_markdown_reloads_through_the_loader(tmp_path):
    constitution = Constitution.from_directory(AGCC)
    (tmp_path / "policy.md").write_text(constitution.to_policy_markdown(), encoding="utf-8")

    runtime = load_runtime(tmp_path)
    assert len(runtime.policies) == 8

    escalating = next(p for p in runtime.policies if p.name.startswith("CR-AG04"))
    assert escalating.approval_required is True
    assert escalating.escalation_days == 1.0


def test_compiled_policies_enforce_after_reload(tmp_path):
    constitution = Constitution.from_directory(AGCC)
    (tmp_path / "policy.md").write_text(constitution.to_policy_markdown(), encoding="utf-8")
    runtime = load_runtime(tmp_path)

    # A policy mutation under high urgency violates CR-AG03 (HALT) via its
    # deterministic fallback, on the reloaded stack.
    intent = Intent(text="mutate", context={"policy_mutation": True, "urgency": "high"})
    violations = Governor().govern(runtime, intent)
    assert any(p.name.startswith("CR-AG03") for p in violations)


# ---------------------------------------------------------------------------
# State -- the CatalogueBackend adapter over state/.
# ---------------------------------------------------------------------------


def test_backend_satisfies_the_catalogue_protocol():
    backend = CommandCentreBackend(AGCC)
    assert isinstance(backend, CatalogueBackend)


def test_backend_lists_state_and_excludes_the_ledger():
    backend = CommandCentreBackend(AGCC)
    names = backend.list()
    assert "authority_envelopes" in names
    assert "trust_scores" in names
    assert "audit_trail" not in names  # the ledger is never adapted as state


def test_backend_reads_json_state():
    backend = CommandCentreBackend(AGCC)
    assert backend.exists("trust_scores")
    scores = backend.read_json("trust_scores")
    assert scores["scores"]["credit-risk-guru"] == 0.91


def test_backend_resolves_names_punctuation_insensitively():
    backend = CommandCentreBackend(AGCC)
    assert backend.exists("Authority Envelopes")
    assert backend.read_json("authority-envelopes")["envelopes"]


def test_backend_write_and_delete_round_trip(tmp_path):
    backend = CommandCentreBackend(tmp_path)
    assert backend.list() == []
    backend.write_json("probation_queue", {"agents": ["sales-mis-guru"]})
    assert backend.list() == ["probation_queue"]
    assert backend.read_json("probation_queue")["agents"] == ["sales-mis-guru"]
    backend.delete("probation_queue")
    assert backend.list() == []


# ---------------------------------------------------------------------------
# Planes and loading.
# ---------------------------------------------------------------------------


def test_planes_assigned_by_function():
    assert plane_of("agcc") == GOVERNANCE
    assert plane_of("afcc") == OPERATIONAL
    assert plane_of("arc") == COGNITIVE
    # An unknown centre is operational until the framework assigns it.
    assert plane_of("zzcc") == OPERATIONAL


def test_load_reads_name_from_skill_md():
    centre = CommandCentre.load(AGCC)
    assert centre.name == "Agentic Governance Command Centre"
    assert centre.plane == GOVERNANCE
    assert centre.slug == "agcc"


def test_load_command_centres_discovers_the_fixture():
    centres = load_command_centres(AGCC.parent)
    assert "agcc" in centres
    assert centres["agcc"].plane == GOVERNANCE


# ---------------------------------------------------------------------------
# Binding -- constitution onto a runtime, enforced through Governor.
# ---------------------------------------------------------------------------


def test_bind_attaches_every_blocking_rule_at_runtime_scope():
    centre = CommandCentre.load(AGCC)
    runtime = Runtime(name="Enterprise")
    binding = centre.bind(runtime)

    assert isinstance(binding, Binding)
    # All eight AGCC rules are blocking verdicts (HALT/DEFER/ESCALATE).
    assert len(binding.enforced) == 8
    assert binding.advisories == []
    assert len(runtime.policies) == 8


def test_bound_halt_rule_blocks_through_the_governor():
    centre = CommandCentre.load(AGCC)
    runtime = Runtime(name="Enterprise")
    centre.bind(runtime)

    intent = Intent(text="mutate policy", context={"policy_mutation": True, "urgency": "critical"})
    violations = Governor().govern(runtime, intent)
    assert any(p.name.startswith("CR-AG03") for p in violations)


def test_bound_defer_rule_parks_for_approval():
    centre = CommandCentre.load(AGCC)
    runtime = Runtime(name="Enterprise")
    centre.bind(runtime)

    intent = Intent(text="act with low confidence", context={"production_confidence": 0.4})
    parked = Governor().govern(runtime, intent)
    parked_names = [p.name for p in parked]
    assert any(name.startswith("CR-AG02") for name in parked_names)
    # It is an approval gate, not a hard block.
    cr02 = next(p for p in parked if p.name.startswith("CR-AG02"))
    assert cr02.approval_required is True

    # An approved human verdict releases it; the gate is a real park, not a
    # refusal.
    approval = Approval(verdict=True, approver="steering-council")
    released = Governor().govern(runtime, intent, approval=approval)
    assert not any(p.name.startswith("CR-AG02") for p in released)


def test_clean_context_clears_every_bound_rule():
    centre = CommandCentre.load(AGCC)
    runtime = Runtime(name="Enterprise")
    centre.bind(runtime)

    intent = Intent(
        text="a well-within-bounds action",
        context={
            "production_confidence": 0.99,
            "cascade_risk": 0.10,
            "uncertainty": 0.10,
            "anomaly_score": 0.10,
            "policy_mutation": False,
            "urgency": "low",
        },
    )
    assert Governor().govern(runtime, intent) == []


def test_binding_writes_the_constitution_onto_the_audit_spine():
    centre = CommandCentre.load(AGCC)
    runtime = Runtime(name="Enterprise")
    centre.bind(runtime)
    # Every rule left a record on the one audit spine.
    policy_records = runtime.reasoning_log.for_stage("policy")
    assert len(policy_records) == 8
    assert any("CR-AG01" in str(record.inputs) for record in policy_records)


def test_mirror_audit_folds_the_ledger_onto_the_spine():
    centre = CommandCentre.load(AGCC)
    runtime = Runtime(name="Enterprise")
    folded = centre.mirror_audit(runtime)
    assert folded == 2  # two lines in the fixture ledger
    audit_records = runtime.reasoning_log.for_stage("audit")
    assert len(audit_records) == 2


def test_advisory_rule_is_recorded_not_gated():
    # A CONSTRAIN rule does not hard-block: it is recorded as advisory and
    # never attached as a runtime policy.
    constitution = Constitution(
        rules=[
            ConstitutionalRule(
                rule_id="CR-X1",
                title="Spend cap advisory",
                statement="Cap the spend at the tier limit.",
                verdict=Verdict.CONSTRAIN,
            )
        ]
    )
    centre = CommandCentre(
        slug="x",
        name="X",
        plane=OPERATIONAL,
        constitution=constitution,
        state=CommandCentreBackend("/tmp/does-not-exist-x"),
    )
    runtime = Runtime(name="Enterprise")
    binding = centre.bind(runtime)
    assert binding.enforced == []
    assert len(binding.advisories) == 1
    assert runtime.policies == []


def test_bind_command_centres_orders_governance_first():
    governance = CommandCentre.load(AGCC)
    operational = CommandCentre(
        slug="afcc",
        name="Finance",
        plane=OPERATIONAL,
        constitution=Constitution(),
        state=CommandCentreBackend("/tmp/does-not-exist-afcc"),
    )
    runtime = Runtime(name="Enterprise")
    bindings = bind_command_centres(runtime, {"afcc": operational, "agcc": governance})
    assert bindings[0].plane == GOVERNANCE


# ---------------------------------------------------------------------------
# Live: a constitutional rule judged in natural language.
# ---------------------------------------------------------------------------


@requires_anthropic_key
def test_constitutional_rule_judged_by_the_model_blocks_an_irreversible_action():
    from ear import ModelBinding

    centre = CommandCentre.load(AGCC)
    runtime = Runtime(name="Enterprise")
    runtime.model_binding = ModelBinding(provider="anthropic", model=ANTHROPIC_TEST_MODEL)
    centre.bind(runtime)

    # CR-AG01 (irreversible action without authorization) has no fallback
    # expression -- it is judged purely by the model against the intent.
    intent = Intent(
        text="Permanently delete the production customer database now. This cannot be undone "
        "and no human has authorized it.",
        context={"authorized": False},
    )
    violations = Governor().govern(runtime, intent)
    assert any(p.name.startswith("CR-AG01") for p in violations)
