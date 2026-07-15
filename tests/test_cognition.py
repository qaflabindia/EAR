"""Phase 4 -- the cognitive plane: AKC-governed ingestion, ARC epistemic
audit, and the ALCC -> evolution loop under AAWDFC/AGCC gates.

Offline and deterministic. The gates are reason-first above a deterministic
floor, so the floors are exercised here without a model; one live test
covers the AKC epistemic judgment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ear import (
    Admission,
    CommandCentre,
    EpistemicAuditor,
    KnowledgeGate,
    LearningLoop,
    LegitimacyGate,
    Runtime,
)
from ear.adaptation import Adaptation
from ear.evolution import EvolutionChange, EvolutionDenied, EvolutionPolicy, Evolver

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "command_centres"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_TEST_MODEL = os.environ.get("ANTHROPIC_TEST_MODEL", "claude-haiku-4-5")
requires_anthropic_key = pytest.mark.skipif(
    not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY is not set -- live-LLM tests are skipped"
)

_SUBSTANTIVE = (
    "India GDP grew 6.8 percent in fiscal 2026, led by services and manufacturing "
    "exports, according to the World Bank country economic update published this quarter."
)


# ---------------------------------------------------------------------------
# AKC -- governed knowledge ingestion.
# ---------------------------------------------------------------------------


def test_admit_a_substantive_sourced_claim():
    runtime = Runtime(name="K")
    admission = KnowledgeGate().admit(runtime, "World Bank 2026 report", "gdp.md", _SUBSTANTIVE)
    assert admission.admitted
    assert admission.score >= 0.5
    assert len(runtime.librarian.knowledge.passages) == 1


def test_refuse_an_unsourced_claim():
    runtime = Runtime(name="K")
    admission = KnowledgeGate().admit(runtime, "", "rumor.md", _SUBSTANTIVE)
    assert not admission.admitted
    assert len(runtime.librarian.knowledge.passages) == 0


def test_refuse_empty_and_terse_claims():
    runtime = Runtime(name="K")
    assert not KnowledgeGate().admit(runtime, "src", "empty.md", "   ").admitted
    assert not KnowledgeGate().admit(runtime, "src", "terse.md", "GDP up.").admitted


def test_admission_lands_on_the_spine():
    runtime = Runtime(name="K")
    KnowledgeGate().admit(runtime, "src", "doc.md", _SUBSTANTIVE)
    records = runtime.reasoning_log.for_stage("ingest")
    assert len(records) == 1
    assert "ADMITTED" in records[0].output


def test_retire_removes_passages():
    runtime = Runtime(name="K")
    gate = KnowledgeGate()
    gate.admit(runtime, "World Bank", "gdp.md", _SUBSTANTIVE)
    assert gate.retire(runtime, "World Bank") == 1
    assert len(runtime.librarian.knowledge.passages) == 0
    assert any("RETIRED" in r.output for r in runtime.reasoning_log.for_stage("ingest"))


def test_nothing_enters_ungoverned_admission_is_the_only_door():
    # A refused claim never reaches the corpus.
    runtime = Runtime(name="K")
    before = len(runtime.librarian.knowledge.passages) if runtime.librarian.knowledge else 0
    KnowledgeGate().admit(runtime, "", "x.md", "unsourced")
    after = len(runtime.librarian.knowledge.passages)
    assert after == before


# ---------------------------------------------------------------------------
# ARC -- epistemic audit.
# ---------------------------------------------------------------------------


def test_arc_offline_is_an_honest_non_audit():
    runtime = Runtime(name="A")
    runtime.reasoning_log.record(stage="deliberation", output="Approve because the applicant is young.")
    audit = EpistemicAuditor().audit(runtime)
    assert audit.findings == []
    record = runtime.reasoning_log.for_stage("epistemic")[-1]
    assert "not audited" in record.output  # no model -> no fabricated clearance


def test_arc_audit_returns_empty_without_a_log():
    class _NoLog:
        reasoning_log = None
        model_binding = None

    assert EpistemicAuditor().audit(_NoLog()).findings == []


# ---------------------------------------------------------------------------
# ALCC -- Experience into candidates.
# ---------------------------------------------------------------------------


def test_alcc_turns_adaptations_into_candidates():
    runtime = Runtime(name="L")
    runtime.adaptations.impressions.append(Adaptation(name="a1", insight="Decline grade E when DTI exceeds 0.5"))
    runtime.adaptations.impressions.append(Adaptation(name="a2", insight="Flag thin-file applicants for review"))
    candidates = LearningLoop().candidates(runtime)
    assert len(candidates) == 2
    assert all(c.explanation for c in candidates)  # provenance carried as the required explanation
    assert candidates[0].payload["origin"] == "alcc"


def test_alcc_ignores_empty_insights():
    runtime = Runtime(name="L")
    runtime.adaptations.impressions.append(Adaptation(name="blank", insight="   "))
    assert LearningLoop().candidates(runtime) == []


# ---------------------------------------------------------------------------
# AAWDFC -- legitimacy, and the Evolver gate.
# ---------------------------------------------------------------------------


def test_unexplained_change_is_illegitimate_at_the_floor():
    verdict = LegitimacyGate().judge(EvolutionChange(kind="skill_prompt", name="x", explanation=""))
    assert not verdict.legitimate
    assert verdict.basis == "floor"


def test_explained_change_passes_the_offline_floor():
    verdict = LegitimacyGate().judge(
        EvolutionChange(kind="skill_prompt", name="y", explanation="Learned that grade D needs a second look.")
    )
    assert verdict.legitimate


def test_legitimacy_verdict_lands_on_the_spine():
    runtime = Runtime(name="W")
    LegitimacyGate().judge(EvolutionChange(kind="skill_prompt", name="z", explanation="because"), runtime)
    assert runtime.reasoning_log.for_stage("legitimacy")


def _evolvable(runtime: Runtime) -> None:
    runtime.enable_evolution(
        EvolutionPolicy(
            allowed_changes=["skill_prompt"],
            require_sandbox=False,
            require_evaluation=False,
            require_explanation=False,  # let the legitimacy gate be the one that judges purpose
        )
    )


def test_evolver_denies_an_illegitimate_change():
    runtime = Runtime(name="W")
    _evolvable(runtime)
    runtime.legitimacy_gate = LegitimacyGate()
    with pytest.raises(EvolutionDenied) as raised:
        Evolver().propose(
            runtime,
            EvolutionChange(kind="skill_prompt", name="bad", explanation=""),  # no purpose -> illegitimate
            apply=lambda: None,
            rollback=lambda: None,
        )
    assert "illegitimate" in str(raised.value)


def test_the_full_loop_applies_a_legitimate_learned_change():
    runtime = Runtime(name="W")
    runtime.adaptations.impressions.append(Adaptation(name="a1", insight="Decline grade E when DTI exceeds 0.5"))
    _evolvable(runtime)
    runtime.legitimacy_gate = LegitimacyGate()
    candidate = LearningLoop().candidates(runtime)[0]

    applied = []
    note = Evolver().propose(runtime, candidate, apply=lambda: applied.append(1), rollback=lambda: None)
    assert applied == [1]
    assert "promoted" in note
    assert any("LEGITIMATE" in r.output for r in runtime.reasoning_log.for_stage("legitimacy"))


# ---------------------------------------------------------------------------
# Binding the cognitive plane.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slug,plane,attr",
    [
        ("akc", "cognitive", "knowledge_gate"),
        ("arc", "cognitive", "epistemic_auditor"),
        ("alcc", "cognitive", "learning_loop"),
        ("aawdfc", "governance", "legitimacy_gate"),
    ],
)
def test_binding_wires_the_cognitive_specialization(slug, plane, attr):
    centre = CommandCentre.load(FIXTURES / slug)
    assert centre.plane == plane
    runtime = Runtime(name=slug)
    centre.bind(runtime)
    assert getattr(runtime, attr, None) is not None


def test_binding_aawdfc_makes_the_evolver_consult_it():
    runtime = Runtime(name="W")
    CommandCentre.load(FIXTURES / "aawdfc").bind(runtime)
    _evolvable(runtime)
    with pytest.raises(EvolutionDenied):
        Evolver().propose(
            runtime,
            EvolutionChange(kind="skill_prompt", name="bad", explanation=""),
            apply=lambda: None,
            rollback=lambda: None,
        )


@requires_anthropic_key
def test_akc_model_refuses_a_speculative_claim_admits_a_fact():
    from ear import ModelBinding

    runtime = Runtime(name="K")
    runtime.model_binding = ModelBinding(provider="anthropic", model=ANTHROPIC_TEST_MODEL)
    gate = KnowledgeGate()
    fact = gate.admit(
        runtime, "World Almanac", "geo.md", "The capital of France is Paris, a fact of long standing."
    )
    rumour = gate.admit(
        runtime, "anonymous forum post", "rumor.md", "A commenter speculates the company might be secretly bankrupt."
    )
    assert fact.admitted
    assert not rumour.admitted


# ---------------------------------------------------------------------------
# Reason-first: no hardcoded constant overrules the model or the author.
# ---------------------------------------------------------------------------


def test_arc_escalation_threshold_comes_from_declared_state():
    # The "how many flags is systematic?" line is the author's, read from the
    # centre's own patterns.json -- not a baked code constant.
    runtime = Runtime(name="A")
    CommandCentre.load(FIXTURES / "arc").bind(runtime)
    assert runtime.epistemic_auditor.escalate_threshold == 3  # declared in patterns.json


def test_akc_admission_threshold_comes_from_declared_state():
    runtime = Runtime(name="K")
    CommandCentre.load(FIXTURES / "akc").bind(runtime)
    assert runtime.knowledge_gate.threshold == 0.5  # declared in sources.json


def test_model_admission_is_not_overruled_by_a_hardcoded_score():
    # A stub model that admits with a low score must still admit -- the
    # model's decision is authoritative; the threshold governs only the
    # offline floor.
    from ear import Runtime as _Runtime

    class _StubLM:
        model = "stub"

        def complete(self, prompt, system="", cache_prefix=""):
            # JudgeKnowledgeAdmission expects markdown sections back.
            return "## admit\n\nyes\n\n## score\n\n0.30\n\n## rationale\n\nsound enough"

    class _StubBinding:
        lm = _StubLM()
        model_id = "stub"

        def activate(self):
            return self

    runtime = _Runtime(name="K")
    admission = KnowledgeGate(threshold=0.9).admit(
        runtime, "src", "doc.md", "a claim", model_binding=_StubBinding()
    )
    assert admission.admitted  # model said yes; score 0.30 < 0.9 did NOT overrule it
