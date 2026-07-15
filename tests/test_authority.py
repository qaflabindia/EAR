"""Tests for Phase 3's AECC capability-envelope enforcement
(`ear/authority.py`).

All offline: envelope authorization is deterministic, so the whole gate is
exercised without a model. Transitions that persist to a centre's state run
against a *copy* of the fixture, never the committed one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ear import (
    CapabilityEnvelope,
    CommandCentre,
    EnvelopePolicy,
    EnvelopeRegistry,
    Governor,
    Intent,
    Runtime,
    enforce_envelopes,
)
from ear.authority import ACTIVE, PROBATION, REVOKED, SUSPENDED
from ear.enterprise import CommandCentreBackend

AECC = Path(__file__).resolve().parent / "fixtures" / "command_centres" / "aecc"


def _aecc_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "aecc"
    shutil.copytree(AECC, destination)
    return destination


# ---------------------------------------------------------------------------
# The envelope itself.
# ---------------------------------------------------------------------------


def test_active_envelope_authorizes_in_scope_and_tier():
    envelope = CapabilityEnvelope(
        agent="guru", certified=True, scopes=["reason:credit"], max_autonomy_tier=2, status=ACTIVE
    )
    ok, _reason = envelope.authorized(scope="reason:credit", tier=2)
    assert ok


def test_out_of_scope_is_refused():
    envelope = CapabilityEnvelope(
        agent="guru", certified=True, scopes=["reason:credit"], max_autonomy_tier=2, status=ACTIVE
    )
    ok, reason = envelope.authorized(scope="delete:database")
    assert not ok
    assert "scope" in reason


def test_over_tier_is_refused():
    envelope = CapabilityEnvelope(
        agent="guru", certified=True, scopes=["x"], max_autonomy_tier=1, status=ACTIVE
    )
    ok, reason = envelope.authorized(scope="x", tier=3)
    assert not ok
    assert "tier" in reason


def test_uncertified_and_revoked_and_suspended_all_refuse():
    assert not CapabilityEnvelope(agent="a", certified=False).authorized()[0]
    assert not CapabilityEnvelope(agent="a", certified=True, status=REVOKED).authorized()[0]
    assert not CapabilityEnvelope(agent="a", certified=True, status=SUSPENDED).authorized()[0]


def test_probation_authorizes_but_flags():
    envelope = CapabilityEnvelope(agent="a", certified=True, scopes=["x"], status=PROBATION)
    ok, reason = envelope.authorized(scope="x")
    assert ok
    assert "probation" in reason


def test_signature_detects_tampering():
    envelope = CapabilityEnvelope(
        agent="a", certified=True, scopes=["x"], max_autonomy_tier=1, status=ACTIVE
    ).sign()
    assert envelope.signature_valid()
    # Flip the standing back to active by hand without re-signing: the
    # signature no longer verifies, and authorization refuses.
    envelope.status = SUSPENDED
    assert not envelope.signature_valid()


def test_tampered_active_envelope_is_refused_by_signature():
    signed = CapabilityEnvelope(
        agent="a", certified=True, scopes=["x"], max_autonomy_tier=1, status=REVOKED
    ).sign()
    # Someone edits the stored status to active but cannot forge the signature.
    signed.status = ACTIVE
    ok, reason = signed.authorized(scope="x")
    assert not ok
    assert "tampered" in reason


# ---------------------------------------------------------------------------
# The registry.
# ---------------------------------------------------------------------------


def test_registry_loads_from_a_centre_backend():
    registry = EnvelopeRegistry.from_backend(CommandCentreBackend(AECC))
    assert set(registry.envelopes) == {"credit risk guru", "sales mis guru", "rogue agent"}
    ok, _ = registry.authorized("credit-risk-guru", scope="reason:credit", tier=2)
    assert ok
    assert not registry.authorized("rogue-agent")[0]
    assert not registry.authorized("nobody")[0]


def test_certify_then_revoke_round_trips_and_persists(tmp_path):
    centre = _aecc_copy(tmp_path)
    registry = EnvelopeRegistry.from_backend(CommandCentreBackend(centre))
    registry.certify("new-agent", scopes=["read:x"], max_autonomy_tier=1)
    assert registry.authorized("new-agent", scope="read:x")[0]

    # Persisted to the copy's state, readable back by a fresh registry.
    reloaded = EnvelopeRegistry.from_backend(CommandCentreBackend(centre))
    assert reloaded.get("new-agent") is not None

    registry.revoke("new-agent", reason="test")
    assert not registry.authorized("new-agent")[0]


def test_signed_envelope_survives_persistence(tmp_path):
    centre = _aecc_copy(tmp_path)
    registry = EnvelopeRegistry.from_backend(CommandCentreBackend(centre))
    registry.certify("signed-agent", scopes=["read:x"], max_autonomy_tier=1)
    reloaded = EnvelopeRegistry.from_backend(CommandCentreBackend(centre))
    envelope = reloaded.get("signed-agent")
    assert envelope.signature_valid()


# ---------------------------------------------------------------------------
# Enforcement through the Governor -- the single choke point.
# ---------------------------------------------------------------------------


def test_human_intent_without_an_agent_is_not_applicable():
    registry = EnvelopeRegistry.from_backend(CommandCentreBackend(AECC))
    runtime = Runtime(name="Ent")
    enforce_envelopes(runtime, registry)
    assert Governor().govern(runtime, Intent(text="a human request", context={})) == []


def test_certified_agent_passes_and_uncertified_blocks():
    registry = EnvelopeRegistry.from_backend(CommandCentreBackend(AECC))
    runtime = Runtime(name="Ent")
    policy = enforce_envelopes(runtime, registry)

    passing = Intent(text="act", context={"agent": "credit-risk-guru", "scope": "reason:credit"})
    assert Governor().govern(runtime, passing) == []

    blocked = Intent(text="act", context={"agent": "ghost"})
    assert policy in Governor().govern(runtime, blocked)


def test_revocation_is_immediate_on_the_next_cycle(tmp_path):
    centre = _aecc_copy(tmp_path)
    registry = EnvelopeRegistry.from_backend(CommandCentreBackend(centre))
    runtime = Runtime(name="Ent")
    enforce_envelopes(runtime, registry)

    intent = Intent(text="act", context={"agent": "credit-risk-guru", "scope": "reason:credit"})
    assert Governor().govern(runtime, intent) == []

    registry.revoke("credit-risk-guru", reason="anomaly")
    # No reload, no re-attach: the very next cycle fails the gate.
    assert Governor().govern(runtime, intent) != []


def test_out_of_scope_agent_blocks_through_the_gate():
    registry = EnvelopeRegistry.from_backend(CommandCentreBackend(AECC))
    runtime = Runtime(name="Ent")
    enforce_envelopes(runtime, registry)
    over = Intent(text="delete", context={"agent": "sales-mis-guru", "scope": "delete:database"})
    assert Governor().govern(runtime, over) != []


def test_envelope_policy_is_a_policy_subclass():
    # So it attaches and is judged exactly like any other runtime policy.
    from ear import Policy

    registry = EnvelopeRegistry.from_backend(CommandCentreBackend(AECC))
    runtime = Runtime(name="Ent")
    policy = enforce_envelopes(runtime, registry)
    assert isinstance(policy, EnvelopePolicy)
    assert isinstance(policy, Policy)
    assert policy in runtime.policies


def test_binding_aecc_attaches_envelope_enforcement():
    centre = CommandCentre.load(AECC)
    runtime = Runtime(name="Ent")
    binding = centre.bind(runtime)
    assert binding.envelope_policy is not None
    assert binding.envelope_registry is not None
    # An agent-initiated intent for a revoked agent is blocked on the bound
    # runtime -- constitution and envelope both attached.
    blocked = Governor().govern(runtime, Intent(text="act", context={"agent": "rogue-agent"}))
    assert any(isinstance(p, EnvelopePolicy) for p in blocked)


def test_transition_records_on_the_audit_spine(tmp_path):
    centre = _aecc_copy(tmp_path)
    runtime = Runtime(name="Ent")
    registry = EnvelopeRegistry.from_backend(
        CommandCentreBackend(centre), reasoning_log=runtime.reasoning_log
    )
    registry.suspend("sales-mis-guru", reason="under review")
    certification_records = runtime.reasoning_log.for_stage("certification")
    assert len(certification_records) == 1
    assert "sales-mis-guru" in str(certification_records[0].inputs)
