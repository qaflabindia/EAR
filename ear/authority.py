"""Authority -- AECC capability envelopes: the authority model for every
non-human actor in the enterprise.

Phase 1 bound a centre's constitution; Phase 2 compiled a whole centre.
Phase 3 answers *who may act at all*. An agent -- a spawned persona, an
MCP-attached command centre, an evolved workflow -- must hold a **certified
capability envelope** before `Governor.govern` will pass its intents
(framework architecture §4). Certification, trust scoring, probation,
suspension and revocation are AECC state transitions; enforcement is an
ordinary EAR runtime-scope `Policy` whose judgment consults the live
envelope registry. Revocation is therefore *immediate*: the registry is
updated, and the very next `reason()` call fails the gate.

The pieces:

* `CapabilityEnvelope` -- one non-human actor's authority record: whether it
  is certified, the capability scopes it holds, its maximum autonomy tier,
  its standing (active / probation / suspended / revoked), a trust score,
  and a content signature that makes tampering with the stored record
  detectable (`identity` + `signatures`, the architecture's backing).
* `EnvelopeRegistry` -- the envelopes for a command centre, loaded from and
  persisted to its `state/` through the Phase-1 `CommandCentreBackend`. The
  state transitions live here; every one is recorded on the runtime's one
  audit spine and written back to state.
* `EnvelopePolicy` -- the runtime-scope `Policy` that gates a cycle on the
  acting agent's envelope. A human-initiated intent carries no agent and is
  not applicable (the "off unless declared" posture, like `Claim` and
  `Tenant`); an agent-initiated one blocks unless the agent holds an active,
  in-scope, in-tier envelope.

Standard library only. The signature uses `hashlib`; a signing secret, when
declared, is named by environment variable only -- never a literal.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .policy import Policy
from .section import normalize

# Standing -- an envelope's current authority state.
ACTIVE = "active"
PROBATION = "probation"
SUSPENDED = "suspended"
REVOKED = "revoked"

# The context keys an intent may name its acting agent under.
_ACTOR_KEYS = ("agent", "actor", "agent_id", "acting_agent")
_SCOPE_KEYS = ("scope", "capability", "required_scope")
_TIER_KEYS = ("autonomy_tier", "tier", "required_tier")


def _first(context: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in context and context[key] not in (None, ""):
            return context[key]
    return None


@dataclass
class CapabilityEnvelope:
    """One non-human actor's authority record."""

    agent: str
    certified: bool = False
    scopes: list[str] = field(default_factory=list)
    max_autonomy_tier: int = 0
    status: str = REVOKED
    trust_score: float = 0.0
    issued_at: str = ""
    signature: str = ""

    def is_active(self) -> bool:
        return self.certified and normalize(self.status) == ACTIVE

    def holds_scope(self, scope: Any) -> bool:
        """Whether this envelope grants `scope`. No scope asked for is always
        granted; scopes match case- and punctuation-insensitively like every
        other cross-reference in the stack."""
        if scope in (None, ""):
            return True
        wanted = normalize(str(scope))
        return any(normalize(held) == wanted for held in self.scopes)

    def within_tier(self, tier: Any) -> bool:
        if tier in (None, ""):
            return True
        try:
            return int(tier) <= int(self.max_autonomy_tier)
        except (TypeError, ValueError):
            return False

    def authority_fields(self) -> dict[str, Any]:
        """The fields the signature covers -- the authority-bearing ones, so
        any tampering with certification, scope, tier or standing breaks the
        signature."""
        return {
            "agent": self.agent,
            "certified": self.certified,
            "scopes": sorted(normalize(s) for s in self.scopes),
            "max_autonomy_tier": self.max_autonomy_tier,
            "status": normalize(self.status),
            "issued_at": self.issued_at,
        }

    def compute_signature(self, secret: str = "") -> str:
        """A content signature over the authority fields. With a secret
        (an HMAC-style keyed hash) it is unforgeable without the key; without
        one it is still a tamper-evidence checksum. `hashlib` only."""
        payload = json.dumps(self.authority_fields(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(f"{secret}\n{payload}".encode("utf-8")).hexdigest()

    def sign(self, secret: str = "") -> "CapabilityEnvelope":
        self.signature = self.compute_signature(secret)
        return self

    def signature_valid(self, secret: str = "") -> bool:
        """Whether the stored signature matches the current authority fields.
        A record edited on disk (a status flipped back to active by hand)
        no longer verifies."""
        return bool(self.signature) and self.signature == self.compute_signature(secret)

    def floor(self, secret: str = "") -> tuple[bool, str]:
        """The absolute, non-waivable authority floor: whether this envelope
        exists as a certified, un-withdrawn, untampered credential at all.
        This is the part no reasoning may override -- a revoked envelope
        authorizes nothing, and the model is never asked to reconsider that,
        exactly as a human's approval waiver is never the model's to give.
        Scope and tier (the nuanced part) are judged above this floor."""
        if not self.certified:
            return False, f"agent '{self.agent}' holds no certified envelope"
        standing = normalize(self.status)
        if standing == REVOKED:
            return False, f"agent '{self.agent}' envelope is revoked"
        if standing == SUSPENDED:
            return False, f"agent '{self.agent}' envelope is suspended"
        if self.signature and not self.signature_valid(secret):
            return False, f"agent '{self.agent}' envelope signature does not verify -- record tampered"
        return True, f"agent '{self.agent}' holds a live envelope (standing: {standing})"

    def authorized(self, scope: Any = None, tier: Any = None, secret: str = "") -> tuple[bool, str]:
        """The full deterministic decision: the floor, then scope and tier.
        Used as the offline fallback and by out-of-process evaluation; when a
        model is bound, `EnvelopePolicy` judges scope/tier above the floor
        instead of calling this."""
        floor_ok, floor_reason = self.floor(secret)
        if not floor_ok:
            return False, floor_reason
        if not self.holds_scope(scope):
            return False, f"agent '{self.agent}' envelope does not hold scope '{scope}'"
        if not self.within_tier(tier):
            return False, f"agent '{self.agent}' envelope tier {self.max_autonomy_tier} is below the required tier {tier}"
        standing = normalize(self.status)
        # An active envelope authorizes outright; a probationary one is
        # authorized but flagged, so the ATC adversarial pass can pick it up.
        if standing == PROBATION:
            return True, f"agent '{self.agent}' is on probation -- authorized but flagged for review"
        return True, f"agent '{self.agent}' holds an active, in-scope envelope"

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "certified": self.certified,
            "scopes": list(self.scopes),
            "max_autonomy_tier": self.max_autonomy_tier,
            "status": self.status,
            "trust_score": self.trust_score,
            "issued_at": self.issued_at,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityEnvelope":
        return cls(
            agent=str(data.get("agent") or data.get("name") or ""),
            certified=bool(data.get("certified", False)),
            scopes=list(data.get("scopes") or []),
            max_autonomy_tier=int(data.get("max_autonomy_tier") or 0),
            status=str(data.get("status") or (ACTIVE if data.get("certified") else REVOKED)),
            trust_score=float(data.get("trust_score") or 0.0),
            issued_at=str(data.get("issued_at") or ""),
            signature=str(data.get("signature") or ""),
        )


@dataclass
class EnvelopeRegistry:
    """The capability envelopes for one command centre -- the authority
    model every gate consults. Loaded from and persisted to the centre's
    `state/authority_envelopes.json` through the Phase-1 backend; state
    transitions are recorded on the runtime's one audit spine."""

    envelopes: dict[str, CapabilityEnvelope] = field(default_factory=dict)
    backend: Optional[Any] = None
    state_name: str = "authority_envelopes"
    secret_env_var: str = ""
    reasoning_log: Optional[Any] = None

    @property
    def secret(self) -> str:
        """The signing secret, read from its environment variable by name
        only -- never stored. Absent, signatures are unkeyed checksums."""
        return os.environ.get(self.secret_env_var, "") if self.secret_env_var else ""

    @classmethod
    def from_backend(
        cls,
        backend: Any,
        state_name: str = "authority_envelopes",
        secret_env_var: str = "",
        reasoning_log: Optional[Any] = None,
    ) -> "EnvelopeRegistry":
        registry = cls(
            backend=backend,
            state_name=state_name,
            secret_env_var=secret_env_var,
            reasoning_log=reasoning_log,
        )
        if backend is not None and backend.exists(state_name):
            data = json.loads(backend.read(state_name))
            rows = data.get("envelopes", data) if isinstance(data, dict) else data
            for row in rows or []:
                envelope = CapabilityEnvelope.from_dict(row)
                if envelope.agent:
                    registry.envelopes[normalize(envelope.agent)] = envelope
        return registry

    @classmethod
    def from_centre(cls, centre: Any, reasoning_log: Optional[Any] = None) -> "EnvelopeRegistry":
        """Build a registry from a Phase-1 `CommandCentre`'s state."""
        return cls.from_backend(centre.state, reasoning_log=reasoning_log)

    def get(self, agent: str) -> Optional[CapabilityEnvelope]:
        return self.envelopes.get(normalize(agent))

    def authorized(self, agent: str, scope: Any = None, tier: Any = None) -> tuple[bool, str]:
        """Whether `agent` may act, consulting the *live* registry -- so a
        revocation between two cycles is enforced on the next one. The full
        deterministic decision (floor + scope + tier); the reason-first gate
        judges scope/tier above `floor` instead."""
        envelope = self.get(agent)
        if envelope is None:
            return False, f"agent '{agent}' holds no envelope -- uncertified actors may not act"
        return envelope.authorized(scope=scope, tier=tier, secret=self.secret)

    def floor(self, agent: str) -> tuple[bool, str]:
        """The absolute authority floor for `agent`, live from the registry:
        does a certified, un-withdrawn, untampered envelope exist at all.
        Never model-waivable -- this is what makes revocation immediate."""
        envelope = self.get(agent)
        if envelope is None:
            return False, f"agent '{agent}' holds no envelope -- uncertified actors may not act"
        return envelope.floor(self.secret)

    # -- state transitions --------------------------------------------------

    def certify(
        self,
        agent: str,
        scopes: Optional[list[str]] = None,
        max_autonomy_tier: int = 1,
        trust_score: float = 0.5,
        issued_at: str = "",
    ) -> CapabilityEnvelope:
        """Certify an agent -- issue (or re-issue) an active, signed
        envelope. The one transition that grants authority; everything else
        narrows or removes it."""
        envelope = CapabilityEnvelope(
            agent=agent,
            certified=True,
            scopes=list(scopes or []),
            max_autonomy_tier=int(max_autonomy_tier),
            status=ACTIVE,
            trust_score=float(trust_score),
            issued_at=issued_at,
        ).sign(self.secret)
        self.envelopes[normalize(agent)] = envelope
        self._transition(agent, ACTIVE, "certified")
        return envelope

    def set_trust(self, agent: str, trust_score: float) -> None:
        envelope = self._require(agent)
        envelope.trust_score = float(trust_score)
        envelope.sign(self.secret)
        self._transition(agent, envelope.status, f"trust set to {trust_score}")

    def probation(self, agent: str, reason: str = "") -> None:
        self._set_status(agent, PROBATION, reason or "placed on probation")

    def suspend(self, agent: str, reason: str = "") -> None:
        self._set_status(agent, SUSPENDED, reason or "suspended")

    def revoke(self, agent: str, reason: str = "") -> None:
        self._set_status(agent, REVOKED, reason or "revoked")

    def reinstate(self, agent: str, reason: str = "") -> None:
        self._set_status(agent, ACTIVE, reason or "reinstated")

    def _set_status(self, agent: str, status: str, reason: str) -> None:
        envelope = self._require(agent)
        envelope.status = status
        # Re-sign so the signature always covers the current standing: a
        # revoked envelope whose signature still says active would be a hole.
        envelope.sign(self.secret)
        self._transition(agent, status, reason)

    def _require(self, agent: str) -> CapabilityEnvelope:
        envelope = self.get(agent)
        if envelope is None:
            raise KeyError(f"no envelope for agent '{agent}' -- certify it first")
        return envelope

    def _transition(self, agent: str, status: str, reason: str) -> None:
        self.persist()
        if self.reasoning_log is not None:
            self.reasoning_log.record(
                stage="certification",
                inputs={"agent": agent, "status": status},
                output=f"{agent} -> {status}",
                rationale=reason,
            )

    def persist(self) -> None:
        """Write the registry back to the centre's state, so a transition
        survives the process -- the JSON stays the source of truth in Phase
        1's adapter model."""
        if self.backend is None:
            return
        payload = {"envelopes": [envelope.to_dict() for envelope in self.envelopes.values()]}
        self.backend.write(self.state_name, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


@dataclass
class EnvelopePolicy(Policy):
    """The runtime-scope policy that gates a cycle on the acting agent's
    capability envelope. A `Policy` subclass, so it attaches with
    `runtime.add_policy` and is judged by `Governor.govern` exactly like any
    other policy.

    Reason-first, above a deterministic floor -- the same division EAR draws
    everywhere:

    * The **floor** (uncertified / revoked / suspended / tampered / no
      envelope at all) is absolute and never model-waivable -- consulted
      live from the registry, which is what makes revocation immediate. The
      model is never asked to reconsider a withdrawn credential, exactly as
      it is never asked to waive a human approval.
    * **Above the floor**, whether the envelope's granted scopes and tier
      authorize the *requested* scope and tier is a judgment: with a model
      bound the model reasons over the envelope facts (injected into the
      context) via the base `Policy.judge`; offline it falls back to the
      deterministic `envelope_authorizes` check, announced as a fallback.

    A human-initiated intent (no agent named in context) is not applicable --
    the "off unless declared" posture of `Claim` and `Tenant`."""

    registry: Optional[EnvelopeRegistry] = None

    def judge(self, model_binding: Optional[Any] = None, **context: Any) -> tuple[bool, str]:
        agent = _first(context, _ACTOR_KEYS)
        if agent is None:
            return True, "no acting agent in context -- envelope policy not applicable (human-initiated)"
        if self.registry is None:
            return True, "no envelope registry bound -- policy inert"

        # The deterministic floor: absolute, never model-waivable.
        floor_ok, floor_reason = self.registry.floor(str(agent))
        if not floor_ok:
            return False, floor_reason

        # Above the floor, scope/tier authorization is judged. The envelope's
        # granted authority and the requested scope/tier go into the context
        # the model reasons over; the deterministic result rides as the
        # `envelope_authorizes` fallback variable for the offline path.
        envelope = self.registry.get(str(agent))
        scope = _first(context, _SCOPE_KEYS)
        tier = _first(context, _TIER_KEYS)
        deterministic_ok, deterministic_reason = envelope.authorized(
            scope=scope, tier=tier, secret=self.registry.secret
        )
        enriched = dict(context)
        enriched.update(
            {
                "acting_agent": str(agent),
                "envelope_scopes": ", ".join(envelope.scopes) or "(none)",
                "envelope_max_autonomy_tier": envelope.max_autonomy_tier,
                "envelope_standing": envelope.status,
                "envelope_trust_score": envelope.trust_score,
                "requested_scope": scope if scope not in (None, "") else "(none specified)",
                "requested_tier": tier if tier not in (None, "") else "(none specified)",
                "envelope_authorizes": deterministic_ok,
            }
        )
        # Delegate to EAR's own model-first / fallback-second machinery: with
        # a model it judges self.statement over `enriched`; offline it
        # evaluates self.fallback_expression (`envelope_authorizes`).
        model_active = model_binding is not None and getattr(model_binding, "lm", None) is not None
        complies, rationale = super().judge(model_binding=model_binding, **enriched)
        if not model_active:
            rationale = f"offline fallback -- {deterministic_reason}"
        return complies, rationale


def enforce_envelopes(
    runtime: Any,
    registry: EnvelopeRegistry,
    name: str = "AECC Capability Envelope",
) -> EnvelopePolicy:
    """Attach envelope enforcement onto `runtime` at runtime scope. Every
    subsequent cycle whose intent names an acting agent must clear the
    agent's live envelope before anything runs -- the single choke point,
    now covering *who* may act as well as *whether* the action is allowed.

    The registry's audit log is bound to the runtime's, so certification and
    revocation land on the same spine as the gate decisions they drive."""
    if registry.reasoning_log is None:
        registry.reasoning_log = getattr(runtime, "reasoning_log", None)
    policy = EnvelopePolicy(
        name=name,
        statement=(
            "The acting agent's capability envelope must authorize this action. Given "
            "the scopes the envelope grants (envelope_scopes) and its maximum autonomy "
            "tier (envelope_max_autonomy_tier), judge whether it covers the requested "
            "scope (requested_scope) and tier (requested_tier). The envelope has "
            "already cleared the certification and revocation floor; decide only "
            "whether the granted authority reaches this particular action."
        ),
        # The deterministic offline decision, computed in judge() and injected
        # as `envelope_authorizes`, is the fallback when no model is bound.
        fallback_expression="envelope_authorizes",
        registry=registry,
    )
    runtime.add_policy(policy)
    return policy
