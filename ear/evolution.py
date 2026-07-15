"""Evolution -- governed self-modification, configured, never assumed.

EAR already lets a runtime change itself in narrow, recorded ways: the
Acquirer declares new tools mid-deliberation, the Optimizer refines and
persists skill prompts, adaptations distill standing lessons. Evolution is
the governance layer over all of that: an `EvolutionPolicy` says which
*kinds* of change the runtime may make to itself, which are off the table
entirely, and what every permitted change must carry before it is promoted
-- a sandbox to trial in, an evaluation to pass, an explanation on the
record, a human approval for the sensitive kinds, and a rollback so no
change is a one-way door.

    runtime.enable_evolution(EvolutionPolicy(
        allowed_changes=["skill_prompt", "skill_creation", "strategy",
                         "workflow_branch", "validation_rule", "tool_adapter"],
        prohibited_changes=["hard_policy", "approval_authority",
                            "audit_logging", "data_access_boundary"],
        require_sandbox=True,
        require_evaluation=True,
        require_explanation=True,
        require_human_approval_for=["generated_code", "workflow_structure",
                                    "production_promotion"],
        rollback_required=True,
    ))

The posture is default-deny three times over. A runtime that never called
`enable_evolution` refuses every proposed change. An enabled runtime
refuses any kind its policy does not explicitly allow -- an unlisted kind
is denied, never inferred to be fine. And a prohibited kind is refused even
if it is also listed as allowed: the prohibition always wins, so governance
machinery (hard policies, approval authority, audit logging, data-access
boundaries) can be fenced off in one place and no allow-list mistake ever
opens the fence.

The split of labor is the same as everywhere else in this runtime: the
change itself may be model-proposed (a refined prompt, a generated tool
adapter), but *whether it lands* is enforced in code -- `Evolver.propose`
walks the policy's gates in order, records every refusal, park and
promotion as an `evolution` trail record, applies the change only once
every gate is passed, evaluates the changed runtime, and rolls back on a
failed evaluation or a crashed apply. A human approval reuses the same
`Approval` document and `ApprovalRequired` park as policy.md's gates: the
model never waives its own gate.

Kinds are plain words, matched case- and punctuation-insensitively like
every other cross-reference in the stack ("skill prompt", "skill-prompt"
and "skill_prompt" are the same kind). The vocabulary is the author's, not
an enum's -- the policy governs whatever kinds the author names.

The policy can also be authored in memory.md, like every other piece of
operating strategy -- an `## Evolution` section, read by the Loader into
`enable_evolution` on load:

    ## Evolution

    The runtime may improve itself within these fences. Trial every change
    in the sandbox, evaluate it before promotion, explain it on the record,
    and keep a rollback.

    - Allowed: skill prompt, skill creation, strategy, workflow branch,
      validation rule, tool adapter
    - Prohibited: hard policy, approval authority, audit logging,
      data access boundary
    - Approval required: generated code, workflow structure,
      production promotion

Enabling evolution also puts the existing self-extension surfaces under
the same policy: the Acquirer's `create_tool`/`retire_tool` are governed
as `tool_adapter` changes once a policy is enabled (see ear/acquirer.py),
so the tools-that-create-tools loop cannot outrun the fence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .approval import ApprovalRequired
from .policy import Policy
from .section import Body, normalize


class EvolutionDenied(PermissionError):
    """Raised when a proposed self-modification is refused -- by a missing
    policy, a prohibited or unlisted kind, or an unmet requirement (no
    explanation, no sandbox, no rollback, no or failed evaluation). A
    PermissionError, so handlers that treat governance stops as refusals
    keep working; carries the kind and the reason."""

    def __init__(self, kind: str, reason: str) -> None:
        self.kind = kind
        self.reason = reason
        super().__init__(f"Evolution denied for '{kind}': {reason}")


@dataclass
class EvolutionChange:
    """One proposed self-modification: what kind of change it is, what it
    touches, and the explanation that must accompany it when the policy
    requires one. The payload carries whatever the appliers need (a new
    prompt, generated source, a strategy delta) -- opaque to the gate,
    which judges the kind and the requirements, never the content."""

    kind: str
    name: str = ""
    description: str = ""
    explanation: str = ""
    payload: dict = field(default_factory=dict)

    def label(self) -> str:
        return f"{self.name} ({self.kind})" if self.name else self.kind


@dataclass
class EvolutionPolicy:
    """Which kinds of self-modification a runtime may make, which it may
    never make, and what every permitted change must carry. Default-deny:
    a kind neither listed nor allowed is refused, and a prohibited kind is
    refused even when the allow-list also names it."""

    allowed_changes: list[str] = field(default_factory=list)
    prohibited_changes: list[str] = field(default_factory=list)
    require_sandbox: bool = True
    require_evaluation: bool = True
    require_explanation: bool = True
    require_human_approval_for: list[str] = field(default_factory=list)
    rollback_required: bool = True

    def refusal(self, kind: str) -> Optional[str]:
        """Why this kind of change is refused, or None when it is allowed.
        The prohibition always wins over the allow-list."""
        key = normalize(kind)
        if key in {normalize(prohibited) for prohibited in self.prohibited_changes}:
            return f"'{kind}' is a prohibited change -- the prohibition always wins, even over the allow-list"
        if key not in {normalize(allowed) for allowed in self.allowed_changes}:
            return f"'{kind}' is not among the allowed changes -- evolution is default-deny, an unlisted kind is refused"
        return None

    def permits(self, kind: str) -> bool:
        return self.refusal(kind) is None

    def needs_approval(self, kind: str) -> bool:
        """Whether this kind of change needs a human verdict before it may
        apply -- the model never waives its own gate."""
        key = normalize(kind)
        return key in {normalize(gated) for gated in self.require_human_approval_for}

    def describe(self) -> str:
        """One reviewable line for the trail: the fences this policy sets."""
        requirements = [
            name
            for name, required in (
                ("sandbox", self.require_sandbox),
                ("evaluation", self.require_evaluation),
                ("explanation", self.require_explanation),
                ("rollback", self.rollback_required),
            )
            if required
        ]
        parts = [
            f"allowed: {', '.join(self.allowed_changes) or '(none)'}",
            f"prohibited: {', '.join(self.prohibited_changes) or '(none)'}",
            f"requires: {', '.join(requirements) or '(nothing)'}",
        ]
        if self.require_human_approval_for:
            parts.append(f"human approval for: {', '.join(self.require_human_approval_for)}")
        return "; ".join(parts)

    @classmethod
    def from_prose(cls, body: Body) -> "EvolutionPolicy":
        """Read a policy out of an `## Evolution` section's prose and
        bullets. Kind lists ride bullets ('- Allowed: skill prompt, ...',
        '- Prohibited: ...', '- Approval required: ...'); the four
        requirements default on -- the safe posture -- and only explicit
        relaxing language in the prose ('no sandbox needed', 'skip the
        evaluation', 'rollback is optional') turns one off."""
        policy = cls()
        for line in list(body.bullets) + body.prose.split("\n"):
            if ":" not in line:
                continue
            label, values = line.split(":", 1)
            kinds = [item.strip() for item in values.replace(";", ",").split(",") if item.strip()]
            key = normalize(label)
            if "prohibit" in key or "forbid" in key or "never" in key:
                policy.prohibited_changes.extend(kinds)
            elif "approval" in key or "approve" in key or "human" in key:
                policy.require_human_approval_for.extend(kinds)
            elif "allow" in key or "permit" in key:
                policy.allowed_changes.extend(kinds)
        text = "\n".join(filter(None, [body.prose] + list(body.bullets)))
        policy.require_sandbox = _still_required(text, "sandbox")
        policy.require_evaluation = _still_required(text, "evaluat")
        policy.require_explanation = _still_required(text, "explanation", "explain")
        policy.rollback_required = _still_required(text, "rollback", "roll back")
        return policy


_RELAXING = ("no ", "not required", "not needed", "optional", "skip", "without")


def _still_required(text: str, *keywords: str) -> bool:
    """True unless a sentence naming the requirement carries explicit
    relaxing language -- the default is the safe posture, and only the
    author saying so turns a requirement off."""
    for sentence in text.lower().replace(";", ".").split("."):
        if any(keyword in sentence for keyword in keywords):
            if any(relaxer in sentence for relaxer in _RELAXING):
                return False
    return True


@dataclass
class Evolver:
    """The gate every proposed self-modification walks through: the policy
    judges the kind, code enforces the requirements in order, and every
    refusal, park and promotion is an `evolution` trail record. Standalone,
    like the Optimizer and the Acquirer -- not a per-cycle pipeline stage."""

    def propose(
        self,
        runtime: Any,
        change: EvolutionChange,
        apply: Callable[[], Any],
        rollback: Optional[Callable[[], Any]] = None,
        approval: Optional[Any] = None,
        evaluate: Optional[Callable[[], Any]] = None,
    ) -> str:
        """Walk one proposed change through the policy's gates: kind
        allowed, explanation present, human approval where the policy
        demands one, a sandbox to trial in, a rollback in hand, an
        evaluation to pass. Only then does `apply` run; then `evaluate`
        judges the changed runtime, and a failed evaluation (or a crashed
        apply) rolls the change back. Returns the promotion note; every
        refusal raises `EvolutionDenied` (or parks as `ApprovalRequired`),
        on the record either way."""
        policy = getattr(runtime, "evolution_policy", None)
        if policy is None:
            raise self._deny(runtime, change, "evolution is not enabled on this runtime -- call enable_evolution with an EvolutionPolicy first")

        refusal = policy.refusal(change.kind)
        if refusal is not None:
            raise self._deny(runtime, change, refusal)

        if policy.require_explanation and not change.explanation.strip():
            raise self._deny(runtime, change, "the policy requires an explanation and the change carries none")

        approver = ""
        if policy.needs_approval(change.kind):
            if approval is None:
                gate = Policy(
                    name=f"Evolution approval: {change.label()}",
                    statement=f"A human must approve every '{change.kind}' change before it applies.",
                    approval_required=True,
                )
                self._record(runtime, change, f"PENDING -- human approval required for '{change.kind}' changes")
                raise ApprovalRequired([gate])
            if approval.verdict is not True:
                verdict = "rejected" if approval.verdict is False else "unreadable"
                raise self._deny(runtime, change, f"the human verdict was {verdict} -- only an approved verdict releases the gate")
            approver = getattr(approval, "approver", "") or ""

        if policy.require_sandbox and getattr(runtime, "sandbox", None) is None:
            raise self._deny(runtime, change, "the policy requires a sandbox trial and no Sandbox confines this runtime")

        if policy.rollback_required and rollback is None:
            raise self._deny(runtime, change, "the policy requires a rollback and none was provided -- no change may be a one-way door")

        if policy.require_evaluation and evaluate is None:
            raise self._deny(runtime, change, "the policy requires an evaluation and none was provided -- an unevaluated change is never promoted")

        # AAWDFC legitimacy gate (framework §7): a machine-created change must
        # be judged fit to exist -- explained, constitutionally compatible,
        # coherent -- before it is applied. Attached by binding the AAWDFC
        # command centre; absent, the loop is unchanged.
        legitimacy = getattr(runtime, "legitimacy_gate", None)
        if legitimacy is not None:
            verdict = legitimacy.judge(change, runtime)
            if not getattr(verdict, "legitimate", True):
                raise self._deny(runtime, change, f"AAWDFC judged the change illegitimate -- {verdict.reason}")

        try:
            apply()
        except Exception as error:
            if rollback is not None:
                rollback()
            self._record(runtime, change, f"FAILED -- apply raised {type(error).__name__}: {error}" + ("; rolled back" if rollback is not None else ""))
            raise

        if policy.require_evaluation:
            outcome = evaluate()
            passed = bool(getattr(outcome, "passed", outcome))
            if not passed:
                if rollback is not None:
                    rollback()
                raise self._deny(runtime, change, "the evaluation failed -- the change was rolled back, not promoted")

        note = f"promoted '{change.label()}'"
        if approver:
            note += f", approved by {approver}"
        if policy.require_evaluation:
            note += ", evaluation passed"
        self._record(runtime, change, note)
        return note

    # -- helpers -------------------------------------------------------------

    def _deny(self, runtime: Any, change: EvolutionChange, reason: str) -> EvolutionDenied:
        self._record(runtime, change, f"DENIED -- {reason}")
        return EvolutionDenied(change.kind, reason)

    @staticmethod
    def _record(runtime: Any, change: EvolutionChange, output: str) -> None:
        log = getattr(runtime, "reasoning_log", None)
        if log is None:
            return
        log.record(
            stage="evolution",
            inputs={
                "kind": change.kind,
                "name": change.name,
                "description": change.description,
                "explanation": change.explanation,
            },
            output=output,
        )
