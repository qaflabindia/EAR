"""Enterprise AGI -- binding acc-skills' constitutional command centres
onto EAR's execution substrate.

Two substrates already exist. EAR is the *execution* substrate: the
declarative, English-authored stack `Intent -> Skill -> Persona -> Workflow
-> Process -> Policy -> Runtime -> Reasoner`, where `Governor.govern` is the
one gate a cycle clears before anything runs. `acc-skills` is the
*constitutional* substrate: thirteen command centres, each carrying a
constitution (`references/constitutional_rules.md`), deterministic
machinery (`scripts/`), persistent state (`state/*.json`), and an
append-only ledger (`state/audit_trail.jsonl`).

This module is the binding between them -- Phase 1 of the framework
architecture, "least invasion first":

* **Constitutions become policies.** Each rule in a centre's
  `references/constitutional_rules.md` compiles to an EAR `Policy`: the
  rule's prose is the `statement` an LLM judges, any mechanically checkable
  clause is the policy's `Fallback:` deterministic expression, and the
  rule's declared scope becomes the policy's `Applies to:`. Nothing an
  author wrote is dropped -- a rule that cannot be judged is a
  documentation bug, not an exemption (`Constitution.to_policy_markdown`
  renders a `policy.md` the existing `Loader` reads unchanged, so English
  stays the source of truth).

* **AGCC's verdict vocabulary maps onto the one policy gate.** A
  constitutional rule carries the verdict that applies when it triggers
  (`HALT`, `DEFER`, `ESCALATE`, `CONSTRAIN`, `EXECUTE_WITH_ADVISORY`), and
  `Verdict` translates each onto `Governor.govern` behaviour: `HALT` is a
  hard, unwaivable block; `DEFER`/`ESCALATE` park the cycle for a human
  (`ESCALATE` with a declared deadline); advisory verdicts ride the
  reasoning log rather than block. Enforcement therefore flows through the
  same choke point everything else does -- there is no private governance
  path.

* **State sits behind one store abstraction.** `CommandCentreBackend`
  exposes a centre's `state/*.json` through EAR's `CatalogueBackend`
  protocol (`list / exists / read / write / delete`). Phase 1 is an
  adapter: the JSON files stay the source of truth, zero changes to
  acc-skills; the canonical per-`Tenant` store is a later, opt-in phase.
  The `audit_trail.jsonl` is never adapted -- it folds onto EAR's one audit
  spine (`CommandCentre.mirror_audit`), so there is exactly one ledger.

Everything here is structural and deterministic -- code enforces, records,
and compiles; the model still judges every constitutional statement at
reasoning time against the active `ModelBinding`, exactly as any other
policy. Standard library only, like the rest of the package.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from .policy import Policy
from .section import Document, normalize, parse_document
from .strategy import days_in_prose

# --------------------------------------------------------------------------
# Planes -- which EAR subsystem a centre attaches to (framework architecture
# section 2). A plane is a binding contract, not a deployment boundary. The
# governance plane governs the other two, including the cognitive plane's
# self-modification loop.
# --------------------------------------------------------------------------

GOVERNANCE = "governance"
OPERATIONAL = "operational"
COGNITIVE = "cognitive"

# The thirteen command centres, each assigned to a plane by function. Keys
# are the acc-skills directory slugs; values are (plane, human title).
COMMAND_CENTRES: dict[str, tuple[str, str]] = {
    # Governance plane -- binds to Governor, Policy, approval, identity.
    "agcc": (GOVERNANCE, "Agentic Governance Command Centre"),
    "atc": (GOVERNANCE, "Adversarial Testing Command Centre"),
    "aecc": (GOVERNANCE, "Agent Envelope & Certification Command Centre"),
    "aawdfc": (GOVERNANCE, "Agentic Workflow Design & Formation Command Centre"),
    # Operational plane -- binds to Persona, Workflow, Process, Store, Tenant.
    "afcc": (OPERATIONAL, "Agentic Finance Command Centre"),
    "hrcc": (OPERATIONAL, "Agentic HR Command Centre"),
    "taic": (OPERATIONAL, "Talent Acquisition & Intelligence Command Centre"),
    "algcc": (OPERATIONAL, "Agentic Logistics Command Centre"),
    "arcc": (OPERATIONAL, "Agentic Relationships Command Centre"),
    "aitcc": (OPERATIONAL, "Agentic IT Operations Command Centre"),
    # Cognitive plane -- binds to Learner, Knowledge, Experience, evolution.
    "alcc": (COGNITIVE, "Agentic Learning Command Centre"),
    "akc": (COGNITIVE, "Agentic Knowledge Command Centre"),
    "arc": (COGNITIVE, "Agentic Reasoning Command Centre"),
}


def plane_of(slug: str) -> str:
    """The plane a command centre belongs to, by its acc-skills slug. An
    unknown centre defaults to the operational plane rather than failing --
    a new centre is operational until the framework assigns it otherwise."""
    entry = COMMAND_CENTRES.get(normalize(slug).replace("-", "").replace("_", ""))
    return entry[0] if entry else OPERATIONAL


# --------------------------------------------------------------------------
# Verdicts -- AGCC's vocabulary, mapped onto the EAR policy gate
# (framework architecture section 3.2).
# --------------------------------------------------------------------------


class Verdict:
    """The AGCC verdict a constitutional rule carries -- what happens when
    the rule triggers -- and its translation onto `Governor.govern`.

    | Verdict                 | EAR gate behaviour                          |
    |-------------------------|---------------------------------------------|
    | `EXECUTE`               | rule never constrains (advisory no-op)      |
    | `EXECUTE_WITH_ADVISORY` | passes; advisory appended to the log        |
    | `CONSTRAIN`             | passes; advisory (context mutation: a later |
    |                         | phase injects caps/narrowed scope)          |
    | `DEFER`                 | parks pending a human verdict               |
    | `ESCALATE`              | parks, and escalates after a declared period|
    | `HALT`                  | hard, unwaivable violation; cycle refused   |

    `HALT`, `DEFER` and `ESCALATE` are *blocking* verdicts: they compile to
    policies attached to the runtime and genuinely enforced. The advisory
    verdicts compile too -- nothing in the constitution is dropped -- but at
    bind time they are recorded on the reasoning log rather than gated,
    because EAR's gate today has two outcomes (block or park), not three."""

    EXECUTE = "EXECUTE"
    EXECUTE_WITH_ADVISORY = "EXECUTE_WITH_ADVISORY"
    CONSTRAIN = "CONSTRAIN"
    DEFER = "DEFER"
    ESCALATE = "ESCALATE"
    HALT = "HALT"

    _PARKING = {DEFER, ESCALATE}
    _BLOCKING = {HALT, DEFER, ESCALATE}
    _ALL = {EXECUTE, EXECUTE_WITH_ADVISORY, CONSTRAIN, DEFER, ESCALATE, HALT}

    @classmethod
    def read(cls, value: str) -> str:
        """Read a rule's declared `Verdict:` tolerantly -- case- and
        punctuation-insensitively, accepting a few natural spellings. An
        absent verdict defaults to `HALT`: a constitutional rule with no
        stated consequence is a hard constraint, never a silent pass.
        An unreadable verdict fails loudly -- a governance hole, not a
        default."""
        if not value or not value.strip():
            return cls.HALT
        folded = normalize(value).replace(" ", "_").replace("-", "_")
        aliases = {
            "execute": cls.EXECUTE,
            "proceed": cls.EXECUTE,
            "allow": cls.EXECUTE,
            "advisory": cls.EXECUTE_WITH_ADVISORY,
            "execute_with_advisory": cls.EXECUTE_WITH_ADVISORY,
            "warn": cls.EXECUTE_WITH_ADVISORY,
            "constrain": cls.CONSTRAIN,
            "cap": cls.CONSTRAIN,
            "defer": cls.DEFER,
            "approve": cls.DEFER,
            "approval": cls.DEFER,
            "escalate": cls.ESCALATE,
            "review": cls.ESCALATE,
            "halt": cls.HALT,
            "block": cls.HALT,
            "stop": cls.HALT,
            "deny": cls.HALT,
        }
        verdict = aliases.get(folded)
        if verdict is None:
            raise ValueError(
                f"Unreadable Verdict '{value}' -- write one of "
                f"EXECUTE, EXECUTE_WITH_ADVISORY, CONSTRAIN, DEFER, ESCALATE, HALT"
            )
        return verdict

    @classmethod
    def blocks(cls, verdict: str) -> bool:
        """Whether a violation under this verdict stops the cycle (as a hard
        block or a parked approval), rather than only advising."""
        return verdict in cls._BLOCKING

    @classmethod
    def parks(cls, verdict: str) -> bool:
        """Whether a violation parks for a human verdict rather than
        blocking outright."""
        return verdict in cls._PARKING


# --------------------------------------------------------------------------
# Constitutional rules -- one rule, compiled to one EAR Policy.
# --------------------------------------------------------------------------

# A rule id at the head of a heading: "CR-AG01", "CR-FIN-02", "AG7", etc.
_RULE_ID = re.compile(r"^\s*([A-Za-z]{1,6}[-_]?[A-Za-z0-9]{0,6}[-_]?\d{1,3})\b[\s\.:\-—–]*")


def _split_id_and_title(heading: str) -> tuple[str, str]:
    """Split a rule heading into (rule_id, title). "CR-AG01 -- Irreversible
    actions require authorization" -> ("CR-AG01", "Irreversible actions
    require authorization"). A heading with no recognizable id keeps the
    whole heading as the title and has an empty id."""
    match = _RULE_ID.match(heading)
    if not match:
        return "", heading.strip()
    rule_id = match.group(1)
    title = heading[match.end():].strip().lstrip("-—–:.").strip()
    return rule_id, title or heading.strip()


@dataclass
class ConstitutionalRule:
    """One rule from a command centre's constitution, and its compilation
    to an EAR `Policy`.

    The heading names the rule (`CR-AG01 -- <title>`); the prose beneath is
    the `statement` an LLM judges. Recognized fields translate the rule's
    governance metadata onto EAR's policy vocabulary:

        Verdict:      HALT | DEFER | ESCALATE | CONSTRAIN | EXECUTE_WITH_ADVISORY
        Fallback:     a deterministic boolean expression (offline enforcement)
        Applies to:   runtime | tools | <workflow name>   (the policy's scope)
        Rank:         constitutional priority (lower prevails in conflict)
        Escalate:     when a parked (ESCALATE) rule escalates, e.g. "after 1 day"
    """

    rule_id: str
    title: str
    statement: str
    verdict: str = Verdict.HALT
    fallback_expression: str = ""
    scope: str = "runtime"
    rank: Optional[int] = None
    escalation: str = ""

    @property
    def policy_name(self) -> str:
        """The name this rule's policy carries on the audit trail -- the id
        and title together, so a governance stop names the exact
        constitutional rule that produced it."""
        if self.rule_id and self.title:
            return f"{self.rule_id} · {self.title}"
        return self.rule_id or self.title

    def to_policy(self) -> Policy:
        """Compile this rule to an EAR `Policy`. The verdict decides the
        gate: a parking verdict (`DEFER`/`ESCALATE`) sets `approval_required`
        so a violation parks for a human; `ESCALATE` also carries the
        escalation deadline; `HALT` and the advisory verdicts leave the
        policy ungated (a HALT violation blocks; an advisory one is recorded
        by the binding rather than attached as a gate)."""
        parks = Verdict.parks(self.verdict)
        escalation = self.escalation if self.verdict == Verdict.ESCALATE else ""
        escalation_days = days_in_prose(escalation) if escalation else None
        if escalation and escalation_days is None:
            raise ValueError(
                f"Rule '{self.policy_name}' declares Escalate '{escalation}' but no "
                "readable period -- write 'Escalate: after 1 day'"
            )
        return Policy(
            name=self.policy_name,
            statement=self.statement,
            fallback_expression=self.fallback_expression,
            approval_required=parks,
            escalation=escalation,
            escalation_days=escalation_days,
        )

    def to_policy_section(self) -> str:
        """Render this rule as one `policy.md` section -- the exact shape
        `Loader._load_policies` reads back, `Applies to:` included (scope is
        part of a stack's policy.md, unlike a catalogued PolicyStore entry).
        Round-tripping a constitution through this and the Loader yields the
        same policies this rule's `to_policy` builds."""
        lines = [f"## {self.policy_name}", ""]
        if self.fallback_expression:
            lines.append(f"Fallback: {self.fallback_expression}")
        if Verdict.parks(self.verdict):
            lines.append("Approval: required")
        if self.verdict == Verdict.ESCALATE and self.escalation:
            lines.append(f"Escalate: {self.escalation}")
        lines.append(f"Applies to: {self.scope or 'runtime'}")
        lines.append("")
        # The verdict and rank ride as a leading note in the statement so a
        # compiled policy.md still reads as the constitution it came from,
        # and nothing (rank, verdict) is silently dropped in compilation.
        provenance = f"[{self.verdict}"
        if self.rank is not None:
            provenance += f", rank {self.rank}"
        provenance += "]"
        if self.statement:
            lines.append(f"{provenance} {self.statement}")
        else:
            lines.append(provenance)
        return "\n".join(lines).rstrip() + "\n"


@dataclass
class Constitution:
    """A command centre's constitution -- its `references/
    constitutional_rules.md`, parsed into `ConstitutionalRule`s and
    compilable to EAR policies.

    Parsed with EAR's own Section codec (the same one every stacked file
    uses): each `##` heading is a rule, its prose the statement, its
    recognized fields the governance metadata. The document's own preamble
    (the constitution's summary) is kept but is not a rule."""

    centre: str = ""
    preamble: str = ""
    rules: list[ConstitutionalRule] = field(default_factory=list)

    _FIELD_KEYS = (
        "verdict",
        "fallback",
        "fallback expression",
        "applies to",
        "applies",
        "scope",
        "rank",
        "escalate",
        "escalation",
    )

    @classmethod
    def from_markdown(cls, text: str, centre: str = "") -> "Constitution":
        document: Document = parse_document(text)
        rules: list[ConstitutionalRule] = []
        for section in document.sections:
            body = section.body(field_keys=cls._FIELD_KEYS)
            statement = "\n".join(
                filter(None, [body.prose] + [f"- {bullet}" for bullet in body.bullets])
            )
            rule_id, title = _split_id_and_title(section.name)
            rank_text = body.field("rank")
            rank: Optional[int] = None
            if rank_text:
                match = re.search(r"\d+", rank_text)
                rank = int(match.group()) if match else None
            rules.append(
                ConstitutionalRule(
                    rule_id=rule_id,
                    title=title,
                    statement=statement,
                    verdict=Verdict.read(body.field("verdict")),
                    fallback_expression=body.field("fallback", "fallback expression"),
                    scope=body.field("applies to", "applies", "scope") or "runtime",
                    rank=rank,
                    escalation=body.field("escalate", "escalation"),
                )
            )
        return cls(centre=centre, preamble=document.preamble or document.title, rules=rules)

    @classmethod
    def from_directory(cls, directory: Union[str, Path], centre: str = "") -> "Constitution":
        """Read a centre's `references/constitutional_rules.md`. A centre
        with no constitution file yields an empty Constitution rather than
        failing -- not every operational centre carries constitutional
        rules."""
        root = Path(directory)
        path = root / "references" / "constitutional_rules.md"
        centre = centre or root.name
        if not path.exists():
            return cls(centre=centre)
        return cls.from_markdown(path.read_text(encoding="utf-8"), centre=centre)

    def policies(self) -> list[Policy]:
        """Every rule compiled to an EAR Policy, in constitutional order --
        ranked rules first (lower rank prevails), then declared order."""
        return [rule.to_policy() for rule in self._ordered()]

    def _ordered(self) -> list[ConstitutionalRule]:
        return sorted(
            self.rules,
            key=lambda rule: (rule.rank if rule.rank is not None else 10_000),
        )

    def to_policy_markdown(self) -> str:
        """Render the whole constitution as a `policy.md` stack file the
        existing `Loader` reads unchanged. This is the compilation the
        framework's invariant demands: a constitution is not advisory prose,
        it is enforceable governance, and the bridge between the two is a
        markdown file an author can read, diff and review."""
        title = f"# {self.centre or 'Command Centre'} Constitution -- compiled policy"
        parts = [title, ""]
        if self.preamble:
            parts.append(self.preamble)
            parts.append("")
        parts.extend(rule.to_policy_section() for rule in self._ordered())
        return "\n".join(parts).rstrip() + "\n"


# --------------------------------------------------------------------------
# State -- one centre's state/ directory behind EAR's CatalogueBackend.
# --------------------------------------------------------------------------

_SLUG_UNSAFE = re.compile(r"[^a-z0-9]+")


def _state_slug(name: str) -> str:
    slug = _SLUG_UNSAFE.sub("_", normalize(name)).strip("_")
    return slug or "unnamed"


@dataclass
class CommandCentreBackend:
    """A command centre's `state/*.json` behind EAR's `CatalogueBackend`
    protocol -- the one store abstraction, satisfied structurally
    (`list / exists / read / write / delete`) so a centre's state is
    addressable the same way every other catalogue is.

    Phase 1 is an adapter: the JSON files stay the source of truth and
    acc-skills is unchanged. The append-only ledger `state/audit_trail.jsonl`
    is deliberately *not* a catalogue entry -- it is not state, it is the
    audit ledger, and it folds onto EAR's one audit spine
    (`CommandCentre.mirror_audit`) rather than being adapted here."""

    directory: Union[str, Path]
    AUDIT_TRAIL = "audit_trail.jsonl"

    def __post_init__(self) -> None:
        self.state_dir = Path(self.directory) / "state"

    def _path(self, name: str) -> Path:
        return self.state_dir / f"{_state_slug(name)}.json"

    def _match(self, name: str) -> Optional[Path]:
        """Resolve a name to an existing file case- and punctuation-
        insensitively, like every cross-reference in the stack. Falls back
        to the slugged path (used by `write` to create a new entry)."""
        target = self._path(name)
        if target.exists():
            return target
        wanted = _state_slug(name)
        for path in self.state_dir.glob("*.json"):
            if _state_slug(path.stem) == wanted:
                return path
        return None

    def list(self) -> list[str]:
        if not self.state_dir.exists():
            return []
        return sorted(
            path.stem
            for path in self.state_dir.glob("*.json")
            if path.name != self.AUDIT_TRAIL
        )

    def exists(self, name: str) -> bool:
        return self._match(name) is not None

    def read(self, name: str) -> str:
        path = self._match(name)
        if path is None:
            known = ", ".join(self.list()) or "none"
            raise FileNotFoundError(f"'{name}' is not in {self.state_dir} -- known: {known}")
        return path.read_text(encoding="utf-8")

    def write(self, name: str, text: str) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self._match(name) or self._path(name)).write_text(text, encoding="utf-8")

    def delete(self, name: str) -> None:
        path = self._match(name)
        if path is not None:
            path.unlink(missing_ok=True)

    # -- JSON convenience over the same primitives -------------------------

    def read_json(self, name: str) -> Any:
        """A centre's state parsed as JSON -- the shape acc-skills' scripts
        read and write it in."""
        return json.loads(self.read(name))

    def write_json(self, name: str, value: Any) -> None:
        self.write(name, json.dumps(value, indent=2, ensure_ascii=False) + "\n")

    @property
    def audit_trail_path(self) -> Path:
        """The centre's append-only ledger -- read to fold onto EAR's one
        audit spine, never adapted as state."""
        return self.state_dir / self.AUDIT_TRAIL


# --------------------------------------------------------------------------
# The command centre, and binding it onto a runtime.
# --------------------------------------------------------------------------


@dataclass
class Binding:
    """The result of binding a command centre onto a runtime: which
    constitutional policies were attached and enforced, which advisory rules
    were recorded rather than gated, and any governance-plane specialization
    the centre carries (AECC envelope enforcement, an ATC adversarial hook).
    Returned so a caller (and the audit trail) sees exactly what the binding
    did."""

    centre: str
    plane: str
    enforced: list[Policy] = field(default_factory=list)
    advisories: list[ConstitutionalRule] = field(default_factory=list)
    envelope_registry: Any = None
    envelope_policy: Any = None
    adversarial_review: Any = None
    knowledge_gate: Any = None
    epistemic_auditor: Any = None
    legitimacy_gate: Any = None
    learning_loop: Any = None

    def summary(self) -> str:
        line = (
            f"{self.centre} ({self.plane} plane): "
            f"{len(self.enforced)} enforced, {len(self.advisories)} advisory"
        )
        extras = []
        if self.envelope_policy is not None:
            extras.append(f"envelope enforcement ({len(self.envelope_registry.envelopes)} agents)")
        if self.adversarial_review is not None:
            extras.append("adversarial review hook")
        if extras:
            line += " + " + ", ".join(extras)
        return line


@dataclass
class CommandCentre:
    """One acc-skills command centre, loaded and bindable onto an EAR
    runtime. Its constitution becomes enforceable policies, its state sits
    behind the store abstraction, and its ledger folds onto the one audit
    spine.

    Load a centre from its directory with `CommandCentre.load(path)`, then
    `bind(runtime)` to attach its constitution as governance the runtime
    enforces through `Governor.govern` -- the same choke point every other
    intent clears."""

    slug: str
    name: str
    plane: str
    constitution: Constitution
    state: CommandCentreBackend
    directory: Optional[Path] = None

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "CommandCentre":
        root = Path(directory)
        slug = normalize(root.name).replace("-", "").replace("_", "")
        _plane, title = COMMAND_CENTRES.get(slug, (plane_of(root.name), ""))
        name = cls._read_name(root) or title or root.name.upper()
        return cls(
            slug=root.name,
            name=name,
            plane=plane_of(root.name),
            constitution=Constitution.from_directory(root, centre=name),
            state=CommandCentreBackend(root),
            directory=root,
        )

    @staticmethod
    def _read_name(root: Path) -> str:
        """The centre's display name from its `SKILL.md` -- a YAML-ish
        `name:` field or the first `#` heading. Absent, the caller falls
        back to the framework's title or the slug."""
        skill = root / "SKILL.md"
        if not skill.exists():
            return ""
        text = skill.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("name:"):
                return stripped.partition(":")[2].strip().strip("\"'")
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return ""

    def bind(self, runtime: Any) -> Binding:
        """Attach this centre's constitution onto `runtime` as enforceable
        governance. Blocking-verdict rules (`HALT`/`DEFER`/`ESCALATE`)
        become policies at their declared scope -- runtime-wide, tool-scoped,
        or attached to a named workflow -- and are enforced through
        `Governor.govern`. Advisory-verdict rules are recorded on the
        runtime's reasoning log rather than gated (EAR's gate blocks or
        parks; it does not advise), so nothing in the constitution is
        dropped and every rule is on the record.

        Returns a `Binding` naming what was enforced and what was advised."""
        binding = Binding(centre=self.name, plane=self.plane)
        log = getattr(runtime, "reasoning_log", None)
        for rule in self.constitution._ordered():
            if Verdict.blocks(rule.verdict):
                policy = rule.to_policy()
                self._attach(runtime, policy, rule.scope)
                binding.enforced.append(policy)
                self._record(
                    log,
                    stage="policy",
                    output=f"attached ({rule.verdict}, scope: {rule.scope or 'runtime'})",
                    rule=rule,
                )
            else:
                binding.advisories.append(rule)
                self._record(
                    log,
                    stage="policy",
                    output=f"advisory ({rule.verdict})",
                    rule=rule,
                )
        self._bind_specializations(runtime, binding)
        return binding

    def _bind_specializations(self, runtime: Any, binding: Binding) -> None:
        """Attach the governance-plane bindings a centre carries beyond its
        constitution. AECC contributes capability-envelope enforcement (Phase
        3): a runtime-scope policy gating every agent-initiated cycle on the
        acting agent's live envelope. ATC contributes the adversarial-review
        hook: an adversarial pass a caller runs over flagged intents. Both
        are additive -- the constitution has already bound above."""
        slug = normalize(self.slug).replace("-", "").replace("_", "")
        if slug == "aecc" and self.state.exists("authority_envelopes"):
            from .authority import EnvelopeRegistry, enforce_envelopes

            registry = EnvelopeRegistry.from_centre(
                self, reasoning_log=getattr(runtime, "reasoning_log", None)
            )
            binding.envelope_registry = registry
            binding.envelope_policy = enforce_envelopes(runtime, registry)
            # The registry is reachable on the runtime so operators and the
            # ATC hook can consult (and transition) it.
            runtime.envelope_registry = registry
        if slug == "atc":
            from .adversary import AdversarialReview

            review = AdversarialReview()
            binding.adversarial_review = review
            runtime.adversarial_review = review
        if slug == "akc":
            from .knowledge_governance import KnowledgeGate

            # The admission threshold governs only the offline floor; honour
            # an author-declared value from the centre's state rather than a
            # baked constant.
            declared = self._declared_number("sources", "admission_threshold")
            gate = KnowledgeGate(threshold=declared) if declared is not None else KnowledgeGate()
            binding.knowledge_gate = gate
            runtime.knowledge_gate = gate
        if slug == "arc":
            from .epistemic import EpistemicAuditor

            # Honour the escalation threshold the centre declares in its own
            # state (patterns.json) rather than the code default -- the "how
            # many flags is systematic?" line is the author's to set.
            declared = self._declared_number("patterns", "escalation_threshold")
            auditor = EpistemicAuditor(escalate_threshold=int(declared)) if declared is not None else EpistemicAuditor()
            binding.epistemic_auditor = auditor
            runtime.epistemic_auditor = auditor
        if slug == "aawdfc":
            from .evolution_loop import LegitimacyGate

            gate = LegitimacyGate()
            binding.legitimacy_gate = gate
            runtime.legitimacy_gate = gate
        if slug == "alcc":
            from .evolution_loop import LearningLoop

            loop = LearningLoop()
            binding.learning_loop = loop
            runtime.learning_loop = loop

    def _declared_number(self, state_name: str, key: str) -> Optional[float]:
        """A number the centre declares in its own state, or None when the
        centre declared none -- so a threshold that shapes a decision comes
        from the author, not a code constant. Absent state or a bad value
        reads as 'not declared', never a guess."""
        try:
            if not self.state.exists(state_name):
                return None
            data = self.state.read_json(state_name)
            value = data.get(key) if isinstance(data, dict) else None
            return float(value) if isinstance(value, (int, float)) else None
        except Exception:  # noqa: BLE001 -- a malformed state file is 'not declared', not fatal
            return None

    @staticmethod
    def _attach(runtime: Any, policy: Policy, scope: str) -> None:
        """Attach a compiled policy at its declared scope. Runtime and tool
        scopes bind directly; a named scope binds to a matching workflow if
        the runtime carries one, and otherwise falls back to runtime scope
        (a workflow that does not exist yet is not a reason to silently drop
        a constitutional rule)."""
        lowered = normalize(scope or "runtime")
        tool_scopes = {"tools", "tool", "tool calls", "tool call", "tool invocations", "any tool"}
        if lowered in tool_scopes:
            tool_policies = getattr(runtime, "tool_policies", None)
            if tool_policies is not None and policy not in tool_policies:
                tool_policies.append(policy)
            return
        if lowered in {"runtime", "the runtime", "all", "everything", "global"} or "runtime" in lowered:
            runtime.add_policy(policy)
            return
        for process in getattr(runtime, "processes", []):
            for workflow in getattr(process, "workflows", []):
                if normalize(workflow.name) == lowered:
                    if policy not in workflow.policies:
                        workflow.add_policy(policy)
                    return
        runtime.add_policy(policy)

    @staticmethod
    def _record(log: Any, stage: str, output: str, rule: ConstitutionalRule) -> None:
        if log is None:
            return
        log.record(
            stage=stage,
            inputs={
                "command_centre": rule.rule_id or rule.title,
                "constitutional_rule": rule.policy_name,
                "verdict": rule.verdict,
            },
            output=output,
            rationale=rule.statement,
        )

    def mirror_audit(self, runtime: Any) -> int:
        """Fold this centre's existing `state/audit_trail.jsonl` onto EAR's
        one audit spine -- each ledger line recorded through the runtime's
        reasoning log, so a centre's private history joins the single
        auditable trail rather than staying a separate ledger. Returns how
        many lines were folded. A centre with no prior ledger folds nothing.

        The centre's own ledger is left untouched (Phase 1 does not rewrite
        acc-skills); this only *mirrors* it onto the spine so future audit
        lives in one place."""
        log = getattr(runtime, "reasoning_log", None)
        path = self.state.audit_trail_path
        if log is None or not path.exists():
            return 0
        folded = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                entry = {"raw": line}
            log.record(
                stage="audit",
                inputs={"command_centre": self.name, "ledger_entry": entry},
                output=str(entry.get("action") or entry.get("event") or "ledger entry"),
                rationale=str(entry.get("rationale") or entry.get("reason") or ""),
            )
            folded += 1
        return folded


def load_command_centres(root: Union[str, Path]) -> dict[str, CommandCentre]:
    """Load every command centre under an acc-skills root -- one per
    subdirectory that carries a constitution or a `SKILL.md`. Keyed by the
    centre's directory slug."""
    base = Path(root)
    centres: dict[str, CommandCentre] = {}
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        has_constitution = (child / "references" / "constitutional_rules.md").exists()
        if has_constitution or (child / "SKILL.md").exists():
            centres[child.name] = CommandCentre.load(child)
    return centres


def bind_command_centres(runtime: Any, centres: dict[str, CommandCentre]) -> list[Binding]:
    """Bind a set of command centres onto a runtime, governance plane first
    -- the governance plane governs the others (framework architecture
    section 2), so its constitution attaches before anything it governs.
    Returns one `Binding` per centre."""
    order = {GOVERNANCE: 0, OPERATIONAL: 1, COGNITIVE: 2}
    ordered = sorted(centres.values(), key=lambda centre: order.get(centre.plane, 3))
    return [centre.bind(runtime) for centre in ordered]
