# Governance

Governance in EAR follows one division of labour everywhere:

> The model judges **whether** a rule is triggered; code enforces the outcome and
> records it. Only a human can waive what is authored to need a human.

This page covers policies, the gate, approval, tool-scoped rules, budgets, the
tamper-evident trail, tenancy, the sandbox, and governed self-modification.

---

## Policies and the gate

A `Policy` is a plain-English statement the model judges, with an optional
deterministic fallback and a scope. `Governor.govern` is the single choke point a
cycle clears before anything runs — there is no separate governance path.

```markdown
## Loan Amount Cap

The loan must not exceed $75,000.

Fallback: loan_amount <= 75000
Applies to: Underwriting Workflow
```

- **statement** — judged by the model against the intent and its context.
- **`Fallback:`** — a deterministic expression that keeps the rule enforced
  offline (no model). Evaluated safely, not with `eval`.
- **`Applies to:`** — `runtime` (every cycle), a **named workflow**, or `tools`
  (see [tool-scoped policies](#tool-scoped-policies)).

A violation is a **recorded outcome, not a crash**: the cycle raises a
`PermissionError`, and the decision document is written with `Status: BLOCKED`
naming the policy and the judge's rationale. Both passes and blocks are `policy`
records on the [trail](#the-tamper-evident-trail) — a refusal is an audit event,
never a gap.

```python
try:
    runtime.reason(Intent(text="…", context={"loan_amount": 200000}))
except PermissionError as blocked:
    print("Blocked:", blocked)
```

---

## Approval gates

A policy authored with `Approval: required` converts its hard block into a
**parkable gate** — the model judges whether the gate triggers, but only a human
can waive it:

```markdown
## Large Loan Human Approval

Loan amounts above $50,000 must be approved by a human approver.

Fallback: loan_amount <= 50000
Approval: required
Applies to: Underwriting Workflow
```

When such a policy is violated the cycle raises `ApprovalRequired` (a
`PermissionError`, so existing handlers keep working) and the decision document is
written with `Status: PENDING APPROVAL`, naming the policies awaiting a verdict. A
human releases it by dropping an approval document beside the intent
(`approvals/<name>.md`):

```markdown
# Approval -- Underwrite a $60,000 loan

Verdict: approved
Approver: alice@example.com

> Reviewed the exception; the collateral covers it.
```

The next `Exchange.run` finishes the cycle: **approved** passes the gate on the
record, with the approver's name and note; **rejected** blocks it like any
violation. A hard (ungated) violation always wins over a pending gate, an
unreadable verdict fails loudly rather than leaving the cycle silently parked, and
a parked cycle is fully accounted but writes no memory — nothing was decided yet.

Restrict **who** may waive with an allow-list, matched case- and
punctuation-insensitively:

```markdown
Approvers: alice@example.com, risk-council
```

An approved verdict from someone off the list waives nothing — the record says who
was refused and why. A gate may also declare `Escalate: after 3 days`; a parked
cycle found past that deadline is marked `ESCALATED` (still releasable, but no
longer quietly waiting) by the [Journeys runner](OPERATIONS.md#journeys-durable-resumable-execution).

---

## Tool-scoped policies

A policy scoped `Applies to: tools` is judged against a tool's name and arguments
**before the call runs**:

```markdown
## Transfer Cap

The wire transfer tool must never move more than $10,000 in one call.

Fallback: amount <= 10000
Applies to: tools
```

A violation blocks that one call — the refusal returns to the model as text
(exactly like a tool failure, so the model reasons on) and the block is a `tool`
trail record naming the policy. This governs native tools and connected MCP tools
alike.

---

## Budgets

Budgets are enforced in code, before work runs, and land on the trail like any
other stop:

- **Turn / iteration budgets** — the tool loop, panels (`rounds`, `max_turns`),
  routing revisits, and the autonomous goal loop are all bounded so nothing runs
  away.
- **Pricing → dollars** — declare a `Pricing` section in `memory.md` ("Input
  tokens cost $3 per million; output tokens cost $15 per million.") and usage
  records carry real dollars. A figure nobody declared is never invented.

---

## The tamper-evident trail

Every reasoning step is recorded on `runtime.reasoning_log` (a `ReasoningLog`),
one stage-labelled record per judgment. The trail is the system's audit spine —
see [Operations → The audit trail](OPERATIONS.md#the-audit-trail) for the full
stage list and how to persist and export it.

For governance specifically, three properties matter:

- **Blocked cycles are logged.** A policy violation is an audit event with the
  judge's rationale, not a missing record.
- **Hash-chained.** Every flushed record carries a hash chained over the previous
  record's (stdlib `hashlib`), in both codecs. `ReasoningLog.verify(path)`
  recomputes the chain over the file's own bytes and either proves it unbroken or
  names the exact record where an edit, insertion or deletion first breaks it.
- **Retention is rotation, not deletion.** Declare "keep 90 days" in the audit
  prose and cycles past the window are replaced by a single `retention` note and
  the file is rewritten, re-chained and still verifiable — what was rotated out is
  accounted for, never silently gone.

```python
ok, first_break = runtime.reasoning_log.verify(".ear/reasoning.md")
```

---

## Tenancy and claims

A stack's [`tenant.md`](AUTHORING.md#tenantmd) names the org it belongs to. When
instances belong to different orgs, pass a `Claim` alongside the work to enforce
the boundary:

```python
from ear import Claim

runtime.reason(intent, claim=Claim(subject="alice", org_ids=("org_acme_prod",)))
# or: kernel.submit("lending", intent, claim=claim)
```

A `Claim` not authorized for the target instance's `org_id` refuses the cycle
before it starts (`TenantBoundaryViolation`). Omit `claim` entirely and nothing
changes from before it existed — the boundary is off unless declared.

---

## Sandbox

Declare a `## Sandbox` section in `memory.md` and every instance gets its own
filesystem-confined, resource-limited workspace — built from the standard library
(`pathlib` + `subprocess` + POSIX `resource`), no Docker:

```markdown
## Sandbox

Isolate each runtime under `.ear/box`. Shell commands time out after 30 seconds;
limit memory to 512 MB. Expose file and shell tools.
```

The sandbox confines the runtime's file tools to its root (an escaping path raises
`SandboxViolation`, which returns to the model as text), runs commands with a
wall-clock timeout and optional CPU/memory limits, and strips the ambient
process's secrets from the command environment — so a spawned command never
inherits your `ANTHROPIC_API_KEY`. Exposed tools (`read_file` / `write_file` /
`list_files` / `run_shell`) join the cycle's tool loop, each governed by the same
tool-scoped policies. Isolation nests: a spawned subagent gets its own child box.

Stated honestly: a pure-stdlib sandbox is a *containment convention* for EAR's own
file tools plus a *resource and time boundary* for spawned commands — **not a
security jail against hostile code** (`cwd` confinement is not `chroot`). For a
true isolation boundary, plug an OS-container provider into the same seam:
anything exposing `resolve` / `read_text` / `write_text` / `run` / `as_tools` can
stand in for `Sandbox` on `runtime.sandbox`.

---

## Evolution

`enable_evolution` is the governance layer over a runtime changing itself. An
`EvolutionPolicy` says which *kinds* of change are allowed, which are off the
table entirely, and what every permitted change must carry:

```python
from ear import EvolutionPolicy, EvolutionChange, Examiner

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
```

The posture is **default-deny three times over**: a runtime that never enabled
evolution refuses every change; an unlisted kind is refused, never inferred fine;
and a prohibited kind is refused even when the allow-list also names it — so the
governance machinery itself (hard policies, approval authority, audit logging,
data-access boundaries) stays fenced off no matter what.

The change may be model-proposed, but whether it lands is enforced in code: the
`Evolver` walks the gates in order, applies only once every gate passes, rolls
back on a failed evaluation or a crashed apply, and human approval rides the same
`Approval` / `ApprovalRequired` machinery as policy gates — so the model never
waives its own gate. Every refusal, park and promotion is an `evolution` trail
record.

The policy is also authorable in `memory.md`, applied on load:

```markdown
## Evolution

Trial every change in the sandbox, evaluate it before promotion, explain it on
the record, and keep a rollback.

- Allowed: skill prompt, skill creation, strategy, workflow branch, validation rule, tool adapter
- Prohibited: hard policy, approval authority, audit logging, data access boundary
- Approval required: generated code, workflow structure, production promotion
```

The four requirements default **on** — only explicit relaxing prose turns one off,
and a section authored with disabling language ("evolution is disabled") leaves
the runtime refusing every change, exactly as if the section were absent.

---

## Where to go next

- **[Authoring Guide](AUTHORING.md)** — how to write policies and gates.
- **[Operations](OPERATIONS.md)** — the trail, the server, running a fleet.
- **[Concepts](CONCEPTS.md)** — the model behind the guardrails.
