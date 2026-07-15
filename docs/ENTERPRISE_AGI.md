# Enterprise AGI — binding constitutions onto the runtime

**Status:** Phases 1–2 shipped (`ear/enterprise.py`, `ear/compiler.py`, `ear/mcp_command_centre.py`) ·
**Repos:** [`qaflabindia/EAR`](https://github.com/qaflabindia/EAR) ·
[`qaflabindia/acc-skills`](https://github.com/qaflabindia/acc-skills)

## 1. Thesis

Enterprise AGI is the composition of two substrates that already exist:

- **EAR (Enterprise Agentic Runtime)** — the *execution* substrate. A
  declarative, English-authored stack: `Intent → Skill → Persona →
  Workflow → Process → Policy → Runtime → Reasoner`. `Governor.govern` is
  the single gate a cycle clears before anything runs; policies are
  enforced before a workflow executes.
- **acc-skills** — the *constitutional* substrate. Thirteen command
  centres, each carrying a constitution (`references/`), deterministic
  machinery (`scripts/`), persistent state (`state/*.json`), and an
  append-only ledger (`state/audit_trail.jsonl`).

The binding turns every command centre's constitutional rules into
**enforceable EAR policies** rather than advisory prose, puts every
centre's state behind **one store abstraction**, and folds every centre's
ledger onto **one audit spine** — so a human-, agent-, or evolution-
initiated intent all pass through the same choke point.

This document describes what Phase 1 actually ships. Later phases (§6) are
sketched, not built.

## 2. The three planes

Command centres bind to EAR subsystems by function. A plane is a binding
contract, not a deployment boundary — it dictates *which EAR subsystem a
centre attaches to*. The assignment lives in `enterprise.COMMAND_CENTRES`
and is read with `plane_of(slug)`.

| Plane | Command centres | Binds to (EAR) |
|---|---|---|
| **Governance** | AGCC, ATC, AECC, AAWDFC | `Governor`, `Policy`, `approval`, `identity` |
| **Operational** | AFCC, HRCC, TAIC, ALGCC, ARCC, AITCC | `Persona`, `Workflow`, `Process`, `Store`, `Tenant` |
| **Cognitive** | ALCC, AKC, ARC | `Learner`, `Knowledge`, `Experience`, `evolution` |

The governance plane governs the other two. `bind_command_centres` binds
governance first for exactly this reason.

## 3. Binding model (Phase 1)

### 3.1 Constitutions → Policies

`Constitution.from_directory(centre)` reads
`references/constitutional_rules.md` with EAR's own `Section` codec — the
same one every stacked file uses. Each `##` heading is one rule; its prose
is the `statement`; its recognized fields translate governance metadata
onto EAR's policy vocabulary:

| Constitutional field | Compiles to |
|---|---|
| heading `CR-AG01 — <title>` | `Policy.name` = `CR-AG01 · <title>` (named on the trail) |
| prose beneath the heading | `Policy.statement` (the LLM judges it) |
| `Fallback:` | `Policy.fallback_expression` (deterministic offline enforcement) |
| `Applies to:` | scope: `runtime`, `tools`, or a named workflow |
| `Verdict:` | the gate behaviour (§3.2) |
| `Rank:` | constitutional priority (lower prevails; orders compilation) |
| `Escalate:` | an `ESCALATE` rule's deadline |

`ConstitutionalRule.to_policy()` produces the EAR `Policy`;
`Constitution.to_policy_markdown()` renders a whole `policy.md` the
existing `Loader` reads back unchanged. A constitution compiled to
`policy.md` and reloaded through `load_runtime` yields the same policies it
started as — **English stays the source of truth**, and nothing an author
wrote is silently dropped.

### 3.2 AGCC verdicts → the policy gate

`Governor.govern` is the single choke point. `enterprise.Verdict` maps
AGCC's verdict vocabulary onto it:

| AGCC verdict | EAR behaviour |
|---|---|
| `EXECUTE` | rule never constrains (advisory no-op) |
| `EXECUTE_WITH_ADVISORY` | passes; recorded as an advisory on the log |
| `CONSTRAIN` | passes; advisory (context mutation — caps/narrowed scope — is a later phase) |
| `DEFER` | parks pending a human verdict (`Approval` / `ApprovalRequired`) |
| `ESCALATE` | parks, and escalates after the declared deadline |
| `HALT` | hard, unwaivable violation; the cycle is refused |

`HALT`, `DEFER` and `ESCALATE` are *blocking* verdicts — they attach as
policies and are genuinely enforced through `Governor.govern`. The advisory
verdicts compile too (nothing is dropped) but ride the reasoning log rather
than gate, because EAR's gate today blocks or parks; it does not advise. An
absent verdict defaults to `HALT`: a constitutional rule with no stated
consequence is a hard constraint, never a silent pass.

### 3.3 State → Store

`CommandCentreBackend(centre)` exposes a centre's `state/*.json` through
EAR's `CatalogueBackend` protocol (`list / exists / read / write /
delete`), satisfied structurally (PEP 544, no inheritance), plus
`read_json`/`write_json` for the JSON shape acc-skills' scripts use.

- **Phase 1 (adapter, shipped):** the JSON files remain the source of
  truth; the adapter exposes them to `Store`. Zero changes to acc-skills.
- **Phase 2 (canonical, later):** `Store` becomes canonical, keyed per
  `Tenant`, and the JSON files become an export format — what buys
  multi-tenancy across all thirteen centres.

The `audit_trail.jsonl` is **never adapted as state**:
`CommandCentre.mirror_audit(runtime)` folds each ledger line onto EAR's one
audit spine (`reasoning_log`), so there is exactly one auditable trail.

## 4. The intent path

Binding a centre attaches its constitution at runtime scope, so every
action takes the same path regardless of origin:

```
Intent (human | agent | goal-pursuit | evolution)
  → Governor.govern → Policies incl. the bound constitution   (whether and how)
  → Workflow steps → Personas → Skills                        (execution)
  → reasoning_log                                             (one audit spine)
```

The constitution is not a parallel gate bolted alongside the runtime — its
rules *are* runtime policies, judged and recorded exactly like any other.

## 5. Invariants (upheld by Phase 1)

1. **One choke point.** Bound constitutional rules enforce through
   `Governor.govern`; there is no private governance path.
2. **One audit spine.** Constitutional checks record on the runtime's
   `reasoning_log`; `mirror_audit` folds a centre's prior ledger onto it.
3. **Constitutions are policies.** A rule that cannot be judged as a
   statement (with optional deterministic fallback) is a documentation bug,
   not an exemption.
4. **English is the source of truth.** A constitution compiles to a
   `policy.md` markdown stack the Loader reads; code handlers stay optional.
5. **Least invasion first.** Phase 1 adapts to acc-skills as-is; the
   canonical-store migration is opt-in per centre, later.

## 6. Phasing

| Phase | Deliverable | Status |
|---|---|---|
| 1 | `CommandCentreBackend` (Store adapter over `state/`), constitutional-rules → `policy.md` compiler, AGCC verdict → gate mapping, bound at runtime scope | **shipped** |
| 2 | Centre → EAR-stack compiler for one operational centre end-to-end (AFCC), MCP packaging, single audit spine | **shipped** |
| 3 | AECC capability-envelope enforcement (a runtime-scope policy over `identity`/`signatures`), ATC adversarial-deliberation hook | planned |
| 4 | Cognitive plane: AKC-governed knowledge ingestion, ALCC → evolution loop under AAWDFC/AGCC gates | planned |
| 5 | State migration to canonical per-`Tenant` `Store`; multi-tenant rollout | planned |

## 6a. Phase 2 — the whole centre compiles to a stack

Phase 1 bound a centre's *constitution*. Phase 2 compiles the *whole
centre* into a runnable EAR stack (`ear/compiler.py`,
`StackCompiler` / `compile_command_centre`), mapping each acc-skills
artifact to the stack file an author would otherwise hand-write
(architecture §3.5):

| acc-skills artifact | EAR stack file |
|---|---|
| `SKILL.md` mission prose | `persona.md` (the persona's instructions) |
| `SKILL.md` `## Capabilities` | `skills.md` (one skill per capability) |
| `SKILL.md` `## Procedures` | `workflow.md` (steps delegating to the persona) |
| a process wrapping the workflows | `process.md` (the runtime's title) |
| `references/constitutional_rules.md` | `policy.md` (via `Constitution.to_policy_markdown`) |
| `references/*.md` (the rest) | `knowledge/` (documents the Librarian cites) |
| `SKILL.md` frontmatter org context | `tenant.md` (org id, fiscal year) |
| operating strategy | `memory.md` (knowledge sources, one audit trail) |

Two invariants hold. **Nothing an author wrote is dropped**: a `##` section
the compiler does not consume structurally (triggers, scope, notes) folds
into the persona's instructions as prose, and `compile(verify=True)` loads
the result once so every cross-reference resolves or the loader fails
loudly. **English stays the source of truth**: the output is markdown an
author reads, diffs and edits — the compiler is a starting point, not a
lock-in. The compiled `memory.md` declares one `.ear/reasoning.md` audit
trail, so the whole centre writes to the single spine.

`AFCC` is the worked example: `compile_command_centre("…/afcc", out)` writes
a nine-file stack that `load_runtime` runs as a first-class finance runtime,
its five `CR-FIN-*` constitutional rules enforcing at runtime scope.

## 6b. Phase 2 — a centre served as an MCP server

The out-of-process binding (architecture §3.4, the operational plane's
default): `ear/mcp_command_centre.py` packages a centre as a native MCP
server so any runtime reaches it with `Runtime.connect_mcp` over the same
stdio JSON-RPC `McpClient` already speaks. It exposes the centre's script
pentad as tools:

```
list_state                 the centre's state entries (names)
load_state(name)           one state entry, as JSON
update_state(name, value)  write one state entry (value is JSON)
evaluate(context)          judge the constitution against a context of facts
audit(entry)               append one line to the centre's ledger
```

`evaluate` runs the constitution's *deterministic* fallbacks (the server is
a plain subprocess, no model bound), so an out-of-process centre still
enforces its mechanically checkable rules; a rule with no fallback reports
not-applicable, never a silent pass. Run one with
`python -m ear.mcp_command_centre <centre-dir>` and declare it in a stack's
`memory.md` MCP section like any other server.

## 7. Worked example

```python
from ear import Runtime, Intent, Governor, Approval
from ear import CommandCentre

# Load the AGCC command centre and bind its constitution onto a runtime.
agcc = CommandCentre.load("path/to/acc-skills/agcc")
runtime = Runtime(name="Enterprise")
binding = agcc.bind(runtime)
print(binding.summary())          # "... (governance plane): 8 enforced, 0 advisory"

# A policy mutation under critical urgency violates CR-AG03 (HALT):
intent = Intent(text="mutate policy", context={"policy_mutation": True, "urgency": "critical"})
Governor().govern(runtime, intent)     # -> [CR-AG03 policy]  (blocks)

# A low-confidence action parks under CR-AG02 (DEFER) until a human waives it:
low = Intent(text="act", context={"production_confidence": 0.4})
Governor().govern(runtime, low)                                  # parks
Governor().govern(runtime, low, approval=Approval(verdict=True, approver="council"))  # released
```

Read a centre's state through the one store abstraction, and fold its
ledger onto the spine:

```python
agcc.state.read_json("trust_scores")     # {"scores": {...}}
agcc.mirror_audit(runtime)               # folds state/audit_trail.jsonl onto reasoning_log
```
