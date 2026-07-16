# Authoring Guide

An EAR stack is a folder of plain-English markdown files. `load_runtime(path)`
reads whichever exist and assembles a `Runtime` from them. This page is the
reference for every file and section.

Cross-references between files are **by name**, case- and
punctuation-insensitive. An unresolved reference fails loudly at load with the
list of known names — nothing you write is silently dropped. A workflow that no
process references is wrapped in a process of its own rather than lost.

| File | Holds | Shape |
|---|---|---|
| [`skills.md`](#skillsmd) | prompts → skills | `## skill name` + prose prompt |
| [`persona.md`](#personamd) | skills → personas | prose instructions + `Skills:` |
| [`workflow.md`](#workflowmd) | steps → workflows | numbered steps, `(Persona)` delegates, `Policies:` |
| [`process.md`](#processmd) | workflows → processes | prose + `Workflows:`; the `#` title names the runtime |
| [`policy.md`](#policymd) | governance | statement + `Fallback:` / `Applies to:` |
| [`tenant.md`](#tenantmd) | the owning org | `Org id:`, fiscal year, timezone (optional) |
| [`memory.md`](#memorymd-the-operating-strategy) | the operating strategy | sections, below |

---

## `skills.md`

Each `##` heading is a skill; the prose beneath it is the prompt. A skill is a
capability described in plain English — no handler code required.

```markdown
# Skills

## assign_risk_grade

Read the applicant's credit score and debt-to-income ratio from the intent
context, and assign a risk grade from A (strongest) to E (weakest).
```

An advanced skill may carry a Python `handler` (bound in code); it stays optional.
A skill whose handler exists binds automatically for the cycle's plan — see
[Tools](#tools).

---

## `persona.md`

A persona is standing instructions plus a set of skills, stacked by name.

```markdown
# Personas

## Credit Risk Guru

Underwrite conservatively; prefer a clear decline over a risky approval.

Skills: assign_risk_grade, decide_application
```

A persona may carry a large skill library without every prompt entering every
call: the `SkillSelector` stacks only the skills relevant to the current intent
(model-ranked, keyword fallback). It is on by default and short-circuits —
returning every skill — whenever a persona has few enough skills, so small
personas behave exactly as before.

---

## `workflow.md`

A workflow is an ordered list of numbered steps, each delegated to a persona in
`(Parentheses)`, governed by its own policies.

```markdown
# Workflows

## Underwriting Workflow

1. Assign a risk grade from the applicant's profile. (Credit Risk Guru)
2. Decide approve or decline against the grade. (Credit Risk Guru)

Policies: Loan Amount Cap
```

A workflow heading can also carry three optional, prose-authored behaviours:

### Panels (multi-persona deliberation)

A `Pattern:` line convenes the workflow's personas as a **panel** instead of
reasoning single-voiced. The pattern is prose, not an enum — it goes into the
prompt verbatim:

```markdown
## Underwriting Workflow

Pattern: adversarial debate; the Credit Risk Guru has the last word

1. Assess the risk. (Credit Risk Guru)
2. Make the applicant's case. (Customer Advocate)
```

Who speaks next is a judgment: each turn a moderator reads the pattern and the
transcript and picks the next speaker — or concludes the panel early once every
persona has spoken and it has genuinely converged. Code guards the rest: only
listed personas speak, and a hard turn budget caps everything. A synthesis
concludes the panel into the one decision the pipeline continues with. Offline the
panel rotates deterministically and says it did not truly deliberate.

### Routes and retries (for journeys)

Control flow is authored in prose and judged at runtime when the workflow runs as
a [journey](OPERATIONS.md#journeys-durable-resumable-execution):

```markdown
## Underwriting Workflow

Routes: if the risk grade is C or worse, skip straight to the decline note.
Retries: retry a failed leg twice before giving up.

1. Band the profile and assign a risk grade. (Credit Risk Guru)
2. Prepare the approval paperwork. (Credit Risk Guru)
3. Write the decline note. (Credit Risk Guru)
```

- **Routes** — after each leg, a routing judgment reads the authored routes and
  the leg's outcome and chooses the next **authored** step (jump, continue, or
  conclude). The model never invents a step; a per-step revisit budget in code
  refuses runaway loops.
- **Retries** — a leg whose cycle raises is retried within the declared budget,
  every attempt on the record; exhaustion ends the journey `FAILED`.

### Contracts (typed deliverables)

A `### Deliverable` section directly beneath a workflow declares the typed facts
its decision must carry — one bullet per field as `name: what it means`:

```markdown
### Deliverable

- decision: exactly one of approve or decline
- risk grade: the letter grade from A to E the decision rests on
```

At runtime the model fills the fields from the prose decision and judges the
filling against those meanings (one hinted retry, then nonconforming data is
withheld, on the record). Conformant data travels as a `## Data` section in the
decision document.

---

## `process.md`

A process is a set of workflows that performs an action. The file's `#` title
names the runtime; each `##` heading is a process.

```markdown
# Credit Risk Runtime

## Underwrite Consumer Loan

Evaluates a consumer loan application end to end.

Workflows: Underwriting Workflow
```

---

## `policy.md`

A policy is a governance rule: a prose statement the model judges, with an
optional deterministic fallback and a scope. Policies are covered in full in the
[Governance guide](GOVERNANCE.md); the authoring shape is:

```markdown
# Policies

## Loan Amount Cap

The loan must not exceed $75,000.

Fallback: loan_amount <= 75000
Applies to: Underwriting Workflow
```

| Field | Meaning |
|---|---|
| prose statement | what the model judges the intent against |
| `Fallback:` | a deterministic expression enforced even with no model |
| `Applies to:` | `runtime` (everything), `tools`, or a named workflow |
| `Approval:` | `required` turns a hard block into a parkable [approval gate](GOVERNANCE.md#approval-gates) |
| `Approvers:` | an allow-list of who may waive the gate |
| `Escalate:` | e.g. `after 3 days` — a deadline for a parked gate |

---

## `tenant.md`

Optional. Declares the org a stack belongs to; absent, the stack uses the
`"default"` tenant.

```markdown
# Tenant

Org id: org_acme_prod
Fiscal year starts in April.
Timezone: Asia/Kolkata
```

Multi-tenant boundaries are enforced with a `Claim` at call time — see
[Governance → Tenancy](GOVERNANCE.md#tenancy-and-claims).

---

## `memory.md` (the operating strategy)

`memory.md` declares how the runtime operates, each section in plain English with
the few values the machinery needs extracted from the prose.

| Section | Declares |
|---|---|
| **Model Selection** | provider/model, the credential's env-var *name*, temperature |
| **Reasoning Audit Trail** | where the trail is logged (see [Operations](OPERATIONS.md#the-audit-trail)) |
| **Knowledge** | reference sources for RAG (below) |
| **Tools** | declared tools (`- name: what it does`) |
| **MCP** | declared MCP servers (`- name: what it provides`, `command`) |
| **Context History** | how many recent cycles stay verbatim before compression |
| **Cross-Session Data** | where memory/experience/adaptations persist |
| **Subagent Spawning** | whether subagents may spawn, and how many |
| **Skills Discovery** | guidance folded into relevance ranking |
| **Ontological Settings** | the vocabulary reasoning works with (`- term: meaning`) |
| **Sandbox** | an isolated workspace per instance (see [Governance](GOVERNANCE.md#sandbox)) |
| **Evolution** | governed self-modification policy (see [Governance](GOVERNANCE.md#evolution)) |
| **Pricing** | token prices, so usage records carry real dollars |

The ontology, declared tools/servers and discovery guidance are rendered into the
Reasoner's prompt, so the model reasons with your vocabulary and knows its
capabilities — *when* to use them stays a judgment, never a hardcoded rule.

### Model Selection

```markdown
## Model Selection

Reason with anthropic/claude-opus-4-8, reading the credential from
ANTHROPIC_API_KEY, at a temperature of 0.2. When the credential is absent, the
runtime stays on its deterministic fallback.
```

The binding only attaches when the credential actually resolves — a stack loaded
without keys degrades cleanly to the deterministic fallback. Use `openai/<model>`
plus an `api_base` for any OpenAI-compatible endpoint.

### Knowledge

Reference material is stacked one bullet per source — a path or glob resolved
relative to the stack, or a URL fetched over EAR's own HTTPS client:

```markdown
## Knowledge

- underwriting manual: `knowledge/underwriting-manual.md`
- market brief: https://example.com/brief.md, refetch weekly
```

Markdown sources are chunked into passages. Narrowing is native BM25, scored over
each passage's text *and* a one-line model-written gist (built once per corpus,
persisted to `.ear/index.md`, keyed by content hash). Per cycle the model judges
which passages a careful analyst would actually consult — choosing none is valid,
and it can only choose among real passages. What was consulted becomes evidence: a
`retrieval` trail record, citations, and a `## Sources` section in the decision.

### Tools and MCP

Tools and MCP servers stay **declared** in `memory.md`; binding a declaration to
an executable happens in code or by connecting the server. See
[Operations → Tools](OPERATIONS.md#tools-and-mcp) for the runtime side.

```markdown
## Tools

- amortization_calculator: computes a monthly payment from principal, rate and term

## MCP

- calc: a calculator server, command `python -m calc_server`
```

---

## The request/response boundary

You don't have to call `reason()` yourself. Drop intent documents in a folder and
let the `Exchange` answer them:

```
intents/<name>.md     #-title = the request, prose elaborates, ## Context bullets carry facts
decisions/<name>.md    the decision, explanation, evidence, and every policy judgment
```

```python
from ear import Exchange, load_runtime

runtime = load_runtime("examples/credit_risk_stack")
Exchange("examples/credit_risk_stack").run(runtime)   # answers every unanswered intents/*.md
```

`Exchange.run` is idempotent — intents whose decision already exists are skipped.
Context bullet values are coerced back to numbers/booleans; free-text values in
outbound documents are blockquoted so they can never be mistaken for structure.

---

## Where to go next

- **[Governance](GOVERNANCE.md)** — policies, approval gates, budgets, the trail.
- **[Operations](OPERATIONS.md)** — the Exchange, kernel, server, Kubernetes, tools, journeys.
- **[Concepts](CONCEPTS.md)** — the model behind the files.
