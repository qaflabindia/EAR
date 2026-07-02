# EAR — Enterprise Agentic Runtime

EAR is a Python package for building an enterprise agentic runtime, named
in plain English throughout. Prompts are stacked inside skills, skills are
stacked inside a persona, a persona is stacked into a workflow, workflows
are stacked into processes, policies are mapped onto those processes,
processes are orchestrated by the runtime, and the runtime reasons.

The user writes no code — they stack declarations in plain English and the
runtime reasons over them:

- narrate a **`Skill`** as a prompt,
- stack skills into a **`Persona`**,
- narrate ordered **steps** into a **`Workflow`** and delegate each to a persona,
- attach a **`Policy`** to govern the workflow (or the whole runtime).

```python
guru = Persona(name="Credit Risk Guru", instructions="Underwrite conservatively.")
guru.add_skill(Skill(name="risk_grade", prompt="Combine the score tier and DTI band into a grade A–E."))

workflow = Workflow(name="Underwriting Workflow")
workflow.add_step("Band the credit profile and assign a risk grade.", persona=guru)
workflow.add_step("Decide approve or decline against the grade.", persona=guru)
workflow.add_policy(Policy(name="Loan Amount Cap", statement="The loan must not exceed $75,000."))
```

For the intent at hand the runtime composes the workflow's ordered steps,
the personas they delegate to and the stacked skill prompts into one
assembled-capabilities block and reasons over it with the active LLM
(any [LiteLLM](https://github.com/BerriAI/litellm) provider — Anthropic,
OpenAI, Gemini, Bedrock, Ollama — selected by `ModelBinding`, never
hardcoded), and enforces the workflow's policies before it runs. So what
you stack is what the model actually reasons with. A deterministic Python
`handler` on a Skill stays available for the advanced case, but it is
optional, never required.

```text
Intent → Skill → Persona → Workflow → Process → Policy → Runtime → Reasoner
```

## Author the whole runtime in Markdown — no code at all

The stack above can be written entirely as natural-language markdown files
in one directory, and `load_runtime` assembles the Runtime from them:

```text
skills.md    prompts stacked into skills       (heading = skill, prose = prompt)
persona.md   skills stacked into personas      (prose = instructions, `Skills:` stacks by name)
workflow.md  steps stacked into workflows      (numbered steps; `(Persona Name)` delegates)
process.md   workflows stacked into processes  (prose = description, `Workflows:` stacks by name;
                                                the file's # title names the runtime)
policy.md    governance, risk and controls     (prose = the statement an LLM judges;
                                                `Fallback:` deterministic expression,
                                                `Applies to:` runtime or a workflow name)
memory.md    the operating strategy            (see below)
```

```python
from ear import Intent, load_runtime

runtime = load_runtime("examples/credit_risk_stack")
decision = runtime.reason(Intent(
    text="Underwrite a $20,000 consumer loan application",
    context={"loan_amount": 20000, "debt_to_income": 0.28, "credit_score": 742},
))
```

Cross-references are by name, case- and punctuation-insensitive, and an
unresolved reference fails loudly with the list of known names — nothing an
author writes is silently dropped. A workflow no process references is
wrapped in a process of its own rather than lost.

`memory.md` declares the runtime's **strategy**, each section in plain
English with the few values the machinery needs extracted from the prose:

```text
Context History       how many recent cycles stay verbatim before compression
Cross-Session Data    where Memory/Experience/Adaptations persist between sessions
                      (a SessionStore; restored on load, saved after every cycle)
Subagent Spawning     whether subagents may spawn, and how many (the Spawner
                      enforces the limit like the Governor enforces Policies)
Model Selection       provider/model (e.g. anthropic/claude-opus-4-8), the
                      credential's environment-variable NAME (never a key in the
                      file), temperature; the binding only attaches when the
                      credential actually resolves, so a stack loaded without
                      keys degrades to the deterministic fallback
Reasoning Audit Trail where every reasoning step is logged (see below)
MCP                   declared MCP servers  (- name: what it provides, via `command`)
Tools                 declared tools        (- name: what it does)
Skills Discovery      guidance the Discoverer folds into relevance ranking
Ontological Settings  the vocabulary reasoning works with (- term: meaning)
```

The ontology, declared tools/MCP servers and discovery guidance are rendered
into the Reasoner's prompt as the runtime's operating strategy, so the model
reasons with the enterprise's own vocabulary and knows its capabilities —
when to use them stays a natural-language judgment, never a hardcoded rule.
Subagents spawn via `runtime.spawn(persona, intent)`: a child runtime scoped
to one persona, sharing the parent's model and strategy but keeping its own
memory, with nested spawns counted against the same strategy budget.

### Markdown in, markdown out

Markdown is the system-native format on *both* sides of the boundary, not
just for authoring. Requests arrive as intent documents and decisions leave
as decision documents, paired by file stem through the `Exchange`:

```text
IN   intents/<name>.md    #-title = the request, prose elaborates it,
                          ## Context bullets carry the facts (values are
                          coerced back to numbers/booleans)
OUT  decisions/<name>.md  the decision, explanation, evidence and every
                          policy judgment with its rationale; a Policy
                          block is written as Status: BLOCKED, not raised
                          away -- a refusal is an outcome on the record
OUT  .ear/reasoning.md    the reasoning audit trail (append-only)
OUT  .ear/session.md      cross-session memory, restored through the same
                          Section parser the stack is authored with
```

```python
from ear import Exchange, load_runtime

runtime = load_runtime("examples/credit_risk_stack")
Exchange("examples/credit_risk_stack").run(runtime)   # answers every unanswered intents/*.md
```

`Exchange.run` is idempotent — an inbox, not a replay: intents whose
decision document already exists are skipped. `Exchange.respond(runtime,
intent_markdown)` is the same boundary as text-in/text-out. `Intent` itself
round-trips (`Intent.from_markdown` / `intent.to_markdown()`), and every
free-text value in an outbound document is blockquoted so it can never be
mistaken for document structure. The persistence codecs are picked by file
extension: `.md` is the default everywhere; declare a `.json`/`.jsonl` path
in memory.md if a machine pipeline needs it instead.

See `examples/credit_risk_stack/` for a complete six-file stack that loads
and reasons offline (deterministic fallbacks) or live (set the environment
variable named in its memory.md), and `examples/credit_risk_guru.ipynb` for
the full loop run end to end — authoring, reasoning, audit review and
prompt optimization — with nothing but natural language as input.

### The reasoning audit trail

`runtime.reasoning_log` (a `ReasoningLog`) records every judgment the
runtime makes, one stage-labelled record per judgment, so LLM reasoning is
reviewable after the fact and the stacked prompts can be optimized against
what the model actually reasoned with:

```text
intent        the cycle opened, with the intent and its context
policy        each Policy judgment, with the judge's rationale (pass AND block)
discovery     which processes were found relevant, and from what catalogue
selection     which candidates were chosen to run, when there was a choice
scheduling    the execution order chosen, when there was more than one workflow
delegation    which persona each undelegated step was assigned to, and why it
              was available -- the runtime completing the authoring, on record
deliberation  the Reasoner's decision, with the full stacked capabilities
              block and memory context -- the exact prompt material to review
recall        what was recalled from memory as relevant to this intent
retrieval     which knowledge passages were consulted, with citations
tool          each tool invocation: arguments, result, duration (failures too)
approval      a human verdict on a parked gate, or the park itself
usage         the cycle's accounting: model calls, tokens, cost, latency
explanation   the Explainer's prose and the evidence it rested on
audit         the auditor's assessment of whether the evidence supports
              the decision
adaptation    each newly distilled lesson, when the Adapter fires
```

Each record carries the model that produced it (`deterministic-fallback`
when no ModelBinding was active). Blocked cycles are logged too — a policy
violation is an audit event, not a gap in the record. Declare a "Reasoning
Audit Trail" section in memory.md and the runtime appends the trail to
disk after every cycle — readable markdown by default (`.ear/reasoning.md`,
one `## Cycle` section per cycle, cycle numbering continuing across
sessions), or JSONL when the declared path ends in `.jsonl`:

```python
print(runtime.reasoning_log.render())                      # the skim view
runtime.reasoning_log.for_stage("deliberation")[-1].inputs  # the full prompt material
```

### Tools — execution on the record

Tools stay *declared* in memory.md; the `ToolBinder` is where a
declaration meets an executable:

```python
runtime.bind_tool("amortization_calculator", monthly_payment)  # any callable
```

The stack remains the source of what exists — binding a name nothing in
the stack declares fails loudly, so code never grows the runtime a
capability the natural-language authoring doesn't show. Skills that carry
a Python handler bind automatically for the cycle's plan (an explicit
binding overrides). With bound tools present, deliberation becomes a DSPy
ReAct loop over them — *when* to call one is the model's judgment, within
the binder's iteration budget — and every invocation is a trail record
(stage `tool`: arguments, result, duration). A failing tool never breaks
the cycle: the failure is recorded and handed back to the model as text.
Declared-but-unbound tools stay context the model knows about, as before.
LangChain tools bind through
`ear.integrations.langchain_backend.bind_langchain_tool` (duck-typed —
`.name`/`.description`/`.invoke`/`.run`), behind a declared name like
everything else.

The deliberation engine itself is also a seam: attach a `backend` to the
`Deliberator` (anything with `deliberate(runtime, intent, plan, research)`
— e.g. a typed-agent adapter) and it replaces the Reasoner for that step
only, with the Governor, Decider/Validator, Contracts and the trail all
still applying — a backend never reasons off the record.

### Approval gates — human-in-the-loop governance, in markdown

A policy authored with `Approval: required` converts its hard block into a
parkable gate:

```markdown
## Large Loan Human Approval

Loan amounts above $50,000 must be approved by a human approver.

Fallback: loan_amount <= 50000
Approval: required
Applies to: Underwriting Workflow
```

When such a policy is violated, the cycle raises `ApprovalRequired`
(a `PermissionError`, so existing handlers keep working) and the Exchange
writes the decision document with `Status: PENDING APPROVAL`, naming the
policies awaiting a verdict and how to give one. A human releases the
cycle by dropping an approval document beside it (`approval.md`, or
`approvals/<name>.md` for `intents/<name>.md`):

```markdown
# Approval -- Underwrite a $60,000 loan

Verdict: approved
Approver: lakshminarasimhan.santhanam@gigkri.com

> Reviewed the exception; the collateral covers it.
```

The next `Exchange.run` finishes the cycle: approved passes the gate — on
the record, with the approver's name and note (trail stage `approval`, an
`## Approval` section in the final document) — and rejected blocks it like
any violation. The split is deliberate: the model judges *whether* the
gate triggers, only a human can *waive* it, and code enforces both. A
hard (ungated) violation always wins over a pending gate, an unreadable
verdict fails loudly rather than leaving the cycle silently parked, and a
parked cycle is fully accounted (usage, trail) but writes no memory —
nothing was decided yet.

### Knowledge — RAG with citations, declared in natural language

Reference material is stacked in memory.md under a Knowledge section, one
bullet per source (paths or globs, resolved relative to the stack; a
source that matches nothing fails loudly at load):

```markdown
## Knowledge

- underwriting manual: `knowledge/underwriting-manual.md`
```

Markdown sources are chunked into passages by the same Section parser the
stack is authored with. Per cycle, the `Librarian` narrows the corpus
structurally and the model judges which passages a careful analyst would
actually consult — choosing none is a valid judgment, and it can only
choose among real passages. What was consulted is first-class evidence:
a `retrieval` trail record, `citations` in the Evidence, a `## Sources`
section in the decision document, and the retrieved text reaches the
Reasoner framed as reference material, never as instructions. A LlamaIndex
retriever plugs into the same seam
(`ear.integrations.llamaindex_backend.LlamaIndexRetriever`,
`pip install 'ear[rag]'`) — the platform narrows, EAR's model still judges
and cites.

### Observability — exporters off the trail, usage on every cycle

The ReasoningLog is the native trace; observability is an exporter, never
a second instrumentation path. Attach anything with `export(record)` to
`runtime.reasoning_log.exporters` — an exporter that raises never breaks a
cycle (failures stay visible in `export_errors`), and the file on disk
remains the canonical record. `pip install 'ear[observability]'` provides
`OpenTelemetryExporter` (one trace per cycle, one span per record), which
covers Langfuse and Phoenix through their native OTLP ingestion — endpoint
and credentials come exclusively from the standard `OTEL_EXPORTER_OTLP_*`
environment variables. Declare it in prose: an audit-trail section that
names OpenTelemetry/Langfuse/Phoenix gets the exporter attached at load.

Every cycle also closes with a `usage` record — model calls, tokens,
approximate cost and wall-clock latency, read from the bound LM's own call
history — written for blocked cycles too: a refusal costs whatever it
cost.

### Contracts — typed deliverables, declared in natural language

A workflow may declare what its decision must *deliver* with a
`### Deliverable` section directly beneath it in workflow.md — prose
describing the deliverable, one bullet per field as `name: what it means`:

```markdown
### Deliverable

- decision: exactly one of approve or decline
- risk grade: the letter grade from A to E the decision rests on
```

At runtime the model fills the fields from the prose decision (a DSPy
signature is built dynamically from the authored meanings) and then judges
the filling against those meanings — one hinted retry, then nonconforming
data is withheld, on the record (trail stage `contract`). Conformant data
travels as a `## Data` section in the decision document, typed by the same
`coerce` codec as intent context, and parses back via
`ear.exchange.data_from_decision_document`. With no model bound nothing is
fabricated: the skip itself is a trail record.

### Evaluation — the Examiner and markdown-native evals

An evaluation is one markdown file in an `evaluations/` directory: an
ordinary intent document plus an `## Expected` section — prose criteria
(a blocked refusal can itself be the expectation) and/or bullets of
`field: value` the delivered Data must carry:

```python
from ear import Examiner
examination = Examiner().examine(runtime, "examples/credit_risk_stack/evaluations")
assert examination.passed          # the CI regression gate
```

With a model, `JudgeDecisionQuality` grades each outcome with a rationale;
offline, only the field bullets are checked structurally and prose-only
criteria are reported **ungraded** rather than faked. Verdicts land on the
trail (stage `evaluation`) and in `evaluations/report.md`.

### Optimization — the trail is the training corpus

```python
from ear import Optimizer
optimizer = Optimizer()
trainset = optimizer.trainset_from_trail(".ear/reasoning.md")   # or .jsonl
labels   = optimizer.verdicts_from_documents("decisions/")      # ## Review + Verdict: lines
optimizer.optimize_from_trail(runtime, ".ear/reasoning.md")     # trail -> GEPA in one call
```

Deliberation records become `dspy.Example`s (the exact intent, context and
stacked capabilities the model reasoned with); a reviewer labels a decision
document by adding a `## Review` section with a `Verdict:` line; and the
metric grades candidates with the same `JudgeDecisionQuality` the Examiner
uses — evaluation and optimization share one notion of quality.

Every judgment is made dynamically at runtime, in natural language against
a live LLM (via [DSPy](https://github.com/stanfordnlp/dspy)) — `Policy`
compliance, `Discoverer` relevance ranking, `Selector` choice among
candidates, `Scheduler` ordering, `Delegator` step delegation, the
`Reasoner`'s decision, the `Explainer`'s prose — each with a
deterministic, dependency-free fallback so the package stays fully usable
and testable with no LLM configured at all, and each judgment written to
the `ReasoningLog`. Only mechanics with no judgment content stay plain
Python — the `Composer`'s flattening, the `Validator`'s shape checks, and
enforcement itself: **the LLM judges; code enforces and records.**

`ModelBinding` is the LLM provider binding — model, credentials, call
parameters — read from an environment variable, never hardcoded.
`Runtime` activates its `ModelBinding` before handing the `Intent` to the
rest of the pipeline, so reasoning runs against a real model instead of
the dependency-free default whenever one is configured.

`Evidence`, `Memory`, `Experience` and `Adaptation` are the runtime's
memory, kept as four distinct layers rather than blurred into one:

```text
Evidence    → why this decision was made
Memory      → what happened
Experience  → the pattern across repeated Memory entries
Adaptation  → how future behaviour should change
```

`Runtime` writes an `Evidence`-backed entry to `Memory` after every
`reason()` call, folds it into `Experience`, and surfaces all three layers
back to the `Reasoner` on the next call, so memory compounds across
cycles instead of resetting each time.

`Runtime.reason()` itself runs through one further named pipeline of
operations — each its own class instead of logic folded into one method:

```text
Governor     → govern       enforce Policy gates (LLM-judged, safe-eval fallback)
Initializer  → initialize   activate the ModelBinding
Discoverer   → discover     find Processes relevant to the Intent (LLM-ranked, keyword fallback)
Selector     → select       choose among candidates (LLM-chosen, dedupe fallback)
Composer     → compose      assemble their Workflows into a plan
Scheduler    → schedule     order the plan (LLM-ordered, composition-order fallback)
Delegator    → delegate     assign undelegated steps to personas (LLM-judged at runtime)
Orchestrator → orchestrate  coordinate a cycle's execution end to end
Executor     → execute      run the cycle's Performer action
Performer    → perform      deliberate, decide, validate
Deliberator  → deliberate   reason via the Reasoner
Decider      → decide       commit to one decision
Validator    → validate     reject a malformed decision
Recaller     → remember     recall relevant Memory as evidence (LLM-recalled, full-window fallback)
Librarian    → research     retrieve relevant Knowledge with citations (LLM-judged, structural fallback)
Explainer    → explain      render why a decision was reached (LLM-written, f-string fallback)
Auditor      → audit        inspect evidence for compliance (LLM-assessed, flag fallback)
Memory       → store memory what happened (overflow compressed by the active LM when bound)
Learner      → learn        fold the cycle into Experience
Adapter      → adapt        periodically distill a new Adaptation (LLM-distilled when bound)
```

`Governor` raises `PermissionError` and stops the cycle before anything
else runs if a `Policy` is violated. `Adapter` only distills a new
`Adaptation` every `adapt_every` observed cycles (default 5), not on every
single one. `Evolver` and `Optimizer` are structural, dev-time operations
on a `Skill` or `Persona` — evolving or optimizing isn't part of running
a cycle, so they sit outside this pipeline and are called directly:

```python
runtime.evolver.evolve(skill, evaluator="path/to/evaluator.py")            # openevolve
trainer = runtime.optimizer.optimize(config="skillopt.yaml", adapter=my_env)
runtime.optimizer.apply(persona, "skill-name", trainer.train())
```

## Roadmap

[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) maps each
best-in-class capability that exists in isolation elsewhere — DSPy+GEPA
optimization, Instructor structured outputs, LangSmith/Phoenix evaluation,
Langfuse observability, LlamaIndex RAG, Temporal durable workflows and
approval gates, LangGraph graphs, AutoGen multi-agent, PydanticAI typed
agents, LangChain tools — onto an existing EAR pipeline seam as an
optional backend, phased so the authored stack never changes shape.

## Install

```bash
pip install -e .
```

DSPy is included as the reasoning-programming dependency.

Two optional extras add evolutionary and reflective skill optimization:

```bash
pip install -e '.[evolve]'    # openevolve — AlphaEvolve-style evolutionary coding
pip install -e '.[skillopt]'  # skillopt   — Microsoft SkillOpt's ReflACT training loop
```

## Minimal example

```python
from ear import Intent, Policy, Process, Runtime, Workflow

runtime = Runtime(name="Procurement-Runtime")

runtime.add_policy(Policy(
    name="PO Approval Policy",
    statement="The purchase amount must not exceed the approver's approval limit.",
    fallback_expression="purchase_amount <= approval_limit",
))

process = Process(name="Create Purchase Order")
process.add_workflow(Workflow(name="Procurement Workflow"))
runtime.add_process(process)

result = runtime.reason(Intent(
    text="Create PO for laptops under approved budget",
    context={"purchase_amount": 4000, "approval_limit": 10000},
))
print(result)
```

Without a `ModelBinding` attached, `Runtime.reason` falls back to a
deterministic summary of which processes the `Intent` cleared, so the
example above runs with no LLM credentials. Give the runtime a real mind:

```python
from ear import ModelBinding

runtime.model_binding = ModelBinding(provider="anthropic", model="claude-opus-4-8")
# reads ANTHROPIC_API_KEY from the environment -- never hardcode a key

result = runtime.reason(Intent(text="Create PO for laptops under approved budget"))
```

`Runtime.reason` activates `runtime.model_binding` (configuring DSPy's LM)
before any judgment-laden stage runs, so `Policy`, `Discoverer`, the
`Reasoner` and the `Explainer` all reason against the same live model.

## Memory: Evidence, Memory, Experience and Adaptation

AI systems routinely conflate four distinct concerns: *why* a decision was
made, *what* happened, the *pattern* across repeated executions, and *how*
behaviour should change as a result. EAR keeps each as its own layer so
one never silently stands in for another.

`runtime.memory` is a two-layer memory that keeps context bounded as
history grows:

- **`working`** — the most recent cycles, kept verbatim (bounded by
  `capacity`, default 20).
- **`compressed`** — once `working` overflows, the oldest entries are
  rolled into one summary string per overflow event. Pass an activated LM
  (`runtime.model_binding.lm`) to `memory.compress(summarizer=...)` for an
  LLM-written summary instead of the deterministic digest used by default.

```python
runtime.memory.capacity = 50                          # raise/lower the verbatim window
runtime.memory.compress(summarizer=runtime.model_binding.lm)  # force an early, LLM-written compression
print(runtime.memory.context_window())                # compressed history + recent working entries
```

`runtime.experience` aggregates `Memory` entries into decision counts and
the evidence seen along the way — the pattern an `Adaptation` is then
distilled from, without yet drawing a conclusion:

```python
print(runtime.experience.summary())              # ranked decision counts across cycles
print(runtime.experience.most_common_decision())  # ("approved", 7), etc.
```

`runtime.adaptations` (an `AdaptationBank`) distills durable lessons out of
`Experience` — standing impressions that bias future reasoning rather than
raw history:

```python
learned = runtime.adaptations.learn_from(runtime.experience, summarizer=runtime.model_binding.lm)
print(learned.insight)
```

On the next `reason()` call, the `Reasoner` pulls in
`runtime.memory.context_window()`, `runtime.experience.summary()` and any
relevant `runtime.adaptations` impressions and folds them into the prompt
as three distinct sections, so memory, experience and learned adaptation
each influence the next decision without blurring together.

## Package layout

```text
ear/
  intent.py        Intent        — prompt / resolved request that starts a reasoning cycle
  skill.py         Skill         — a stacked prompt (a capability), reasoned over by the runtime; no handler code required
  persona.py       Persona       — a stack of Skills plus standing instructions
  step.py          Step          — one narrated instruction in a Workflow, delegated to a Persona
  workflow.py       Workflow      — an ordered list of Steps (each delegated to a Persona), governed by its own Policies
  approval.py      Approval      — a human's verdict on a parked cycle; ApprovalRequired parks it
  tool_binder.py   ToolBinder    — declared tools meet executables; every invocation on the trail
  contract.py      Contract      — a workflow's Deliverable: fields with plain-English meanings, extracted and judged at runtime
  examiner.py      Examiner      — examine: run markdown-native evaluations and grade them, honestly offline
  knowledge.py     Knowledge     — the declared reference corpus, chunked through the Section parser
  librarian.py     Librarian     — research: retrieve relevant Knowledge with citations, on the record
  process.py       Process       — a stack of Workflows that performs an action
  policy.py        Policy        — governance rule, judged in natural language with a safe-eval fallback; attaches runtime-wide or to a Workflow
  runtime.py       Runtime       — runs every cycle through the full operation pipeline below
  model_binding.py ModelBinding  — LLM provider binding (model, credentials, params -> a DSPy LM)
  evidence.py      Evidence      — why this decision was made
  memory.py        Memory        — persistent memory (working + compressed layers)
  experience.py    Experience    — the pattern aggregated from repeated Memory entries
  adaptation.py    Adaptation    — learned adaptations distilled from Experience
  reasoner.py      Reasoner      — the deliberation step (DSPy-backed, with a GEPA optimization hook)
  signatures.py    DSPy signatures shared across the LLM-judged stages
  governor.py      Governor      — govern: enforce Policy gates
  initializer.py   Initializer   — initialize: activate the ModelBinding
  discoverer.py    Discoverer    — discover: find Processes relevant to an Intent
  selector.py      Selector      — select: choose among candidates (LLM-chosen, dedupe fallback)
  delegator.py     Delegator     — delegate: assign undelegated steps to personas at runtime
  composer.py      Composer      — compose: assemble selected processes' Workflows into a plan
  scheduler.py     Scheduler     — schedule: order the composed plan
  orchestrator.py  Orchestrator  — orchestrate: coordinate a cycle's execution end to end
  executor.py      Executor      — execute: run the cycle's Performer action
  performer.py     Performer     — perform: chain Deliberator -> Decider -> Validator
  deliberator.py   Deliberator   — deliberate: reason via the Reasoner
  decider.py       Decider       — decide: commit to one final decision
  validator.py     Validator     — validate: checker layer for every maker stage's output
  recaller.py      Recaller      — remember: recall Memory context as evidence
  explainer.py     Explainer     — explain: render why a decision was reached
  auditor.py       Auditor       — audit: inspect evidence for compliance
  learner.py       Learner       — learn: fold a cycle into Experience
  adapter.py       Adapter       — adapt: periodically distill a new Adaptation
  evolver.py       Evolver       — evolve: transform a Skill's source (openevolve, dev-time)
  optimizer.py     Optimizer     — optimize: trail -> GEPA trainsets, reviewer verdicts, the shared
                                   quality metric; plus the SkillOpt loop (dev-time)
  section.py       Section       — the shared structural parser for stacked markdown files
  loader.py        Loader        — load_runtime: stack skills.md/persona.md/workflow.md/
                                   process.md/policy.md/memory.md into a Runtime
  strategy.py      Strategy      — the memory.md operating strategy, read from plain English
  exchange.py      Exchange      — the markdown boundary: intents/*.md in, decisions/*.md out
  reasoning_log.py ReasoningLog  — the reasoning audit trail (markdown by default, JSONL optional)
  session_store.py SessionStore  — cross-session data (markdown by default, JSON optional)
  spawner.py       Spawner       — spawn subagent runtimes, bounded by the strategy
  tool.py          Tool          — a tool declared in plain English, surfaced to reasoning
  mcp_server.py    McpServer     — an MCP server declared in plain English
  ontology.py      Ontology      — the term→meaning vocabulary folded into reasoning
  integrations/
    evolve_backend.py    openevolve — evolve a Skill's source against an evaluator
    skillopt_backend.py  skillopt   — train a Persona's skill document with ReflACT
    otel_backend.py      OpenTelemetryExporter — the trail as OTLP spans (Langfuse/Phoenix/any)
    llamaindex_backend.py LlamaIndexRetriever  — a LlamaIndex retriever on the Librarian's seam
    langchain_backend.py bind_langchain_tool   — a LangChain tool behind a declared tool name
```
</content>
