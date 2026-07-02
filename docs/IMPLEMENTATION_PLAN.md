# EAR Implementation Plan — one runtime, best-in-class capabilities

The capabilities enterprises need today exist **in isolation, on different
platforms**: prompt optimization in DSPy/GEPA, typed agents in PydanticAI,
durable workflows in Temporal, agent graphs in LangGraph, multi-agent
conversation in AutoGen, RAG in LlamaIndex, structured outputs in
Instructor, evaluation in LangSmith/Phoenix, observability in Langfuse,
tool ecosystems in LangChain. Adopting them all today means ten mental
models, ten config formats, ten places governance can leak.

EAR's position: **the stack the user authors never changes** — six
natural-language markdown files, intents in, decisions out, every judgment
on the audit trail — and each best-in-class capability plugs in *behind*
an existing pipeline seam as an optional backend. The platforms compete on
execution; EAR owns authoring, governance and the record.

```text
Category              Best-in-class        EAR seam it plugs into              Extra
--------------------  -------------------  ----------------------------------  --------------
Prompt Optimization   DSPy + GEPA          Reasoner / Optimizer / ReasoningLog  (core today)
Structured Outputs    Instructor           Decider / Validator / Exchange       [structured]
Evaluation            LangSmith / Phoenix  Examiner (new) / ReasoningLog        [evals]
Observability         Langfuse             ReasoningLog exporters               [observability]
RAG                   LlamaIndex           Recaller / memory.md Knowledge       [rag]
Workflow Runtime      Temporal             Orchestrator / Executor              [temporal]
Enterprise Governance Temporal             Governor approval gates              [temporal]
Stateful Agent Graphs LangGraph            Composer/Scheduler -> graph compile  [langgraph]
Multi-Agent           AutoGen              Spawner / Workflow personas          [autogen]
Typed Agents          PydanticAI           Deliberator / Persona                [typed]
Integrations          LangChain            Tool & McpServer execution binding   [langchain]
```

---

## 1. Where we stand (the foundation this plan builds on)

- **Authoring**: prompts → skills.md → persona.md → workflow.md →
  process.md, governed by policy.md, strategized by memory.md; loaded by
  `Loader`/`load_runtime` through one deterministic `Section` parser.
- **I/O**: markdown-native both directions (`Intent.from_markdown`,
  `Exchange`: intents/*.md → decisions/*.md, BLOCKED refusals included).
- **Runtime**: 19-stage named pipeline; **11 judgment stages are already
  LLM-at-runtime** (policy judgment, discovery, selection, scheduling,
  delegation, deliberation, recall, explanation, audit, compression,
  adaptation), each with a deterministic fallback and a `ReasoningLog`
  record.
- **Memory**: Evidence / Memory / Experience / Adaptation layers,
  cross-session `SessionStore` (.md), subagent `Spawner` with budgets.
- **Optimization hooks**: GEPA on the Reasoner, openevolve (`[evolve]`),
  SkillOpt (`[skillopt]`).

## 2. Non-negotiables — every integration obeys these

1. **The authoring surface stays natural language.** A new capability may
   add a *section* to memory.md or a *field* to a stack file; it may never
   add a config format, a YAML file, or required Python authoring.
2. **The LLM judges; code enforces and records.** Backends can execute and
   reason, but policy blocks, budgets, validation and audit writing stay
   in EAR code, whatever platform runs underneath.
3. **Deterministic fallback always.** Every capability degrades cleanly
   when its platform (or a model) is absent — the core package keeps
   exactly one hard dependency (dspy).
4. **Everything lands on the trail.** A backend that reasons, retrieves,
   converses or executes writes stage-labelled `ReasoningLog` records; a
   capability that can't be audited doesn't ship.
5. **Adapters, not bindings.** One file per platform under
   `ear/integrations/`, imported lazily, installed via extras
   (`pip install ear[temporal]`), tested behind skip-markers. EAR classes
   never subclass platform classes; platform objects are built *from*
   EAR's stack at the seam.
6. **Secrets by environment-variable name only** — in memory.md prose and
   in adapter config alike.

---

## 3. Capability plans

### 3.1 Prompt Optimization — DSPy + GEPA (deepen what's core)

*Today*: `Reasoner.optimize_with_gepa(trainset, metric)` exists but the
author must hand-build the trainset.

**Build**
- `Optimizer.trainset_from_trail(path)` — parse `reasoning.md`/`.jsonl`
  deliberation records into `dspy.Example`s (intent, context,
  capabilities → decision). The audit trail becomes the training corpus.
- Markdown-native labels: a reviewer adds a `Verdict:` field
  (correct / incorrect, with prose) to a copy of any decision document;
  the verdicts become the GEPA metric.
- Per-signature registry: optimize `JudgePolicyCompliance`,
  `DiscoverRelevantProcesses` etc. against labelled trails, not just the
  Reasoner. Optimized programs persist under `.ear/programs/` (DSPy
  save/load) and are reloaded by the Loader.

**Done when** an optimization run measurably improves a held-out markdown
eval set (see 3.3), end to end from trail → trainset → GEPA → reloaded
program, with zero hand-written Python examples.

### 3.2 Structured Outputs — Instructor

*Seam*: `Decider` / `Validator` / decision documents.

**Build**
- **Contract authoring in the stack**: a `Deliverable` section on a
  workflow or process — bullets of `field: what it means` — parsed by the
  existing Section parser into a `Contract` (name, typed fields inferred
  from the prose the way `coerce` already types context values).
- Primary path: DSPy typed signatures generated from the Contract (we
  already run DSPy). `[structured]` extra adds Instructor for
  function-calling-style extraction where teams standardize on it.
- `Validator.validate_against(contract)` — the checker enforces the
  schema; a nonconforming decision is re-asked once, then rejected.
- Decision documents gain a `## Data` section (bullets, md round-trip via
  the Section parser) so the Exchange stays markdown-only while carrying
  machine-usable fields.

**Fallback**: no Deliverable section → prose decisions exactly as today.
**Done when** an intent yields a decision.md whose `## Data` parses back
to the declared typed fields, offline and live, and a schema violation is
visibly rejected on the trail.

### 3.3 Evaluation — LangSmith / Phoenix

*Seam*: new `Examiner` class (English-named, like every stage) + the
ReasoningLog.

**Build**
- **Markdown-native eval suite**: `evaluations/<name>.md` — an intent
  document plus an `## Expected` section (prose criteria and/or expected
  Data fields). No JSON datasets to maintain.
- `Examiner.examine(runtime, directory)` runs the suite, grades each
  decision (new `JudgeDecisionQuality` signature: expected criteria vs
  actual decision, verdict + rationale; deterministic field-equality
  fallback for Data contracts), writes `evaluations/report.md`.
- CI regression gate: `pytest`-invokable, threshold declared in prose in
  the report header or memory.md.
- `[evals]` extra: exporters pushing runs/datasets/verdicts to LangSmith
  and Phoenix, and importing their annotations back as `Verdict:` labels
  (feeding 3.1 — evaluation and optimization share one loop).

**Done when** CI fails on a regressed eval and a LangSmith/Phoenix project
shows the same runs the local report.md shows.

### 3.4 Observability — Langfuse

*Seam*: `ReasoningLog` — it is already the native trace; observability is
an **exporter**, never a second instrumentation path.

**Build**
- `TraceExporter` protocol: `export(record)` / `flush()`; attach any
  number to a ReasoningLog.
- `[observability]` extra: Langfuse adapter (cycle → trace, stage → span,
  model/tokens/cost as attributes) and an OpenTelemetry OTLP adapter for
  vendor-neutral pipelines.
- Extend `ReasoningRecord` with optional `tokens`/`cost`/`latency_ms`
  captured from DSPy usage metadata — visible in the md trail too.
- Declared in memory.md's Reasoning Audit Trail section in prose
  ("also export the trail to Langfuse; keys from LANGFUSE_PUBLIC_KEY /
  LANGFUSE_SECRET_KEY").

**Done when** one reasoned cycle appears simultaneously in
`.ear/reasoning.md` and as a Langfuse trace with identical stage content,
and the md trail remains the canonical record when the exporter is down.

### 3.5 RAG — LlamaIndex

*Seam*: `Recaller` (recall is already an LLM judgment) + a new
**Knowledge** section in memory.md.

**Build**
- Authoring: `## Knowledge` in memory.md — bullets of sources in prose
  (`- underwriting manual: docs/underwriting/*.md`,
  `- product sheets: https://...`).
- `[rag]` extra: LlamaIndex builds/refreshes indices at load (persisted
  under `.ear/index/`); at recall time the Recaller retrieves passages
  relevant to the intent *alongside* Memory recall.
- Citations are first-class: retrieved passages land in
  `Evidence.sources["citations"]`, in the decision document (`## Sources`)
  and on the trail (stage `retrieval`, with source + snippet).
- Retrieval never bypasses governance: retrieved text is context for
  deliberation, not instructions — flagged as untrusted in the prompt
  the same way external content should be.

**Fallback**: no `[rag]` or no Knowledge section → memory-only recall
(today's behavior). **Done when** a decision cites the manual passage
that drove it, and removing the source changes the trail visibly.

### 3.6 Workflow Runtime — Temporal

*Seam*: `Orchestrator` / `Executor` — the pipeline's execution shell
becomes pluggable; stages stay the units.

**Build**
- `[temporal]` extra, `ear/integrations/temporal_backend.py`:
  `Runtime.reason` as a Temporal **workflow**, each pipeline stage as an
  **activity** (LLM calls only in activities — Temporal replay must never
  re-fire a model call), retry/timeout policies per stage, ReasoningLog
  flush as its own activity so the trail survives crashes exactly once.
- SessionStore checkpoints ride workflow state; long cycles survive
  worker restarts and resume at the stage boundary.
- The md contract is unchanged: an Exchange intent starts a durable
  cycle; the decision document is written by the final activity.

**Done when** killing the worker mid-cycle produces, after restart, one
completed cycle, one decision document and one uncorrupted trail.

### 3.7 Enterprise Governance — Temporal (approval gates on top of 3.6)

*Seam*: `Governor` + policy.md.

**Build**
- Authoring: a policy gains `Approval: required` (or prose equivalent —
  "decisions over $50,000 require a human approver") → the Governor emits
  an approval-pending outcome instead of pass/block.
- On Temporal: the cycle parks on a signal; approve/reject arrives as a
  Temporal signal (CLI + a markdown-native option: dropping
  `approvals/<cycle>.md` with `Verdict: approved` into the Exchange).
- The decision document is written with `Status: PENDING APPROVAL`, then
  finalized; approver identity and verdict land on the trail (stage
  `approval`) and in Evidence.
- Without `[temporal]`: synchronous fallback — the Exchange writes
  PENDING APPROVAL and a later `Exchange.run` completes the cycle when
  the approval document appears. Governance never depends on the platform.

**Done when** a capped intent produces a parked cycle that a human
approval document releases, on both backends, fully on the record.

### 3.8 Stateful Agent Graphs — LangGraph

*Seam*: the composed plan (Composer/Scheduler/Delegator output) — a stack
*is* a graph; LangGraph gets it as a compile target.

**Build**
- `[langgraph]` extra: `compile_to_graph(runtime)` → `StateGraph`:
  one node per workflow step (persona-scoped deliberation), edges from
  the Scheduler's order, conditional edges where the author writes
  branching steps in prose ("If the grade is D or E, skip to the customer
  note"), LangGraph checkpointer backed by `SessionStore`.
- The reverse direction: `runtime.as_node()` exposes a whole EAR runtime
  as a single LangGraph node for teams already on LangGraph — governance
  and trail intact inside the node.

**Done when** the credit-risk stack compiles to a runnable LangGraph app
whose checkpoints restore through SessionStore, with the trail identical
to native execution.

### 3.9 Multi-Agent — AutoGen

*Seam*: `Spawner` (budgets stay EAR's) + multi-persona workflows.

**Build**
- Authoring: a workflow section field in plain English —
  `Pattern: debate` / `maker-checker` / `group` — or prose steps like
  "Have the Credit Risk Guru and the Fraud Analyst debate the marginal
  case; the Compliance Officer arbitrates."
- `[autogen]` extra: personas become AutoGen agents (instructions →
  system message, stacked skills → capabilities), the pattern maps to a
  GroupChat/round-robin; the full transcript lands on the trail (stage
  `conversation`) and the outcome flows into the Decider as one
  deliberation.
- Spawner budget = hard cap on agents and turns; the Governor's policies
  gate the *outcome*, exactly as for a single persona.

**Fallback**: sequential per-step persona reasoning (today's behavior).
**Done when** a debate-pattern workflow yields a decision whose trail
shows the conversation, within budget, blockable by policy.

### 3.10 Typed Agents — PydanticAI

*Seam*: `Deliberator` — the deliberation backend becomes pluggable.

**Build**
- `[typed]` extra: build a PydanticAI `Agent` from a Persona
  (instructions → system prompt, prompt-skills → instruction block,
  handler-skills → typed tools), `result_type` from the workflow's
  Contract (3.2), model from the same `ModelBinding` (never a second
  model config).
- `Deliberator.backend` selection declared in memory.md prose ("deliberate
  through typed agents") with the DSPy path as default and fallback.
- Retries/validation errors surface as trail records, and the Validator
  still checks the result — the backend does not get to self-certify.

**Done when** the same stack runs unchanged on both deliberation backends
and produces contract-valid decisions on each.

### 3.11 Integrations — LangChain (and executable tools generally)

*Seam*: `Tool` / `McpServer` — today declarative; this gives them an
**execution binding** without changing the authoring.

**Build**
- `ToolBinder` registry: resolves a declared Tool/McpServer to an
  executable — a LangChain community tool, an MCP client (stdio/HTTP
  from the declared `command`/url), or a Python callable — matched by
  name; unbound tools remain declarative context, exactly as today.
- Deliberation gains tool use: a DSPy ReAct (or PydanticAI tools via
  3.10) program over the bound toolset; **every invocation** is a trail
  record (stage `tool`: tool, arguments, result, duration) and tool use
  is subject to policy (a policy can forbid a tool in prose).
- Reverse adapters: EAR runtime as a LangChain Runnable; a LangChain
  tool usable as a Skill handler.

**Done when** the amortization_calculator declared in memory.md actually
computes during deliberation, on the record, and denying it by policy
blocks the call.

---

## 4. Packaging & dependency policy

```toml
[project.optional-dependencies]
structured   = ["instructor>=1.4"]
evals        = ["langsmith>=0.1", "arize-phoenix-otel>=0.6"]
observability= ["langfuse>=2.50", "opentelemetry-sdk>=1.25"]
rag          = ["llama-index-core>=0.11"]
temporal     = ["temporalio>=1.6"]
langgraph    = ["langgraph>=0.2"]
autogen      = ["autogen-agentchat>=0.4"]
typed        = ["pydantic-ai>=0.0.30"]
langchain    = ["langchain-core>=0.3"]
all          = [everything above]        # plus existing: evolve, skillopt, dev
```

- Core dependency remains **dspy only**. Every adapter is one module in
  `ear/integrations/`, lazily imported, with a `RuntimeError` naming the
  extra to install when missing.
- Version policy: floor-pin (`>=`), adapter test suites marked
  `@requires_<platform>`, run in a CI matrix so ecosystem churn breaks a
  matrix cell, never the core.

## 5. Phasing

| Phase | Weeks | Ships | Exit criteria |
|---|---|---|---|
| **1 — Deepen the core** | 1–3 | Contracts/structured outputs (3.2), trail→GEPA trainsets (3.1), Examiner + md evals (3.3) | eval suite in CI; optimization run improves held-out evals; Data sections round-trip |
| **2 — Measure & retrieve** | 4–6 | Trace exporters (3.4), LangSmith/Phoenix export (3.3), Knowledge/RAG recall (3.5) | one cycle visible in trail + Langfuse identically; decisions carry citations |
| **3 — Durable enterprise** | 7–10 | Temporal backend (3.6), approval gates both backends (3.7) | mid-cycle crash recovery test green; human approval releases a parked cycle |
| **4 — Ecosystem** | 11–14 | ToolBinder + LangChain (3.11), AutoGen patterns (3.9), PydanticAI backend (3.10), LangGraph compile (3.8) | credit-risk stack runs unchanged on every backend with equivalent trails |

Order rationale: Phase 1 makes quality measurable before anything else
changes; Phase 2 makes it observable; Phase 3 makes it durable; Phase 4
widens execution — each phase is independently shippable and the stack
files never change shape.

## 6. Testing & CI

- Keep the two-tier pattern: offline (fallbacks, no key) / live
  (`ANTHROPIC_API_KEY`-gated), now × extras matrix (each adapter suite
  skips cleanly without its platform).
- **Trail-equivalence tests** are the integration contract: the same
  stack + intent must produce stage-equivalent ReasoningLogs on native,
  Temporal, LangGraph and typed backends.
- Crash/chaos test for Temporal (kill worker mid-cycle), budget tests for
  AutoGen (turn/agent caps), injection test for RAG (retrieved text
  attempting to override policy must still be blocked).
- The Examiner suite (3.3) runs as the regression gate from Phase 1 on.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Dependency sprawl / conflicts | extras + lazy imports; core stays dspy-only; CI matrix isolates breakage |
| Trail fidelity across backends | single ReasoningLog writer path; flush as its own Temporal activity; trail-equivalence tests |
| Temporal replay re-firing LLM calls | LLM calls confined to activities, never workflow code |
| Structured outputs breaking md-only I/O | `## Data` rendered/parsed by the same Section parser; JSON never crosses the Exchange |
| Retrieved content injecting instructions (RAG) | retrieved text marked untrusted context; policies judged before and after; injection test in CI |
| Multi-agent cost blowups | Spawner budgets enforced in code (turns, agents, spawns); budgets declared in memory.md prose |
| Ecosystem API churn | adapters are thin, one file each, floor-pinned, matrix-tested |
| Scope creep into a framework-of-frameworks | non-negotiables §2; anything that adds authoring surface beyond md prose is rejected |

## 8. Compatibility & versioning

- All of the above is additive: v0.x stacks load unchanged; new sections
  and fields are optional; defaults preserve current behavior exactly.
- Target: Phase 1–2 → **0.2.0**, Phase 3 → **0.3.0**, Phase 4 → **0.4.0**;
  1.0 when the trail-equivalence contract holds across all backends for
  two consecutive minor releases.
