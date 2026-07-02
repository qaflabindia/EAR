# EAR Native Plan — best-in-class capabilities, built from scratch

The capabilities enterprises need exist in isolation on different
platforms: prompt optimization in DSPy+GEPA, typed agents in PydanticAI,
durable workflows in Temporal, agent graphs in LangGraph, multi-agent
conversation in AutoGen, RAG in LlamaIndex, structured outputs in
Instructor, evaluation in LangSmith/Phoenix, observability in Langfuse,
tool ecosystems in LangChain, workflow-grade governance in Temporal.

**EAR builds every one of them natively, from scratch. No integrations,
no adapters, no dependencies** — `dependencies = []` is a shipping
invariant, verified by running the full suite with every third-party
package uninstalled. The LLM is spoken to over HTTPS from the Python
standard library (`ear/llm.py`); structured prompting is native
(`ear/judgment.py`: the model answers in markdown sections, parsed by the
same Section codec the stack is authored with); and the authored surface
never changes shape — six natural-language markdown files, intents in,
decisions out, every judgment on the audit trail.

Each capability below already exists natively at first depth. This plan
is the **parity ledger**: what the best-in-class platform genuinely
provides beyond our current native feature, and the from-scratch build
that closes each gap.

```text
Category               Inspired by          Native today (shipped)                      This plan builds (native, from scratch)
---------------------  -------------------  ------------------------------------------  -----------------------------------------------
Prompt Optimization    DSPy + GEPA          Judgment prompting; reflective refine();     iterative candidate search graded by evals;
                                            trail examples; reviewer verdicts            worked-example demos; persisted instructions
Typed Agents           PydanticAI           Contracts (meanings, judged conformance);    nested/list deliverables; per-field retry
                                            typed coercion; Deliberator backend seam     feedback
Workflow Runtime       Temporal             Journey: durable legs, crash-resume,         leg retry policies; journey runner; parked-
                                            approval park/release, settled idempotency   deadline escalation; event documents
Stateful Agent Graphs  LangGraph            Journey (sequential), halt-on-block          prose-authored branching and loops, judged
                                                                                         routing on the record, revisit budgets
Multi-Agent            AutoGen              Panel: prose patterns, budgeted turns,       judged next-speaker and early conclusion;
                                            synthesis, transcript on the trail           tools inside turns
RAG                    LlamaIndex           Knowledge chunks, judged selection,          native BM25 index; persisted gist index;
                                            citations, open retriever seam               URL sources over the native client
Structured Outputs     Instructor           Contracts + ## Data round-trip               (folds into Typed Agents work above)
Evaluation             LangSmith / Phoenix  Examiner: md evals, honest offline grading,  report history + regression diffs; rubric
                                            report.md                                    scores; A/B stack comparison
Observability          Langfuse             md/JSONL trail; exporter protocol;           per-judgment tokens+latency; prose-declared
                                            per-cycle usage (tokens)                     pricing -> dollars; usage report; LM retries
Integrations           LangChain            open native seams; MCP servers declared      native MCP client (stdlib JSON-RPC over stdio)
                                            in memory.md                                 binding declared servers' tools into cycles
Enterprise Governance  Temporal             policies + approval gates + budgets +        approver allow-lists; tool-scoped policies;
                                            audit trail                                  hash-chained tamper-evident trail; retention
```

---

## Non-negotiables (unchanged, and they bind every item below)

1. **Zero dependencies.** Every feature is Python standard library plus
   the model. CI runs the suite in a bare environment.
2. **The authoring surface stays natural language.** A feature may add a
   memory.md section or a stack-file field; never a config format or
   required Python authoring.
3. **The LLM judges; code enforces and records.** Search loops, budgets,
   retries, allow-lists, hashes — code. Relevance, quality, routing,
   conformance, when-to-stop — the model, on the record.
4. **Honest fallbacks.** Offline, a stage degrades deterministically and
   says so; a judgment nobody made is never written down as one.
5. **Everything on the trail**, and the markdown file on disk stays the
   canonical record.
6. **Secrets by environment-variable name only.**

---

## N1 — Reasoning & optimization depth (parity: DSPy + GEPA, part of Langfuse)

### N1.1 Native LM hardening
- Retry with exponential backoff on transient failures (429/5xx/network),
  attempts and backoff as plain mechanics constants; every retry visible
  in the record, never silent.
- Per-call latency captured in `LM.history` alongside tokens.

### N1.2 Per-judgment accounting
- `Judgment.run` measures its own call: tokens and latency ride the
  ReasoningRecord of the stage that ran it (`tokens`, `latency_ms`), so
  the trail answers "which stage costs what," not just "what did the
  cycle cost."

### N1.3 Prose-declared pricing → dollars
- A `Pricing` section in memory.md ("input tokens cost $3 per million,
  output $15 per million") read by the Strategy like every other section;
  usage records and the ledger (N4.4) carry real dollars. No price table
  ships in code — prices are the author's declaration.

### N1.4 Iterative instruction search (the GEPA parity)
- `Optimizer.search(judgment, examples, metric, generations, candidates)`:
  each generation, the model proposes K instruction rewrites from the
  current best plus the failures (reflection); each candidate is **graded
  against a held-out slice** by the Examiner-shared metric; the best
  survives; repeat for G generations. Search loop, budgets and selection
  are code; proposal and grading are the model. Offline: refuses loudly —
  optimization is judgment.

### N1.5 Worked-example demos
- A `Judgment` carries optional worked examples rendered into its prompt
  (input sections → output sections, the same markdown shape the model
  answers in). The Optimizer selects which trail examples earn a place
  (reviewer-approved first), bounded by a token budget in code.

### N1.6 Persisted instructions
- Refined instructions and chosen demos persist as markdown
  (`.ear/instructions.md`, one section per Judgment, Section codec); the
  Loader applies them on load, so optimization survives restarts and is
  itself reviewable and diffable.

**Done when**: `search()` measurably beats the baseline instruction on a
held-out evaluations directory; the win survives a reload; the trail shows
per-stage tokens and dollars.

## N2 — Evaluation & knowledge depth (parity: LangSmith/Phoenix, LlamaIndex)

### N2.1 Report history and regression diffs
- Reports append to `evaluations/reports/<timestamp>.md` with `report.md`
  as latest; each run diffs against the previous: **newly failing, newly
  passing, still failing** — the regression story a platform dashboard
  tells, as a markdown document.

### N2.2 Rubric scores
- An `## Expected` section may carry graded criteria (bullets), each
  judged separately with a verdict and rationale — a scored rubric per
  evaluation, not just pass/fail; the suite roll-up reports per-criterion
  rates.

### N2.3 A/B stack comparison
- `Examiner.compare(runtime_a, runtime_b, directory)`: both stacks answer
  the same evaluations; a pairwise judgment (`JudgePreference`: which
  outcome better satisfies the expectation, or tie) produces a preference
  report — how prompt changes are decided on evidence, natively.

### N2.4 Native lexical index (BM25)
- Replace raw word-overlap narrowing with BM25 scoring — idf, term
  saturation, length normalization — in pure Python over the Knowledge
  passages. Deterministic mechanics (narrowing is retrieval plumbing);
  the model still judges final relevance and cites.

### N2.5 Persisted gist index
- On first load of a corpus, the model writes a one-line gist per passage,
  persisted to `.ear/index.md` (Section codec) and reused until the source
  file changes (content hash, stdlib). Narrowing scores against gist+text;
  offline, BM25 alone stands, labelled as such.

### N2.6 URL knowledge sources
- Previously refused for lack of a transport; EAR now owns one. A URL
  source is fetched once over the native HTTPS client, cached under
  `.ear/knowledge/`, chunked like any file; refresh is declared in prose
  ("refetch weekly"), checked at load time against the cached timestamp.

**Done when**: a synonym-phrased query retrieves the passage word-overlap
misses; a corpus indexes once and reloads from the gist index; a prompt
edit shows up as "newly passing / newly failing" against the prior report.

## N3 — Execution depth (parity: LangGraph, Temporal runtime, AutoGen)

### N3.1 Prose-authored routing (graphs)
- Steps may carry routing prose: *"If the grade is D or E, skip to the
  customer note."* After each Journey leg, a routing judgment reads the
  authored routes and the leg's outcome and chooses the next authored
  step — **choose among authored steps, never invent one** (the
  Delegator's rule, applied to control flow). Stage `routing` on the
  trail. Loops are legal; a revisit budget per step is code.

### N3.2 Leg retry policies
- Declared in prose on the workflow or in memory.md ("retry a failed leg
  twice before giving up"); a leg whose cycle raises is retried within the
  budget, every attempt on the trail; exhaustion ends the journey as
  `FAILED`, on the record.

### N3.3 Journey runner and deadline escalation
- `Journeys.run_all(directory)`: one pass over every journey record —
  resume the resumable, release the approved, and escalate the expired:
  an approval gate may declare *"escalate after 3 days"*; a parked journey
  found past its deadline gets an `ESCALATED` mark and an escalation note
  in its record. No daemon — the runner is one call, and *when* it runs
  is the operator's cron. Honest about that.

### N3.4 Event documents
- A waiting journey can consume `events/<name>.md` (facts as Context
  bullets) on resume — external signals as markdown, the same way
  approvals already work.

### N3.5 Dynamic panels
- A next-speaker judgment replaces fixed round-robin when the pattern
  calls for it: choose who speaks next **or conclude** — early
  consensus ends the panel before the budget does (budget still capped in
  code). Personas with bound tools may use the native tool loop inside
  their turns, every invocation on the trail as today.

**Done when**: a stack with a skip-route actually skips, on the record; a
crashed leg retries within its declared budget; an expired approval
escalates on the next runner pass; a panel concludes early on consensus.

## N4 — Governance & connectivity depth (parity: Temporal governance, LangChain reach)

### N4.1 Approver allow-lists
- A gated policy may declare `Approvers:` (names/addresses). An approval
  document whose `Approver:` is not on the list is refused loudly — the
  gate stays parked and says why. Who may waive is authored governance,
  enforced in code.

### N4.2 Tool-scoped policies
- `Applies to: tools` scopes a policy to tool invocations: before the
  loop executes a call, the policy is judged against the tool's name and
  arguments; a violation blocks **that call**, the refusal returns to the
  model as text, and the record shows it. Closes the long-open "deny a
  tool by policy" sliver.

### N4.3 Tamper-evident trail
- Each flushed record carries a hash chained over the previous record's
  hash (stdlib hashlib), in both codecs; `ReasoningLog.verify(path)`
  proves a trail unbroken or names the first broken link. An audit trail
  someone could silently edit is not an audit trail.

### N4.4 Retention and the usage ledger
- Retention declared in prose in the audit section ("keep ninety days"),
  applied by the runner (N3.3) — rotation is mechanics, never silent
  deletion: a rotation note replaces what was rotated out.
- A native ledger: `usage-report.md` generated from the trail — cycles,
  stages, tokens, dollars (N1.3), latency, tool calls — the operational
  dashboard, as a markdown document.

### N4.5 Native MCP client (the connectivity flagship)
- MCP is an open JSON-RPC protocol; EAR speaks it from the standard
  library: stdio transport (subprocess + line-delimited JSON-RPC),
  `initialize` / `tools/list` / `tools/call`. Servers stay **declared in
  memory.md** exactly as today; connecting one binds its tools into the
  ToolBinder as BoundTools — same trail records, same budgets, same
  tool-scoped policies (N4.2). No SDK: the protocol is the spec, and the
  spec is JSON over pipes.

**Done when**: an off-list approver is refused; a policy blocks a single
tool call mid-deliberation; `verify()` catches a hand-edited trail record;
a declared MCP server's tool runs in a cycle, on the record.

---

## Sequencing and effort

| Phase | Weeks | Theme | Exit test |
|---|---|---|---|
| **N1** ✅ shipped | 1–3 | reasoning & optimization depth | ✅ live search kept-or-beat baseline on held-out references; per-stage tokens/latency on every record; prose-declared Pricing prices usage; LM retries with backoff, on the record; demos + instructions persisted as markdown and loader-applied |
| **N2** | 4–6 | evaluation & knowledge depth | regression diff between two reports; gist-index reload; synonym retrieval win |
| **N3** | 7–10 | execution depth | routed skip, leg retry, deadline escalation, early panel consensus — each on the record |
| **N4** | 11–14 | governance & connectivity | broken-chain detection; off-list approver refused; tool call policy-blocked; MCP tool invoked natively |

Order rationale: N1 makes every later change *measurable* (metric, evals,
accounting) before behaviour changes; N2 deepens what decisions are made
*from*; N3 changes how execution flows; N4 hardens who may do what and
reaches outward — last, because it guards everything built before it.

## Testing discipline (per feature, no exceptions)

- Offline tests for mechanics and honest fallbacks (search refuses
  offline; BM25 ranks deterministically; hash chain verifies; router
  never invents a step).
- Live tests for each judgment (routing chooses the authored skip; the
  panel concludes early; a candidate instruction wins on the metric).
- The Examiner suite is the regression gate for every phase; N2.1's diff
  report is itself part of that gate from Phase N2 on.
- The bare-environment run (all third-party packages uninstalled) stays in
  CI permanently.

## Risks

| Risk | Mitigation |
|---|---|
| Optimization search cost blows up | generations × candidates × trainset bounded in code; dollars visible per run via N1.3 before anyone scales it |
| Routing loops forever | revisit budgets in code; every routing choice on the trail |
| Gist/index drift from sources | content-hash invalidation; index is markdown, reviewable |
| MCP servers misbehave (hang, bad JSON) | subprocess timeouts, loud LMError-style failures, tool failures already return to the model as text |
| Hash chain breaks on legitimate concurrent writers | one writer per trail file is already the model (Runtime owns its log); verify() names the first break, humans adjudicate |
| Scope creep back toward a framework | the non-negotiables; anything needing a new dependency is redesigned or dropped |
