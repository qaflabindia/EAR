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
(Anthropic natively, or any OpenAI-compatible endpoint — Azure, Ollama,
vLLM and the like — selected by `ModelBinding`, never hardcoded, spoken to
directly over HTTPS by EAR's own dependency-free client), and enforces the
workflow's policies before it runs. So what
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
tenant.md    the org this stack belongs to     (`Org id:`, fiscal year, timezone; optional --
                                                defaults to the "default" tenant when absent)
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
conversation  each panel turn: speaker, round, statement
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

### Panels — multi-persona deliberation, native

A workflow authored with a `Pattern:` line convenes its personas as a
panel instead of reasoning single-voiced — and the pattern is **prose,
not an enum**: it goes into the prompt verbatim, so the deliberation
style is itself a natural-language instruction, never a hardcoded
protocol:

```markdown
## Underwriting Workflow

Pattern: adversarial debate; the Credit Risk Guru has the last word

1. Assess the risk. (Credit Risk Guru)
2. Make the applicant's case. (Customer Advocate)
```

Who speaks next is a judgment, not a rotation: each turn a moderator
judgment reads the pattern and the transcript and chooses the next
speaker — or concludes the panel early when it has genuinely converged,
so consensus ends the deliberation before the budget does. Code guards
what the model may not decide: only listed personas speak (an unreadable
choice falls back to rotation, on the record), conclusion is honored only
once every persona has spoken, and the turn budget still caps everything.
Personas whose skills carry bound tools may use the native tool loop
inside their turns — get the facts, then speak — every invocation a
`tool` record exactly as in deliberation.

Each turn one persona speaks — instructions and stacked skills in hand,
the transcript in view — and a synthesis concludes the panel into the one
decision the pipeline continues with. Governance is untouched: the
Governor gated the cycle before the panel sat, the Validator and
Contracts still check the synthesis, every turn is a trail record (stage
`conversation`, with who chose the speaker and why) and the synthesis is
the cycle's `deliberation`. Budgets are code: `rounds` and a hard
`max_turns` cap. Offline the panel never fakes a debate — it rotates
deterministically, reports who would have deliberated, and says so.

### Journeys — durable, resumable execution, native

`Journey` walks the authored stack one step at a time, each leg a **full
governed cycle** (gates, knowledge, tools, trail, memory all apply),
writing its state to a markdown record after every leg:

```python
from ear import Journey
journey = Journey("journeys/big-loan.md")
journey.run(runtime, intent)      # crash mid-journey? the record has every walked leg
journey.run(fresh_runtime)        # resumes exactly where the record ends
```

A hard block ends the journey (`BLOCKED`); an approval gate parks it
(`PENDING APPROVAL`, stamping when and which policies it awaits) until
`run` is called again with the human's `Approval`; a completed journey is
settled and replays nothing. The record is the same natural language as
everything else — and a journey refuses to resume over a stack whose
steps no longer match the legs it already walked: continuing a changed
plan would forge the record.

Control flow is authored in prose, judged at runtime:

```markdown
## Underwriting Workflow

Routes: if the risk grade is C or worse, skip straight to the decline note.
Retries: retry a failed leg twice before giving up.

1. Band the profile and assign a risk grade. (Credit Risk Guru)
2. Prepare the approval paperwork. (Credit Risk Guru)
3. Write the decline note. (Credit Risk Guru)
```

- **Routing** — after each leg of a routed workflow, a routing judgment
  reads the authored routes and the leg's outcome and chooses the next
  authored step: jump, continue in order, or conclude. The model chooses
  **only among authored steps, never invents one**; loops are legal, and
  a per-step revisit budget in code refuses runaway ones. Every choice is
  a `routing` record; offline the routes are not judged and the journey
  continues in order, saying so.
- **Retries** — declared on the workflow or in a memory.md
  execution/resilience section ("retry a failed leg twice"); a leg whose
  cycle *raises* is retried within the budget, every attempt a `retry`
  record, exhaustion ending the journey `FAILED` on the record. With no
  budget declared, a crash keeps plain crash-and-resume semantics.
- **Events** — external signals as markdown: drop
  `events/<journey-stem>*.md` beside the record and its Context bullets
  merge into the journey's context on resume, each consumed exactly once
  (an `event` record, and an `## Events` line in the journey file).

`Journeys.run_all(runtime, "journeys/")` is the runner: one pass over
every record — resume the resumable, release the approved
(`approvals/<journey-stem>.md`), and escalate the expired. A gated policy
may declare `Escalate: after 3 days`; a parked journey found past that
deadline is marked `ESCALATED` with the reason in its record — still
releasable by an approval, but no longer quietly waiting. No daemon: the
runner is one call, and *when* it runs is the operator's cron.

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
binding overrides). With bound tools present, deliberation becomes EAR's
native tool loop: the model is asked, one step at a time, whether to call
a tool (told each tool's declared description and real parameter names,
introspected from the handler) or to decide — *when* to call is the
model's judgment, within the binder's iteration budget — and every
invocation is a trail record (stage `tool`: arguments, result, duration).
A failing tool never breaks the cycle: the failure is recorded and handed
back to the model as text. Declared-but-unbound tools stay context the
model knows about, as before. Any callable binds — including one that
wraps a tool from another ecosystem, if you bring it.

The loop also **recovers from a malformed turn**. EAR has no provider
`tool_call_id` sequence to leave dangling, but the equivalent mistake — the
model naming a tool that doesn't exist, or returning neither a call nor a
decision — is caught and corrected rather than silently ending the loop:
EAR feeds the mistake back ("no tool named X; here are the real ones") and
lets the model try again, bounded by a small recovery budget and recorded
as a `RECOVERED` tool-trail note, so a hallucinated call becomes a
self-corrected one instead of a lost turn.

The deliberation engine itself is also a seam: attach a `backend` to the
`Deliberator` (anything with `deliberate(runtime, intent, plan, research)`
— e.g. a typed-agent adapter) and it replaces the Reasoner for that step
only, with the Governor, Decider/Validator, Contracts and the trail all
still applying — a backend never reasons off the record.

### MCP — connect a server, natively

MCP servers stay *declared* in memory.md (`- name: what it provides
`command``). MCP is an open JSON-RPC protocol, and EAR speaks it from the
standard library — no SDK: `Runtime.connect_mcp(name)` launches the
declared server as a subprocess, handshakes over stdio (`initialize` /
`tools/list` / `tools/call`), and binds its tools into the ToolBinder as
ordinary BoundTools:

```python
runtime.connect_mcp("calc")     # launches the server declared as `calc`
# ... its tools now join every cycle's toolset ...
runtime.disconnect_mcp()        # shuts the subprocess down
```

A connected server's tools run through the **same logged handler** as any
native tool — every MCP call is a `tool` trail record with arguments,
result and duration, obeys the same tool-loop budget, and is judged by the
same tool-scoped policies. A server that hangs, dies, or answers with
malformed JSON fails loudly as an `McpError`, and — wrapped by the binder
— that failure returns to the model as text. The server was declared by
the author; connecting one is the runtime reaching out to what memory.md
already names, never a capability from nowhere.

### Server — EAR as a control-plane service

The `Server` puts an HTTP front door on the Kernel, so a fleet of instances
can be created, driven and observed over the network — the server-side face
of the same picture. Zero dependencies: the standard library's threading
HTTP server speaking JSON, with the Kernel running the work behind it.

```python
from ear import Server
Server(stacks_root="./stacks", port=8080).serve()   # blocking; Ctrl-C to stop
```
```
python -m ear.server --stacks ./stacks --port 8080
```

| method | path | what |
| --- | --- | --- |
| `GET` | `/health` | uptime, instance count, queue depth |
| `GET` | `/kernel` | the scheduler snapshot (process table, recent dispatches) |
| `GET`/`POST` | `/instances` | list, or create `{name, stack?}` |
| `DELETE` | `/instances/{name}` | remove |
| `POST` | `/instances/{name}/submit` | enqueue `{intent, context?, goal?, every?}` |
| `GET` | `/instances/{name}/status` | health + progress from the trail |
| `GET` | `/instances/{name}/decision` · `/trail` | the latest decision · recent records |

Solid by construction, not afterthought: a **bearer token** read from
`EAR_SERVER_TOKEN` (never hardcoded, compared in constant time) guards
every request, including a health check — unset means open, and the server
says so loudly on start. Loading a stack is **confined** under `stacks_root`; a path that
escapes it is refused, the same discipline as the sandbox. Request bodies
are capped, malformed JSON is a 400 not a crash, and every handler is
wrapped so one bad request can never take the server down. The routing is a
**pure function** — `handle(method, path, body) → (status, payload)` — so
the whole API is testable without opening a socket.

### Kubernetes — instances as Jobs and CronJobs, natively

To run the fleet on Kubernetes — every process a pod, live for the
recurring occurrence of a task — EAR speaks the **Kubernetes REST API
directly over the standard library** (`urllib` + `ssl` + `json`), no
`kubernetes` SDK, no dependency, the same way it speaks to LLM providers
and MCP servers:

```python
from ear import KubeConfig, KubeClient, KubeProvider
provider = KubeProvider(KubeClient(KubeConfig.in_cluster()), image="your-ear-image:1.0")

provider.run("lending", intent)                 # one governed cycle in a Job
provider.schedule("mortgage", intent, every=86400)   # a daily CronJob
kernel.dispatcher = provider.as_dispatcher()    # or let the Kernel schedule, pods execute
```

The mapping is direct: a **runtime instance** runs in a **Job** — one pod,
one cycle, via the in-pod entrypoint `python -m ear.run <stack>`
(`ear/run.py`), the intent handed in through the environment and the exit
code reflecting the outcome (0 decided, 2 blocked, 1 error); a **recurring
task** is a **CronJob** (`every=` mapped to a cron schedule — minutes,
hours, days — sub-minute steered back to the in-process Kernel, which has
no such floor); and `as_dispatcher()` plugs the provider into the **Kernel's
dispatcher seam**, so the Kernel stays the single scheduler while each
firing runs in its own pod. Config is the standard in-cluster
service-account, or an explicit `KubeConfig`; the manifest builders are
pure functions and the client's transport is injectable, so the whole
provider is unit-tested against a faithful fake. (It speaks the real API
but has **not** been run against a live cluster from this repo — the tests
hold it to the API's shape, not a running control plane.)

### Kernel — EAR as an OS scheduler

For a server-side, always-on deployment, the `Kernel` runs EAR the way a
CPU runs a kernel — the classic idle loop:

```text
while running:
    if there_is_work:  run_work()            # dispatch the next task to its instance
    else:              sleep_until_interrupt()  # block until a task or a timer fires
```

```python
from ear import Kernel
kernel = Kernel()
kernel.register("lending", rt_a)             # the process table: named instances
kernel.register("mortgage", rt_b)
kernel.submit("lending", intent)             # enqueue work (an interrupt that wakes the loop)
kernel.schedule("mortgage", intent, every=3600)   # a recurring timer task
kernel.start()                               # drive the loop in the background; stop() to halt
```

The Kernel holds a **process table** of `Runtime` instances (each with its
own sandbox, memory and trail) and a **run queue**. `submit()` enqueues a
task and wakes the loop, the way a syscall raises an interrupt;
`schedule(…, every=…)` makes it recur, the way a timer fires — so an
instance **stays live for the recurring occurrence of a task** without a
busy-wait (between firings the loop genuinely sleeps on a
`threading.Event`). Dispatch runs the instance's normal cycle (`reason`, or
`pursue` for a goal), so policies still gate it, the sandbox still confines
it, the trail still records it — and a governance stop parks the task as
`blocked` (an error as `failed`) without taking the kernel down. `tick()` /
`drain()` advance it synchronously (the testable heartbeat); `snapshot()`
gives a control-room glance for the Monitor. The Kernel decides only *when*
work runs — the control plane — while the judgment stays in the instances.

### Session Goals — a completion condition that drives itself

Attach a plain-English **completion condition** and let the runtime pursue
it — running cycle after cycle until the goal is met, genuinely blocked, or
the budget runs out:

```python
outcome = runtime.pursue(
    "Reach a clear approve-or-decline decision, with the risk grade stated.",
    intent,
)
outcome.status      # "satisfied" | "blocked" | "exhausted" | "ungraded"
outcome.blocker     # the typed reason it stopped
```

After each cycle a `JudgeGoalProgress` judgment decides whether the goal is
met and, if not, names exactly **one typed blocker**:

- `goal_not_met_yet` — more work would help → **continue autonomously**
- `needs_user_input` — a human must supply something → stop, surface it
- `external_wait` — waiting on an outside event/system → stop, surface it
- `missing_evidence` — the work can't be verified → stop, surface it
- `run_failed` — it went wrong and can't recover → stop, surface it

Only `goal_not_met_yet` earns a continuation: the keeper takes the
evaluator's own `next_step` and drives another cycle. The loop is **bounded
in code** — a maximum number of continuations (default 8) and a
**no-progress breaker** that stops after the same non-progress verdict
repeats (default 2×), so an autonomous loop can never run away. Governance
stops map to blockers with no special-casing: an approval gate
(`ApprovalRequired`) is `needs_user_input`; any other refusal is
`run_failed`. Every evaluation is a `goal` trail record with its blocker
and evidence. Offline, with no model to judge, the keeper stops at
`ungraded` after the first cycle and **never fabricates** satisfaction or a
continuation — a judgment nobody made is never written down.

### Sandbox — each runtime instance in its own workspace

Declare a `## Sandbox` section in memory.md and every runtime instance gets
its own **filesystem-confined, resource-limited workspace** — what a
heavyweight harness gives you with Docker, EAR gives you from the standard
library (`pathlib` + `subprocess` + POSIX `resource`), no dependency:

```markdown
## Sandbox

Isolate each runtime under `.ear/box`. Shell commands time out after 30
seconds; limit memory to 512 MB. Expose file and shell tools.
```

The sandbox confines the runtime's file tools to its root (an absolute
path or a `..` that escapes raises `SandboxViolation` — a `PermissionError`
that returns to the model as text), and its `run()` executes commands with
a **wall-clock timeout**, optional **CPU/memory rlimits**, and an
environment **stripped of the ambient process's secrets** — so a spawned
command never inherits your `ANTHROPIC_API_KEY`. When the section asks for
tools ("expose a shell", "read and write files"), the sandbox binds
`read_file` / `write_file` / `list_files` / `run_shell` into the cycle's
tool loop, each on the trail through the same logged handler and governed
by the same tool-scoped policies. The opening is a `sandbox` trail record.
**Isolation nests:** a spawned subagent gets its own `child()` box under
the parent's root.

Stated honestly, the way a serious system must: a pure-stdlib sandbox is a
*containment convention* for EAR's own file tools plus a *resource and time
boundary* for spawned commands — **not a security jail against hostile
code** (`cwd` confinement is not `chroot`). For a true isolation boundary,
plug an OS-container provider into the same seam: anything exposing
`resolve` / `read_text` / `write_text` / `run` / `as_tools` can stand in
for `Sandbox` on `runtime.sandbox`, and the rest of the runtime never
changes.

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

A gate may also declare **who** may waive it with `Approvers:` (names or
addresses) — an allow-list, matched case- and punctuation-insensitively.
An approved verdict from someone off the list waives nothing: the gate
stays parked and the record says who was refused and why. Who may waive is
authored governance, enforced in code.

### Tool-scoped policies — deny a tool by policy

A policy scoped `Applies to: tools` is judged against a tool's name and
arguments *before the call runs*:

```markdown
## Transfer Cap

The wire transfer tool must never move more than $10,000 in one call.

Fallback: amount <= 10000
Applies to: tools
```

A violation blocks that one call — the refusal returns to the model as
text (exactly like a tool failure, so the model reasons on) and the block
is a `tool` trail record naming the policy. The statement is judged by the
model against the call's context; the fallback expression (`amount <=
10000`) keeps it enforced offline. This governs native tools and connected
MCP tools alike.

### Knowledge — RAG with citations, declared in natural language

Reference material is stacked in memory.md under a Knowledge section, one
bullet per source — paths or globs resolved relative to the stack (a
source that matches nothing fails loudly at load), or a URL fetched over
EAR's own HTTPS client, cached under `.ear/knowledge/`, and refreshed on
the cadence the same bullet declares in prose:

```markdown
## Knowledge

- underwriting manual: `knowledge/underwriting-manual.md`
- market brief: https://example.com/brief.md, refetch weekly
```

Markdown sources are chunked into passages by the same Section parser the
stack is authored with. Narrowing is native BM25 — inverse document
frequency, term saturation, length normalization, in pure Python — scored
over each passage's text *and* its gist: a one-line, model-written summary
in everyday words, built once per corpus and persisted to `.ear/index.md`
keyed by content hash (edit a source and only its entries re-gist on the
next load; delete the file and it rebuilds). The gist is what lets a
question phrased in synonyms find the passage whose jargon never uses
them; offline the gists are simply absent, BM25 over the raw text stands,
and the `retrieval` record's `narrowing` input says which is in force.

Per cycle, the `Librarian` narrows the corpus
structurally and the model judges which passages a careful analyst would
actually consult — choosing none is a valid judgment, and it can only
choose among real passages. What was consulted is first-class evidence:
a `retrieval` trail record, `citations` in the Evidence, a `## Sources`
section in the decision document, and the retrieved text reaches the
Reasoner framed as reference material, never as instructions. A custom
retriever — anything with `retrieve(query) -> list[Passage]` — plugs into
the same seam; your retriever narrows, EAR's model still judges and
cites.

### Observability — exporters off the trail, usage on every cycle

The ReasoningLog is the native trace; observability is an exporter, never
a second instrumentation path. Attach anything with `export(record)` to
`runtime.reasoning_log.exporters` — an exporter that raises never breaks a
cycle (failures stay visible in `export_errors`), and the file on disk
remains the canonical record. The protocol is native and two methods
small, so shipping the trail to any external system is a few lines of your
own code — never a dependency of EAR's.

Accounting is per judgment, not just per cycle: every stage record carries
the tokens and latency its own model calls spent (fallback judgments are
never billed for a model they didn't use), and the LM client retries
transient failures with backoff — retry counts on the record, auth errors
failing fast. Declare a `Pricing` section in memory.md ("Input tokens cost
$3 per million; output tokens cost $15 per million.") and usage records
carry real dollars; a figure nobody declared is never invented.

The trail is **tamper-evident**: every flushed record carries a hash
chained over the previous record's (stdlib `hashlib`), in both codecs —
an HTML comment in markdown, a `chain` field in JSONL. `ReasoningLog.verify(path)`
recomputes the chain over the file's own bytes and either proves it
unbroken or names the exact record where an edit, insertion or deletion
first breaks it. Retention is declared in the audit prose ("keep 90 days")
and applied automatically after every cycle — `Runtime.reason` rotates the
trail on its own, whether or not Journeys are ever used, and the Journey
runner rotates once more per batch pass as a backstop for a pass where no
journey ran a fresh cycle — as *rotation, not deletion*: cycles past the
window are replaced by a single `retention` note and the file is
rewritten, re-chained and still verifiable — what was rotated out is
accounted for, never silently gone. `runtime.write_usage_report(path)`
renders the operational ledger from the trail: one row per cycle — model
calls, tokens, dollars (when Pricing is declared), latency, tool calls —
with totals, as a markdown document.

### Dashboard — a visual runtime board, native

What TensorBoard is for a training run, the `Dashboard` is for a runtime —
and, like everything else here, it is a plain file, not a service:

```python
from ear import Dashboard
Dashboard().write(runtime, "dashboard.html")   # a self-contained snapshot
```

The board is a *view* of the ReasoningLog, never a second instrumentation
path: one HTML document with its CSS, its charts (inline SVG) and its few
lines of script all embedded — no CDN, no framework, no build step, no
dependency. It opens in any browser, works offline, and is theme-aware.
A **cycle** is the step, the **scalars** are the tokens, latency and
dollars each cycle spent (bar charts), and on top of the scalars it shows
what a training board cannot: the hash-chain integrity badge, the
governance table (which policies passed, blocked or parked), the tool
calls, and a per-cycle accordion where every stage expands to the output,
rationale and inputs the model actually reasoned with. `render(source)`
returns the HTML and takes a Runtime, a ReasoningLog, or a JSONL trail
path (rebuilt losslessly via `ReasoningLog.from_trail`); `serve(source,
port)` runs a live, self-refreshing view over the standard-library HTTP
server — the closest thing to `tensorboard --logdir` the stdlib allows.

**Live Gantt.** For progression over time there's a Gantt view — every
process on a wall-clock axis, coloured by health, with a "now" marker:

```python
Dashboard().write_gantt(runtime, "gantt.html")            # one lane per cycle
Dashboard().render_gantt({"a": rt_a, "b": rt_b})          # one lane per runtime
serve(runtime, gantt=True, refresh=3)                      # live, auto-ticking
```

A cycle is a bar from its first record's timestamp to its last; the lane
dot and each bar carry the runtime's **health** and the cycle's **status**
(green decided/healthy · amber pending/awaiting · red policy-blocked or
retry-exhausted). Status is read only from the governance stages (policy,
approval, retry, evaluation) — a loan *declined on the merits* is a sound
decision and stays green; only a governance stop reads red. The page can
**tick itself**: pass `refresh` (seconds) and it emits a meta-refresh, so
`serve(..., refresh=3)` re-reads the trail from disk and re-renders every
few seconds — a separate process writing the trail is watched live, the
bars extending and appearing on their own. A per-runtime **heartbeat**
(`active` / `idle` / `stale`, from the age of the last activity) tells a
live runtime from a quiet or hung one. No daemon, no websocket — an
auto-refreshing static page over `http.server`.

**The control room: a live TUI.** For a wall-of-screens view there's the
`Monitor` — the whole fleet as a **factory assembly line**, drawn in the
terminal with nothing but ANSI and Unicode (truecolor, zero dependencies):

```python
from ear import Monitor
Monitor().run({"lending": rt_a, "mortgage": rt_b})   # live, until Ctrl-C
Monitor().run("trails/")                              # a directory of JSONL trails
```
```
python -m ear.monitor trails/
```

Each runtime instance is an **assembly lane**; the latest cycle lights up
the pipeline stations it passed (`GOV·DIS·SEL·SCH·DEL·RES·TOL·DLB·EXP·AUD·LRN`)
with a scanning sweep, a **conveyor** of recent outcomes streams past
coloured by status, a block sparkline trends tokens, and a health glyph +
live pulse mark each lane. A gradient banner, KPI tiles (instances / health
/ cycles / tokens / cost), a running clock and a spinner update on a tick,
so an operator *watches the fleet breathe*. `render_frame(...)` returns one
frame as a string (testable without a terminal); `run(...)` drives the live
loop over the alternate screen buffer, restoring the terminal on exit. It
reads the same fleet the Dashboard does — one source of truth for health,
two ways to look at it.

Run more than one runtime and the board goes fleet-wide:

```python
Dashboard().write_fleet({"lending": rt_a, "mortgage": rt_b}, "fleet.html")
Dashboard().render_fleet("trails/")   # or a directory of JSONL trails, one run each
```

The fleet page leads with health tiles (**healthy / attention / broken**)
and a worst-status badge, then cross-run comparison charts (cycles, tokens
and cost per runtime). Below, one card per runtime shows a health dot, a
tokens-per-cycle **progress sparkline**, flags (blocked / pending / chain),
and last activity — and expands into that runtime's full single board.
Health is honest: a *broken* audit-trail chain is the only hard fault;
failed cycles, exporter errors and pending approvals raise *attention*;
policy blocks are governance working and stay *healthy*, surfaced as a
count. `render_fleet` accepts a `{name: runtime}` dict, a list of runtimes,
or a directory of trails; `serve` renders a fleet the same way, live.

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

At runtime the model fills the fields from the prose decision (an
extraction prompt is built dynamically from the authored meanings, one
markdown section per field) and then judges
the filling against those meanings — one hinted retry, then nonconforming
data is withheld, on the record (trail stage `contract`). Conformant data
travels as a `## Data` section in the decision document, typed by the same
`coerce` codec as intent context, and parses back via
`ear.exchange.data_from_decision_document`. With no model bound nothing is
fabricated: the skip itself is a trail record.

### Evaluation — the Examiner and markdown-native evals

An evaluation is one markdown file in an `evaluations/` directory: an
ordinary intent document plus an `## Expected` section — prose criteria
(a blocked refusal can itself be the expectation), bullets of
`field: value` the delivered Data must carry, and colon-less bullets as a
graded rubric, each criterion judged separately with its own verdict and
rationale:

```python
from ear import Examiner
examination = Examiner().examine(runtime, "examples/credit_risk_stack/evaluations")
assert examination.passed          # the CI regression gate
```

With a model, `JudgeDecisionQuality` grades each outcome with a rationale
— once for the prose-and-fields expectation, once per rubric criterion (a
failed criterion fails the evaluation); offline, only the field bullets
are checked structurally and prose criteria are reported **ungraded**
rather than faked. Verdicts land on the trail (stage `evaluation`) and in
`evaluations/report.md` — and every run is regression history: reports
archive to `evaluations/reports/<timestamp>.md`, and each report diffs
itself against the previous one (**newly failing / newly passing / still
failing**), so a prompt edit shows its consequences as a markdown
document.

Two stacks compare head-to-head over the same directory:

```python
comparison = Examiner().compare(runtime_a, runtime_b, "evaluations", judge=referee_binding)
```

Both answer every evaluation; a pairwise `JudgePreference` judgment picks
A, B or tie per expectation (an unreadable preference records as a tie,
never a silent winner), and the report lands in
`evaluations/comparison.md`. Pass a dedicated `judge` binding to keep the
referee independent of the contestants. `compare` refuses to run without
a model: a preference judgment nobody made is never written down.

### Optimization — the trail is the training corpus

```python
from ear import Optimizer
optimizer = Optimizer()
trainset = optimizer.trainset_from_trail(".ear/reasoning.md")   # or .jsonl
labels   = optimizer.verdicts_from_documents("decisions/")      # ## Review + Verdict: lines
optimizer.refine_reasoner(runtime, ".ear/reasoning.md", "reviews/")  # reflect and rewrite
```

Deliberation records become worked `Example`s (the exact intent, context
and stacked capabilities the model reasoned with); a reviewer labels a
decision document by adding a `## Review` section with a `Verdict:` line;
and `refine` has the model reflectively rewrite a reasoning instruction
against those examples, graded by the same `JudgeDecisionQuality`-backed
metric the Examiner uses — evaluation and optimization share one notion of
quality, natively.

Beyond one-shot refinement, `Optimizer.search(judgment, examples,
model_binding, generations=…, candidates=…)` runs an iterative search:
each generation the model proposes candidate rewrites (reflecting on the
current best and the failures), each candidate is graded on held-out
reference examples, and the best survives — the loop, split and selection
are code; the proposals and grading are the model; a search that finds
nothing better changes nothing. `select_demos` picks reviewer-approved
worked examples into the judgment's prompt within a character budget, and
`save_instructions`/`load_instructions` persist the refined instructions
and demos as reviewable markdown (`.ear/instructions.md`, applied
automatically by the loader) — optimization survives restarts and is
itself diffable.

Every judgment is made dynamically at runtime, in natural language
against a live LLM — through EAR's own structured prompting (a `Judgment`
declares inputs and outputs; the model answers in markdown sections; the
same Section codec that parses the stack parses the answer) — `Policy`
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
single one. `Optimizer` is a structural, dev-time operation — optimizing
isn't part of running a cycle, so it sits outside this pipeline and is
called directly (see *Optimization — the trail is the training corpus*
above).

## Roadmap

[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) is the native
parity plan: for each capability that exists in isolation on other
platforms — prompt optimization, typed agents, durable workflows, agent
graphs, multi-agent deliberation, RAG, structured outputs, evaluation,
observability, tool connectivity, enterprise governance — it records what
EAR already ships natively, the honest gap to best-in-class depth, and the
from-scratch build that closes it. Zero dependencies throughout; the
authored stack never changes shape.

[`CHANGELOG.md`](CHANGELOG.md) tracks what actually shipped, release by
release.

## Install

```bash
pip install -e .
```

New here? Start with **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** —
it takes you from install to a running, governed agent in a few minutes
(offline first, no API key needed), then to a markdown-authored stack, a
live model, the dashboard, and running as a service.

EAR is an independent package with **zero dependencies**: it speaks to LLM
providers directly over HTTPS from the Python standard library
(`ear/llm.py`), and its structured prompting is native (`ear/judgment.py`)
— the model answers in markdown sections, parsed by the same Section codec
the stack is authored with. `pip install -e '.[dev]'` adds pytest, and
nothing else.

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

`Runtime.reason` activates `runtime.model_binding` (building EAR's own `LM`)
before any judgment-laden stage runs, so `Policy`, `Discoverer`, the
`Reasoner` and the `Explainer` all reason against the same live model.

## Provider-agnostic routing: the omni-route `Router`

A single `ModelBinding` speaks to one provider. A `Router` is the
OmniRoute-style, provider-agnostic agent: it stacks *many* provider
bindings — any provider `ear/llm.py` speaks to (Anthropic natively, or
anything OpenAI-compatible via `api_base`: OpenAI, Azure, Ollama, Together,
vLLM, … 250+), each reading its own key from the environment — behind one
binding, picks which to try first by a **routing strategy**, **falls back**
to the next provider when one errors or is rate-limited, and trips a
**circuit breaker** that benches a failing provider for a cooldown window so
the next call routes around it.

A `Router` *is* a drop-in `ModelBinding` — same `activate()`, `lm` and
`model_id` surface — so you assign it to `runtime.model_binding` and the
whole pipeline (`Governor`, `Discoverer`, `Policy`, `Reasoner`, `Explainer`,
...) becomes provider-agnostic without any stage knowing a router is there.
It builds no framework of its own: once a provider's turn comes, `Router`
just calls that provider's own native `LM.complete(prompt, system=...)` —
routing is a seam in front of EAR's dependency-free LLM client, not a
replacement for it.

```python
from ear import ModelBinding, Router, RoutingStrategy

router = Router.across(
    ModelBinding(provider="anthropic", model="claude-opus-4-8"),   # tried first (list order = priority)
    ModelBinding(provider="openai", model="gpt-4o"),
    ModelBinding(provider="groq", model="llama-3.3-70b"),
    strategy=RoutingStrategy.PRIORITY,   # ordered fallback
)
runtime.model_binding = router
```

Routing metadata (`priority`, `cost_per_1k`, `weight`, `is_free`, `label`)
lives on a small `RouterProvider` wrapper, not on the shared `ModelBinding`
class itself — a binding's priority is a property of *this* router, not of
the provider. Use `add_provider` for metadata beyond list-order priority:

```python
router = Router(strategy=RoutingStrategy.FREE_FIRST)
router.add_provider(ModelBinding(provider="anthropic", model="claude-opus-4-8"), priority=10)
router.add_provider(ModelBinding(provider="groq", model="llama-3.3-70b"), is_free=True)
```

The strategy only decides *who goes first*; the Router always walks the
ordered list and falls back on failure:

| Strategy | Orders providers by |
| --- | --- |
| `PRIORITY` | lowest `priority` number first (ordered fallback) |
| `CHEAPEST` | lowest `cost_per_1k` first |
| `FREE_FIRST` | `is_free` providers first, then by priority |
| `ROUND_ROBIN` | rotates the starting provider each call |
| `WEIGHTED` | random order, biased by each provider's `weight` |
| `RANDOM` | uniform random order |

Because config belongs in the environment, never hardcoded, a Router can be
built straight from an env var — a JSON array of providers, or a shorthand
`provider/model` list:

```python
# EAR_ROUTER='anthropic/claude-opus-4-8, openai/gpt-4o, groq/llama-3.3-70b'
router = Router.from_env("EAR_ROUTER", strategy=RoutingStrategy.PRIORITY)
```

The selection order, the fallback walk and the cooldown breaker are plain,
deterministic Python (`router.order()`, `router.dispatch(...)`), so the
routing behaviour is fully testable with fake per-provider `LM`s and no
network at all — the same offline/live two-tier testability the rest of
the package keeps.

## Progressive skill selection

A `Persona` can carry a large library of `Skill`s without every prompt
being stacked into every reasoning call. The `SkillSelector` stacks only
the skills relevant to the current intent — the same relevance ranking
`Discoverer` already does for whole processes, applied one level down to a
persona's skills:

```text
Discoverer     → ranks Processes relevant to an Intent   (LLM-ranked, keyword fallback)
SkillSelector  → ranks a Persona's Skills the same way    (LLM-ranked, keyword fallback)
```

It is on by default (`Reasoner(skill_selector=SkillSelector(top_k=8))`) and
**short-circuits** — returning every skill, in order — whenever a persona
has `top_k` or fewer skills, so the common case costs nothing and small
personas behave exactly as before. Only when a persona exceeds `top_k` does
it rank (natively, when a model is active; by keyword overlap offline) and
keep the most relevant. Pass `skill_selector=None` to always stack every
skill.

`Skill` also carries **provenance** — `version` and `author` — so a
decision's audit trail can be traced back to the exact skill (and version)
that shaped it: which version of which capability produced this decision.

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
  tool_binder.py   ToolBinder    — declared tools meet executables; every invocation on the trail, tool-scoped policies enforced
  sandbox.py       Sandbox       — each runtime instance's confined workspace + governed command runner (stdlib, no Docker); nests per subagent
  goal.py          GoalKeeper    — session goals: a completion condition pursued with typed blockers and a bounded autonomous continuation loop
  mcp_client.py    McpClient     — the native MCP client: JSON-RPC over stdio from the stdlib, tools bound into cycles
  panel.py         Panel         — multi-persona deliberation in authored prose patterns; judged speakers, early consensus, tools in turns
  journey.py       Journey       — durable, resumable, prose-routed execution; the state a markdown record; Journeys is the runner
  contract.py      Contract      — a workflow's Deliverable: fields with plain-English meanings, extracted and judged at runtime
  examiner.py      Examiner      — examine: markdown evals, rubric criteria, report history + regression diffs, A/B compare
  knowledge.py     Knowledge     — the declared reference corpus: Section-parsed chunks, BM25 narrowing, persisted gist index
  librarian.py     Librarian     — research: retrieve relevant Knowledge with citations, on the record
  process.py       Process       — a stack of Workflows that performs an action
  policy.py        Policy        — governance rule, judged in natural language with a safe-eval fallback; attaches runtime-wide or to a Workflow
  runtime.py       Runtime       — runs every cycle through the full operation pipeline below
  model_binding.py ModelBinding  — LLM provider binding (model, credentials, params -> EAR's own LM)
  router.py        Router        — omni-route: routes across many ModelBindings with fallback + cooldown (a drop-in ModelBinding)
  llm.py           LM            — the dependency-free LLM client: stdlib HTTPS, provider wire formats
  judgment.py      Judgment      — native structured prompting: declared inputs/outputs, markdown answers
  evidence.py      Evidence      — why this decision was made
  memory.py        Memory        — persistent memory (working + compressed layers)
  experience.py    Experience    — the pattern aggregated from repeated Memory entries
  adaptation.py    Adaptation    — learned adaptations distilled from Experience
  reasoner.py      Reasoner      — the deliberation step, with the native tool loop
  signatures.py    the native Judgments shared across the LLM-judged stages
  governor.py      Governor      — govern: enforce Policy gates
  initializer.py   Initializer   — initialize: activate the ModelBinding
  discoverer.py    Discoverer    — discover: find Processes relevant to an Intent
  selector.py      Selector      — select: choose among candidates (LLM-chosen, dedupe fallback)
  skill_selector.py SkillSelector — select: stack only the Skills relevant to an Intent (LLM-ranked, keyword fallback)
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
  optimizer.py     Optimizer     — optimize: trail-fed examples, reviewer verdicts, the shared
                                   quality metric, and native reflective instruction refinement
  section.py       Section       — the shared structural parser for stacked markdown files
  loader.py        Loader        — load_runtime: stack skills.md/persona.md/workflow.md/
                                   process.md/policy.md/tenant.md/memory.md into a Runtime
  strategy.py      Strategy      — the memory.md operating strategy, read from plain English
  tenant.py        Tenant        — the org this stack belongs to, stacked from tenant.md: org_id, fiscal year bounds, timezone
  identity.py      Claim         — who is calling and which Tenant org_id(s) they may act as; Runtime.reason/Kernel.submit refuse a Claim not authorized for the instance's tenant
  exchange.py      Exchange      — the markdown boundary: intents/*.md in, decisions/*.md out
  reasoning_log.py ReasoningLog  — the reasoning audit trail (markdown/JSONL); hash-chained + verify(), retention rotation, usage ledger
  dashboard.py     Dashboard     — self-contained HTML runtime board from the trail (TensorBoard-equivalent): render_fleet, live auto-ticking render_gantt, zero deps
  monitor.py       Monitor       — the premium live TUI: the whole fleet as a factory assembly line, pure ANSI truecolor, zero deps
  kernel.py        Kernel        — EAR as an OS scheduler: process table of instances, a run queue, the run-or-sleep idle loop
  server.py        Server        — the control plane: a stdlib HTTP service over the Kernel (token auth, confined stack loading), zero deps
  k8s.py           KubeProvider  — run instances as K8s Jobs/CronJobs, spoken natively over the REST API (stdlib, no SDK); the Kernel's execution seam
  run.py           (entrypoint)  — python -m ear.run <stack>: run one cycle in a pod from EAR_INTENT/EAR_CONTEXT, exit code = outcome
  session_store.py SessionStore  — cross-session data (markdown by default, JSON optional)
  spawner.py       Spawner       — spawn subagent runtimes, bounded by the strategy
  tool.py          Tool          — a tool declared in plain English, surfaced to reasoning
  mcp_server.py    McpServer     — an MCP server declared in plain English
  ontology.py      Ontology      — the term→meaning vocabulary folded into reasoning
```
</content>
