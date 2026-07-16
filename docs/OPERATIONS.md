# Operations

How to run EAR — from answering a folder of requests, to a scheduled fleet, to
pods on Kubernetes — plus the tools, journeys, observability and provider routing
that go with a running system. Everything here is built from the Python standard
library, with no dependencies.

---

## The Exchange — an inbox of requests

The simplest way to run a stack is to let intent documents drive it:

```python
from ear import Exchange, load_runtime

runtime = load_runtime("examples/credit_risk_stack")
Exchange("examples/credit_risk_stack").run(runtime)   # answers every unanswered intents/*.md
```

`Exchange.run` is idempotent — an inbox, not a replay: intents whose decision
document already exists are skipped. `Exchange.respond(runtime, intent_markdown)`
is the same boundary as text-in/text-out. See
[Authoring → boundary](AUTHORING.md#the-requestresponse-boundary) for the document
shapes.

---

## Tools and MCP

Tools stay **declared** in `memory.md`; the `ToolBinder` is where a declaration
meets an executable.

```python
runtime.bind_tool("amortization_calculator", monthly_payment)   # any callable
```

Binding a name nothing in the stack declares fails loudly — code never grows the
runtime a capability the authoring doesn't show. With bound tools present,
deliberation becomes EAR's native **tool loop**: the model is asked, one step at a
time, whether to call a tool (given each tool's description and real parameter
names) or to decide. *When* to call is the model's judgment, within the binder's
iteration budget; every invocation is a `tool` trail record (arguments, result,
duration). A failing tool never breaks the cycle — the failure is recorded and
handed back to the model as text. A malformed turn (a tool that doesn't exist) is
fed back and corrected, bounded by a small recovery budget.

**MCP servers** are declared the same way and connected natively — EAR speaks the
JSON-RPC protocol over stdio from the standard library, no SDK:

```python
runtime.connect_mcp("calc")     # launches the server declared as `calc`
# ... its tools now join every cycle's toolset ...
runtime.disconnect_mcp()
```

A connected server's tools run through the **same logged handler** as any native
tool — every call a `tool` record, obeying the same tool-loop budget and the same
[tool-scoped policies](GOVERNANCE.md#tool-scoped-policies). A server that hangs,
dies, or answers with malformed JSON fails loudly as an `McpError` and returns to
the model as text.

---

## Journeys: durable, resumable execution

A `Journey` walks the authored stack one step at a time, each leg a **full
governed cycle** (gates, knowledge, tools, trail, memory all apply), writing its
state to a markdown record after every leg:

```python
from ear import Journey

journey = Journey("journeys/big-loan.md")
journey.run(runtime, intent)      # crash mid-journey? the record has every walked leg
journey.run(fresh_runtime)        # resumes exactly where the record ends
```

A hard block ends the journey (`BLOCKED`); an approval gate parks it (`PENDING
APPROVAL`) until `run` is called again with the human's `Approval`; a completed
journey is settled and replays nothing. A journey refuses to resume over a stack
whose steps no longer match the legs it already walked — continuing a changed plan
would forge the record.

Control flow (`Routes:`, `Retries:`) and external `events/` are authored in prose;
see [Authoring → routes and retries](AUTHORING.md#routes-and-retries-for-journeys).

**The runner** does one pass over a directory of records:

```python
from ear import Journeys
Journeys.run_all(runtime, "journeys/")   # resume the resumable, release the approved, escalate the expired
```

It releases journeys with a matching `approvals/<stem>.md`, and marks a parked
journey past its `Escalate: after N days` deadline as `ESCALATED`. There is no
daemon — the runner is one call, and *when* it runs is your cron.

---

## Session goals

Attach a plain-English **completion condition** and let the runtime pursue it —
cycle after cycle until the goal is met, genuinely blocked, or the budget runs
out:

```python
outcome = runtime.pursue(
    "Reach a clear approve-or-decline decision, with the risk grade stated.",
    intent,
)
outcome.status      # "satisfied" | "blocked" | "exhausted" | "ungraded"
outcome.blocker     # the typed reason it stopped
```

After each cycle a judgment decides whether the goal is met and, if not, names
exactly one typed blocker. Only `goal_not_met_yet` earns a continuation; the
others (`needs_user_input`, `external_wait`, `missing_evidence`, `run_failed`)
stop and surface. The loop is bounded in code — a maximum number of continuations
(default 8) and a no-progress breaker — so it can never run away. Offline the
keeper stops at `ungraded` after the first cycle and never fabricates satisfaction.

---

## The Kernel — a scheduler for a fleet

For an always-on deployment, the `Kernel` runs EAR the way a CPU runs a kernel —
the classic idle loop, sleeping until there is work:

```python
from ear import Kernel

kernel = Kernel()
kernel.register("lending", rt_a)              # the process table: named instances
kernel.register("mortgage", rt_b)
kernel.submit("lending", intent)              # enqueue work (wakes the loop)
kernel.schedule("mortgage", intent, every=3600)   # a recurring timer task
kernel.start()                                # drive the loop in the background; stop() to halt
```

The Kernel holds a process table of `Runtime` instances (each with its own
sandbox, memory and trail) and a run queue. Dispatch runs the instance's normal
cycle, so policies still gate it and the trail still records it — a governance stop
parks the task as `blocked` (an error as `failed`) without taking the kernel down.
`tick()` / `drain()` advance it synchronously for tests; `snapshot()` gives a
control-room glance. Pass `Kernel(max_workers=N)` (`N>1`) to run *different*
instances concurrently on a bounded pool while keeping at most one in-flight cycle
per instance.

---

## The Server — an HTTP control plane

The `Server` puts an HTTP front door on the Kernel, so a fleet can be created,
driven and observed over the network — the standard library's threading HTTP
server speaking JSON:

```python
from ear import Server
Server(stacks_root="./stacks", port=8080).serve()   # blocking; Ctrl-C to stop
```
```bash
python -m ear.server --stacks ./stacks --port 8080
```

| method | path | what |
|---|---|---|
| `GET` | `/health` | uptime, instance count, queue depth |
| `GET` | `/kernel` | the scheduler snapshot |
| `GET`/`POST` | `/instances` | list, or create `{name, stack?}` |
| `DELETE` | `/instances/{name}` | remove |
| `POST` | `/instances/{name}/submit` | enqueue `{intent, context?, goal?, every?}` |
| `GET` | `/instances/{name}/status` | health + progress from the trail |
| `GET` | `/instances/{name}/decision` · `/trail` | the latest decision · recent records |

A **bearer token** read from `EAR_SERVER_TOKEN` (never hardcoded, compared in
constant time) guards every request, including the health check — unset means
open, and the server says so loudly on start. Loading a stack is **confined**
under `stacks_root`; a path that escapes it is refused. Request bodies are capped,
malformed JSON is a 400 not a crash, and the routing is a pure function
(`handle(method, path, body) → (status, payload)`) so the whole API is testable
without a socket.

---

## Kubernetes — instances as Jobs and CronJobs

To run the fleet on Kubernetes, EAR speaks the **Kubernetes REST API directly over
the standard library** (`urllib` + `ssl` + `json`) — no `kubernetes` SDK:

```python
from ear import KubeConfig, KubeClient, KubeProvider

provider = KubeProvider(KubeClient(KubeConfig.in_cluster()), image="your-ear-image:1.0")
provider.run("lending", intent)                    # one governed cycle in a Job
provider.schedule("mortgage", intent, every=86400) # a daily CronJob
kernel.dispatcher = provider.as_dispatcher()       # or let the Kernel schedule, pods execute
```

A runtime instance runs in a **Job** — one pod, one cycle, via the in-pod
entrypoint `python -m ear.run <stack>`, the intent handed in through the
environment and the exit code reflecting the outcome (`0` decided, `2` blocked, `1`
error). A recurring task is a **CronJob**. `as_dispatcher()` plugs the provider
into the Kernel's dispatcher seam, so the Kernel stays the single scheduler while
each firing runs in its own pod.

> The provider speaks the real API and is unit-tested against a faithful fake, but
> has **not** been run against a live cluster from this repo — the tests hold it to
> the API's shape, not a running control plane.

---

## The audit trail

`runtime.reasoning_log` records every judgment, one stage-labelled record per
judgment:

```
intent · policy · discovery · selection · scheduling · delegation · deliberation
recall · retrieval · conversation · tool · approval · usage · explanation · audit
· contract · evaluation · goal · adaptation · evolution · retention
```

Each record carries the model that produced it (`deterministic-fallback` when no
model was active); blocked cycles are logged too. Declare a **Reasoning Audit
Trail** section in `memory.md` and the runtime appends the trail to disk after
every cycle — readable markdown by default (`.ear/reasoning.md`, one `## Cycle`
section per cycle), or JSONL when the declared path ends in `.jsonl`.

```python
print(runtime.reasoning_log.render())                       # the skim view
runtime.reasoning_log.for_stage("deliberation")[-1].inputs  # the full prompt material
```

The trail is hash-chained and tamper-evident; see
[Governance → the trail](GOVERNANCE.md#the-tamper-evident-trail).

---

## Observability, usage and dashboards

- **Exporters** — the trail is the native trace; observability is an exporter,
  never a second instrumentation path. Attach anything with `export(record)` to
  `runtime.reasoning_log.exporters`. An exporter that raises never breaks a cycle
  (failures stay visible in `export_errors`), and the file on disk stays canonical.
- **Usage** — accounting is per judgment: every stage record carries the tokens
  and latency its own model calls spent. Declare a `Pricing` section and usage
  records carry real dollars. `runtime.write_usage_report(path)` renders the
  operational ledger as a markdown document.
- **Dashboard** — a self-contained HTML board, the training-run equivalent for a
  runtime:

  ```python
  from ear import Dashboard
  Dashboard().write(runtime, "dashboard.html")   # one file, opens in any browser
  ```

  It is a *view* of the trail — one HTML document with its CSS, inline-SVG charts
  and script embedded, no CDN, no build step, theme-aware.
- **Monitor** — a live terminal view of a whole fleet:

  ```python
  from ear import Monitor
  Monitor().run({"lending": runtime})   # Ctrl-C to stop
  ```

---

## Provider-agnostic routing

A `Router` sits in front of EAR's dependency-free LLM client and makes the whole
pipeline provider-agnostic — assign it to `runtime.model_binding` and no stage
knows a router is there:

```python
from ear import ModelBinding, Router, RoutingStrategy

router = Router.across(
    ModelBinding(provider="anthropic", model="claude-opus-4-8"),   # list order = priority
    ModelBinding(provider="openai", model="gpt-4o"),
    ModelBinding(provider="groq", model="llama-3.3-70b"),
    strategy=RoutingStrategy.PRIORITY,   # ordered fallback
)
runtime.model_binding = router
```

The strategy decides *who goes first* (`PRIORITY`, `CHEAPEST`, `FREE_FIRST`,
`ROUND_ROBIN`, `WEIGHTED`, `RANDOM`); the Router always walks the ordered list and
falls back on failure. Because config belongs in the environment, a Router can be
built straight from an env var:

```python
# EAR_ROUTER='anthropic/claude-opus-4-8, openai/gpt-4o, groq/llama-3.3-70b'
router = Router.from_env("EAR_ROUTER", strategy=RoutingStrategy.PRIORITY)
```

The selection order, fallback walk and cooldown breaker are plain deterministic
Python, fully testable with fake per-provider clients and no network.

---

## Where to go next

- **[Getting Started](GETTING_STARTED.md)** — install to a running agent.
- **[Authoring Guide](AUTHORING.md)** — the markdown stack reference.
- **[Governance](GOVERNANCE.md)** — policies, approval, budgets, the trail.
- **[Concepts](CONCEPTS.md)** — the model behind the runtime.
