# Getting Started with EAR

EAR (Enterprise Agentic Runtime) lets you build an agentic system by
**writing plain-English markdown, not code**. You stack prompts into
skills, skills into a persona, steps into a workflow, workflows into a
process, and policies over the top — then point the runtime at that folder
and it reasons.

This guide takes you from install to a running, governed agent in a few
minutes, then shows where to go next. Everything here works with **zero
dependencies** — EAR is the Python standard library plus (optionally) a
language model.

---

## 1. Install

Requires Python 3.10+. From the repository root:

```bash
pip install -e .
```

That's it — there are no third-party dependencies to pull in. Verify:

```bash
python -c "import ear; print(ear.__version__)"
```

To run the tests (the offline suite needs no API key):

```bash
pip install -e ".[dev]"
python -m pytest -q -k "not live"
```

---

## 2. Your first agent, in 60 seconds (no API key needed)

EAR runs even with no model configured — every reasoning step has a
deterministic fallback, so you can see the whole machine work offline first.

```python
from ear import Runtime, Persona, Skill, Workflow, Process, Policy, Intent

# Stack a capability: a persona with a skill
guru = Persona(name="Credit Risk Guru", instructions="Underwrite conservatively.")
guru.add_skill(Skill(name="risk_grade", prompt="Combine the score tier and DTI band into a grade A–E."))

# Narrate a workflow and govern it with a policy
workflow = Workflow(name="Underwriting Workflow")
workflow.add_step("Band the credit profile and assign a risk grade.", persona=guru)
workflow.add_step("Decide approve or decline against the grade.", persona=guru)
workflow.add_policy(Policy(
    name="Loan Amount Cap",
    statement="The loan must not exceed $75,000.",
    fallback_expression="loan_amount <= 75000",   # enforced even with no model
))

# Assemble a process and a runtime
process = Process(name="Underwrite Consumer Loan")
process.add_workflow(workflow)
runtime = Runtime(name="Credit Risk Runtime")
runtime.add_process(process)

# Reason over an intent
decision = runtime.reason(Intent(text="Underwrite a loan", context={"loan_amount": 20000}))
print(decision)

# The Loan Amount Cap blocks anything over $75k — governance, enforced in code
try:
    runtime.reason(Intent(text="Underwrite a big loan", context={"loan_amount": 200000}))
except PermissionError as blocked:
    print("Blocked:", blocked)
```

You just ran the full pipeline — discovery, selection, composition,
scheduling, governance, deliberation, explanation, audit, memory — offline.
A blocked cycle is a first-class outcome, not an error you have to guess at.

---

## 3. The real way: author a stack in markdown

Code is fine for a quick start, but EAR is designed so the whole system is
**a folder of markdown files** — reviewable, diffable, and editable by
someone who doesn't write Python. The loader reads whichever of these files
exist:

| File | Holds | Shape |
|---|---|---|
| `skills.md` | prompts → skills | `## skill name` + prose prompt |
| `persona.md` | skills → personas | prose instructions + `Skills: a, b` |
| `workflow.md` | steps → workflows | numbered steps, `(Persona)` delegates, `Policies:` |
| `process.md` | workflows → processes | prose + `Workflows: a, b` |
| `policy.md` | governance | prose statement + `Fallback:` / `Applies to:` |
| `tenant.md` | the org this stack belongs to | `Org id:`, fiscal year, timezone — optional, defaults to the "default" tenant |
| `memory.md` | the operating strategy | model, memory, tools, knowledge, audit, … |

A minimal stack directory:

**`skills.md`**
```markdown
# Skills

## assign_risk_grade

Read the applicant's credit score and debt-to-income ratio from the intent
context, and assign a risk grade from A (strongest) to E (weakest).

## decide_application

Given the risk grade, decide approve or decline, and say why in one sentence.
```

**`persona.md`**
```markdown
# Personas

## Credit Risk Guru

Underwrite conservatively; prefer a clear decline over a risky approval.

Skills: assign_risk_grade, decide_application
```

**`workflow.md`**
```markdown
# Workflows

## Underwriting Workflow

1. Assign a risk grade from the applicant's profile. (Credit Risk Guru)
2. Decide approve or decline against the grade. (Credit Risk Guru)

Policies: Loan Amount Cap
```

**`process.md`**
```markdown
# Credit Risk Runtime

## Underwrite Consumer Loan

Evaluates a consumer loan application end to end.

Workflows: Underwriting Workflow
```

**`policy.md`**
```markdown
# Policies

## Loan Amount Cap

The loan must not exceed $75,000.

Fallback: loan_amount <= 75000
Applies to: Underwriting Workflow
```

Load and run it:

```python
from ear import load_runtime, Intent

runtime = load_runtime("path/to/your/stack")
print(runtime.reason(Intent(text="Underwrite a loan", context={"loan_amount": 20000})))
```

> A complete, richer example lives in [`examples/credit_risk_stack/`](../examples/credit_risk_stack) —
> copy it as a starting point.

---

## 4. Add a real language model

A model is selected in **`memory.md`**, in prose — never hardcoded, and the
credential is always an environment-variable *name*, never a key in the file:

**`memory.md`**
```markdown
# Memory & Strategy

## Model Selection

Reason with anthropic/claude-opus-4-8, reading the credential from
ANTHROPIC_API_KEY, at a temperature of 0.2. When the credential is absent
from the environment, the runtime stays on its deterministic fallback.
```

Set the key in your environment (never in the repo):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Now the same `load_runtime(...).reason(...)` reasons with the model. EAR
speaks Anthropic's API natively, or any OpenAI-compatible endpoint (Azure,
Ollama, vLLM, OpenRouter, …) — write `openai/gpt-4o` and an `api_base`
instead. With no key present, everything still runs on the deterministic
fallback, so your stack is always testable offline.

---

## 5. See what happened

Every reasoning step is on an audit trail. Add an audit section to
`memory.md` to persist it:

```markdown
## Reasoning Audit Trail

Log every reasoning step to `.ear/reasoning.md`, append-only across sessions.
```

Then look at it as a **self-contained HTML dashboard** (the TensorBoard
equivalent, zero dependencies):

```python
from ear import Dashboard
Dashboard().write(runtime, "dashboard.html")     # open in any browser
```

Or watch a whole fleet live in the terminal — the factory assembly line:

```python
from ear import Monitor
Monitor().run({"lending": runtime})              # Ctrl-C to stop
```

---

## 6. Grow the stack (all in markdown)

Everything below is added by writing more markdown — no new code:

- **Knowledge (RAG):** a `## Knowledge` section in `memory.md` naming source
  files; the runtime retrieves and cites them (BM25 + a model-written gist
  index). See the README's *Knowledge* section.
- **Contracts:** a `### Deliverable` block under a workflow declares the
  typed facts a decision must carry; they're extracted and judged at runtime.
- **Approval gates:** `Approval: required` on a policy parks a cycle for a
  human verdict (a markdown approval document releases it), with optional
  `Approvers:` allow-lists.
- **Evaluation:** drop markdown evals in an `evaluations/` folder and run
  `Examiner().examine(runtime, "evaluations")` — graded, with regression
  diffs between runs.
- **Session Goals:** `runtime.pursue("reach a clear decision", intent)` keeps
  driving cycles until the goal is met, blocked, or a budget is spent.
- **Sandbox:** a `## Sandbox` section gives each instance an isolated,
  resource-limited workspace with confined file/shell tools.

---

## 7. Run it as a service

When you're ready to go server-side, the same runtimes become processes on
a scheduler:

```python
from ear import Kernel, Server

# The kernel: run work when there's work, sleep until an interrupt otherwise
kernel = Kernel()
kernel.register("lending", runtime)
kernel.submit("lending", Intent(text="Underwrite", context={"loan_amount": 20000}))
kernel.schedule("lending", Intent(text="nightly review"), every=86400)  # recurring
kernel.start()

# The HTTP control plane over the kernel (token-authed via EAR_SERVER_TOKEN)
Server(stacks_root="./stacks", port=8080).serve()
```
```bash
python -m ear.server --stacks ./stacks --port 8080
```

To run each instance on **Kubernetes** (Jobs for one-off cycles, CronJobs
for recurring tasks — spoken natively over the K8s REST API, no SDK), see
`ear.k8s` and the README's *Kubernetes* section. The in-pod entrypoint is
`python -m ear.run <stack>`.

**Multi-tenant boundary:** when instances belong to different orgs
(`tenant.md`'s `Org id:`), pass a `Claim` (`ear.identity`) alongside the
work — `kernel.submit("lending", intent, claim=Claim(subject="alice",
org_ids=("org_acme_prod",)))` or `runtime.reason(intent, claim=claim)`
directly. A Claim not authorized for the target instance's `org_id`
refuses the cycle before it starts; omit `claim` entirely and nothing
changes from before this existed.

---

## Where to go next

- **[README.md](../README.md)** — the full feature reference (governance,
  knowledge, panels, journeys, the dashboard/Gantt, the trail, the kernel,
  the server, Kubernetes).
- **[examples/credit_risk_stack/](../examples/credit_risk_stack)** — a
  complete, runnable stack with knowledge and evaluations.
- **[docs/IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** — how the native
  parity features were designed and built.

## The one rule to remember

**The model judges; code enforces and records.** Every judgment is the
model's, made at runtime against your plain-English stack; every guardrail —
policies, budgets, auth, confinement, the tamper-evident trail — is
hardwired. Offline, each step degrades to an honest deterministic fallback
and says so. So what you stack is exactly what the runtime reasons with, and
everything it does is on the record.
```
