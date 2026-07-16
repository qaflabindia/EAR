# Getting Started

EAR lets you build a governed agentic system by **writing plain-English markdown,
not code**. This guide takes you from install to a running agent in a few
minutes. Everything here works with **zero dependencies** and runs offline — a
language model is optional.

For the mental model behind it, read [Concepts](CONCEPTS.md) first (or after).

---

## 1. Install

Requires Python 3.10+. From the repository root:

```bash
pip install -e .
```

There are no third-party dependencies. Verify:

```bash
python -c "import ear; print(ear.__version__)"
```

To run the tests (the offline suite needs no API key):

```bash
pip install -e ".[dev]"
python -m pytest -q -k "not live"
```

---

## 2. Your first agent, offline (no API key)

EAR runs even with no model configured — every reasoning step has a deterministic
fallback, so you can watch the whole machine work before wiring up a model.

```python
from ear import Runtime, Persona, Skill, Workflow, Process, Policy, Intent

# A persona with a skill
guru = Persona(name="Credit Risk Guru", instructions="Underwrite conservatively.")
guru.add_skill(Skill(name="risk_grade", prompt="Combine the score tier and DTI band into a grade A–E."))

# A workflow, governed by a policy
workflow = Workflow(name="Underwriting Workflow")
workflow.add_step("Band the credit profile and assign a risk grade.", persona=guru)
workflow.add_step("Decide approve or decline against the grade.", persona=guru)
workflow.add_policy(Policy(
    name="Loan Amount Cap",
    statement="The loan must not exceed $75,000.",
    fallback_expression="loan_amount <= 75000",   # enforced even with no model
))

# Assemble and reason
process = Process(name="Underwrite Consumer Loan")
process.add_workflow(workflow)
runtime = Runtime(name="Credit Risk Runtime")
runtime.add_process(process)

decision = runtime.reason(Intent(text="Underwrite a loan", context={"loan_amount": 20000}))
print(decision)

# The cap blocks anything over $75k — governance, enforced in code
try:
    runtime.reason(Intent(text="Underwrite a big loan", context={"loan_amount": 200000}))
except PermissionError as blocked:
    print("Blocked:", blocked)
```

A blocked cycle is a first-class outcome, not an error you have to guess at.

---

## 3. The real way: author a stack in markdown

Code is fine for a quick start, but EAR is designed so the whole system is **a
folder of markdown files** — reviewable, diffable, and editable by someone who
doesn't write Python. The loader reads whichever of these exist:

| File | Holds |
|---|---|
| `skills.md` | prompts → skills |
| `persona.md` | skills → personas |
| `workflow.md` | steps → workflows |
| `process.md` | workflows → processes |
| `policy.md` | governance |
| `tenant.md` | the org this stack belongs to (optional) |
| `memory.md` | the operating strategy (model, memory, tools, knowledge, audit) |

A minimal stack:

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

> A complete, richer example lives in
> [`examples/credit_risk_stack/`](../examples/credit_risk_stack) — copy it as a
> starting point. The full file reference is the [Authoring Guide](AUTHORING.md).

---

## 4. Add a real language model

A model is selected in **`memory.md`**, in prose — never hardcoded, and the
credential is always an environment-variable *name*, never a key in the file:

**`memory.md`**
```markdown
# Memory & Strategy

## Model Selection

Reason with anthropic/claude-opus-4-8, reading the credential from
ANTHROPIC_API_KEY, at a temperature of 0.2. When the credential is absent from
the environment, the runtime stays on its deterministic fallback.
```

Set the key in your environment (never in the repo):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Now the same `load_runtime(...).reason(...)` reasons with the model. EAR speaks
Anthropic's API natively, or any OpenAI-compatible endpoint — write
`openai/gpt-4o` and an `api_base` instead. With no key present, everything still
runs on the deterministic fallback, so your stack is always testable offline.

---

## 5. See what happened

Every reasoning step is on an audit trail. Add an audit section to `memory.md` to
persist it:

```markdown
## Reasoning Audit Trail

Log every reasoning step to `.ear/reasoning.md`, append-only across sessions.
```

Then read it, or render a self-contained HTML dashboard:

```python
print(runtime.reasoning_log.render())            # the skim view

from ear import Dashboard
Dashboard().write(runtime, "dashboard.html")     # open in any browser
```

The trail is tamper-evident and the dashboard is a plain file with no
dependencies — see [Operations](OPERATIONS.md) for both.

---

## 6. Grow the stack

Everything below is added by writing more markdown — no new code. Each links to
its full treatment:

- **Knowledge (RAG)** — cite source files or URLs; see [Authoring → Knowledge](AUTHORING.md#knowledge).
- **Contracts** — declare the typed facts a decision must carry; see [Authoring → Contracts](AUTHORING.md#contracts-typed-deliverables).
- **Approval gates** — park a cycle for a human verdict; see [Governance → Approval gates](GOVERNANCE.md#approval-gates).
- **Panels** — convene personas as a multi-voice deliberation; see [Authoring → Panels](AUTHORING.md#panels-multi-persona-deliberation).
- **Journeys** — durable, resumable, prose-routed execution; see [Operations → Journeys](OPERATIONS.md#journeys-durable-resumable-execution).
- **Session goals** — a completion condition the runtime pursues on its own; see [Operations → Session goals](OPERATIONS.md#session-goals).
- **Sandbox** — an isolated, resource-limited workspace per instance; see [Governance → Sandbox](GOVERNANCE.md#sandbox).

---

## 7. Run it as a service

When you're ready to go server-side, the same runtimes become processes on a
scheduler, an HTTP control plane, or pods on Kubernetes:

```python
from ear import Kernel, Server

kernel = Kernel()
kernel.register("lending", runtime)
kernel.submit("lending", Intent(text="Underwrite", context={"loan_amount": 20000}))
kernel.schedule("lending", Intent(text="nightly review"), every=86400)
kernel.start()

Server(stacks_root="./stacks", port=8080).serve()   # token-authed HTTP front door
```

See [Operations](OPERATIONS.md) for the kernel, the server, Kubernetes, and
monitoring.

---

## The one rule to remember

**The model judges; code enforces and records.** Every judgment is the model's,
made at runtime against your plain-English stack; every guardrail — policies,
budgets, auth, confinement, the tamper-evident trail — is hardwired. Offline, each
step degrades to an honest deterministic fallback and says so.
