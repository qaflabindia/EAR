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

Judgment-laden stages — `Policy` compliance, `Discoverer` relevance
ranking, the `Reasoner`'s decision, the `Explainer`'s prose — reason in
natural language against a live LLM (via [DSPy](https://github.com/stanfordnlp/dspy)),
each with a deterministic, dependency-free fallback so the package stays
fully usable and testable with no LLM configured at all. Structural
stages (`Selector`, `Composer`, `Scheduler`) have no judgment call to
make, so they stay plain Python.

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
Selector     → select       choose among discovered processes
Composer     → compose      assemble their Workflows into a plan
Scheduler    → schedule     order the composed plan
Orchestrator → orchestrate  coordinate a cycle's execution end to end
Executor     → execute      run the cycle's Performer action
Performer    → perform      deliberate, decide, validate
Deliberator  → deliberate   reason via the Reasoner
Decider      → decide       commit to one decision
Validator    → validate     reject a malformed decision
Recaller     → remember     recall Memory context as evidence
Explainer    → explain      render why a decision was reached (LLM-written, f-string fallback)
Auditor      → audit        inspect evidence for compliance
Memory       → store memory what happened
Learner      → learn        fold the cycle into Experience
Adapter      → adapt        periodically distill a new Adaptation
Assessor     → assess       (only with a Goal) judge completion; re-enter the cycle, or stop
```

With a `Goal` attached, the `Assessor` judges the decision after each cycle
and either stops or re-enters the pipeline above, bounded by the goal's
`max_cycles` cap (see [Goals: controlled iteration](#goals-controlled-iteration)).

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

## Provider-agnostic routing: the omni-route `Router`

A single `ModelBinding` speaks to one provider. A `Router` is the
OmniRoute-style, provider-agnostic agent: it stacks *many* provider
bindings — any LiteLLM provider (Anthropic, OpenAI, Gemini, Bedrock, Groq,
Ollama, … 250+), each reading its own key from the environment — behind one
binding, picks which to try first by a **routing strategy**, **falls back**
to the next provider when one errors or is rate-limited, and trips a
**circuit breaker** that benches a failing provider for a cooldown window so
the next call routes around it.

Crucially a `Router` *is* a drop-in `ModelBinding` — same `activate()`,
`lm` and `model_id` surface — so you assign it to `runtime.model_binding`
and the whole pipeline (`Governor`, `Discoverer`, `Policy`, `Reasoner`,
`Explainer`) becomes provider-agnostic without any stage knowing a router
is there.

```python
from ear import ModelBinding, Router, RoutingStrategy

router = Router.across(
    ModelBinding(provider="anthropic", model="claude-opus-4-8", priority=10),
    ModelBinding(provider="openai", model="gpt-4o", priority=20),
    ModelBinding(provider="groq", model="llama-3.3-70b", is_free=True, priority=30),
    strategy=RoutingStrategy.PRIORITY,   # ordered fallback: 10 -> 20 -> 30
)
runtime.model_binding = router
```

The strategy only decides *who goes first*; the Router always walks the
ordered list and falls back on failure:

| Strategy | Orders providers by |
| --- | --- |
| `PRIORITY` | lowest `priority` number first (ordered fallback) |
| `CHEAPEST` | lowest `cost_per_1k` first |
| `FREE_FIRST` | `is_free` providers first, then by priority |
| `ROUND_ROBIN` | rotates the starting provider each call |
| `WEIGHTED` | random order, biased by each binding's `weight` |
| `RANDOM` | uniform random order |

Because the config belongs in the environment, never hardcoded, a Router
can be built straight from an env var — a JSON array of providers, or a
shorthand `provider/model` list:

```python
# EAR_ROUTER='anthropic/claude-opus-4-8, openai/gpt-4o, groq/llama-3.3-70b'
router = Router.from_env("EAR_ROUTER", strategy=RoutingStrategy.PRIORITY)
```

The selection order, the fallback walk and the cooldown breaker are plain,
deterministic Python (`router.order()`, `router.dispatch(...)`), so the
routing behaviour is fully testable with fake per-provider callables and no
LLM configured at all — the same offline/live two-tier testability the rest
of the package keeps.

## Goals: controlled iteration

Without a `Goal`, `Runtime.reason` runs exactly one cycle. A `Goal` turns
that single pass into **bounded, audited iteration**: the runtime re-enters
the cycle — feeding each decision forward — until the goal is met, a
blocker is hit, or a hard `max_cycles` cap is reached.

A `Goal` is `Policy`'s mirror image. A `Policy` is a gate the cycle **must
not cross**; a `Goal` is a finish line the runtime **iterates toward**.
Where a violated policy *stops* a cycle, an unmet goal *re-enters* it.

```python
from ear import Goal

runtime.reason(
    Intent(text="Underwrite a $40k loan", context={"consent": True}),
    goal=Goal(
        statement="A final grade A–E and an approve/decline decision are set.",
        fallback_expression="'approve' in decision or 'decline' in decision",
        max_cycles=3,
    ),
)
```

The `Assessor` judges completion after each cycle — exactly like `Policy`,
it reasons in natural language against the active model (the
`AssessGoalCompletion` signature) and falls back to the goal's safe
`fallback_expression` (over the intent's context plus the special
`decision` variable) when no model is configured, so goal-driven iteration
is fully usable and testable offline. The LLM path can also report a
**blocker** (`needs_input`, `blocked`, `failed`) to stop a loop that cannot
make further progress.

Two properties make this safe for an enterprise runtime rather than a
free-running agent: iteration is **always bounded** by `max_cycles`, and
because every cycle already writes an `Evidence`-backed `Memory` entry, the
whole loop is **audited by construction** — you get the decision trail
across cycles, not an opaque "it thought for a while".

## Progressive skill selection

A `Persona` can carry a large library of `Skill`s without every prompt
being stacked into every reasoning call. The `SkillSelector` stacks only
the skills relevant to the current intent — the same relevance ranking
`Discoverer` already does for whole processes, applied one level down to a
persona's skills:

```text
Discoverer     → ranks Processes relevant to an Intent   (LLM-ranked, keyword fallback)
SkillSelector  → ranks a Persona's Skills the same way   (LLM-ranked, keyword fallback)
```

It is on by default (`Reasoner(skill_selector=SkillSelector(top_k=8))`) and
**short-circuits** — returning every skill, in order — whenever a persona
has `top_k` or fewer skills, so the common case costs nothing and small
personas behave exactly as before. Only when a persona exceeds `top_k` does
it rank (by LLM when a model is active, by keyword overlap offline) and keep
the most relevant. Pass `skill_selector=None` to always stack every skill.

`Skill` also carries **provenance** — `version` and `author` — so a
decision's `Evidence` can be traced back to the exact skill (and version)
that shaped it. That's an auditability win, not just metadata: in an
enterprise runtime you can answer *which version of which capability
produced this decision*.

## Governed tools: govern the action, not just the decision

EAR governs *decisions* through `Policy`. A `Tool` lets it also *act* — and
every action is governed the same way. A `Tool` is declared prompt-first,
exactly like a `Skill`: a name and a plain-English `contract` are required
(an LLM reads the contract to know what the tool does); a Python `handler`
is the optional advanced layer, so a tool can be declared and governed
before it is implemented.

```python
from ear import Tool, ToolPolicy

guru.add_tool(Tool(
    name="pull_bureau",
    contract="Fetch the applicant's credit-bureau score.",
    permissions=["read:bureau"],
    handler=bureau_client.fetch,
))

runtime.add_tool_policy(ToolPolicy(
    name="Bureau Consent",
    statement="A bureau pull is only permitted when the applicant's consent is on file.",
    fallback_expression="consent == True",     # deterministic offline
    tool="pull_bureau",
))

runtime.invoke(guru.get_tool("pull_bureau"), consent=True)   # runs -> "score=720"
runtime.invoke(guru.get_tool("pull_bureau"), consent=False)  # raises PermissionError: blocked by Bureau Consent
```

Every call goes through the `Invoker`, which does two things before and
after the action:

1. **Gates** it — `Governor.govern_tool` checks the runtime's
   `ToolPolicy`s (judged in natural language by the active model, with a
   safe-eval `fallback_expression` over the call's arguments plus the
   special `tool` and `permissions` values). A violated policy raises
   `PermissionError` and the action never runs. `ToolPolicy` is `Policy`
   for actions — same LLM-judged / safe-eval-fallback / reject-unsafe
   engine, no divergent second rule system.
2. **Records** it — the call (allowed *or* blocked) lands in the cycle's
   `Evidence.sources["tool_calls"]`, so what the runtime *did* sits in the
   same audit trail as what it *decided*, never off the books.

```text
Governor.govern         → govern a decision  (Policy gate)
Governor.govern_tool    → govern an action   (ToolPolicy gate) ─┐
Invoker.invoke          → run it, or block it, and record it  ──┴─► Evidence.sources["tool_calls"]
```

This is the piece a plain gateway can't offer: not just reaching many
tools, but reaching them **under governance, with an audit trail**.
Execution isolation for side-effecting handlers (a sandbox) is a separate,
later concern; until then a handler runs in-process behind the policy gate.

> The runtime *decides* to call a tool autonomously (DSPy ReAct) is a
> further reasoning-integration step layered on this foundation; today tools
> are invoked explicitly through `runtime.invoke`, always governed and
> audited.

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
  skill.py         Skill         — a stacked prompt (a capability), reasoned over by the runtime; no handler code required; carries version/author provenance
  skill_selector.py SkillSelector — select: stack only the Skills relevant to an Intent (LLM-ranked, keyword fallback)
  persona.py       Persona       — a stack of Skills (reason) plus optional Tools (act) and standing instructions
  tool.py          Tool          — a declared capability the runtime acts through; prompt-first, handler optional
  tool_policy.py   ToolPolicy    — governance for a tool call (Policy for actions), judged with a safe-eval fallback
  invoker.py       Invoker       — invoke: gate a tool call against ToolPolicies, run or block it, record it as Evidence
  step.py          Step          — one narrated instruction in a Workflow, delegated to a Persona
  workflow.py       Workflow      — an ordered list of Steps (each delegated to a Persona), governed by its own Policies
  process.py       Process       — a stack of Workflows that performs an action
  policy.py        Policy        — governance rule, judged in natural language with a safe-eval fallback; attaches runtime-wide or to a Workflow
  goal.py          Goal          — a completion condition that drives bounded, audited iteration of a reasoning cycle
  assessor.py      Assessor      — assess: judge whether a Goal is met after a cycle (LLM-judged, safe-eval fallback)
  runtime.py       Runtime       — runs every cycle through the full operation pipeline below
  model_binding.py ModelBinding  — LLM provider binding (model, credentials, params -> a DSPy LM)
  router.py        Router        — omni-route: routes across many providers with fallback + cooldown (a drop-in ModelBinding)
  evidence.py      Evidence      — why this decision was made
  memory.py        Memory        — persistent memory (working + compressed layers)
  experience.py    Experience    — the pattern aggregated from repeated Memory entries
  adaptation.py    Adaptation    — learned adaptations distilled from Experience
  reasoner.py      Reasoner      — the deliberation step (DSPy-backed, with a GEPA optimization hook)
  signatures.py    DSPy signatures shared across the LLM-judged stages
  governor.py      Governor      — govern: enforce Policy gates
  initializer.py   Initializer   — initialize: activate the ModelBinding
  discoverer.py    Discoverer    — discover: find Processes relevant to an Intent
  selector.py      Selector      — select: choose among discovered processes
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
  optimizer.py     Optimizer     — optimize: refine a Persona's skill document (SkillOpt, dev-time)
  integrations/
    evolve_backend.py    openevolve — evolve a Skill's source against an evaluator
    skillopt_backend.py  skillopt   — train a Persona's skill document with ReflACT
```
</content>
