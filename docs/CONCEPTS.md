# Concepts

EAR (Enterprise Agentic Runtime) is a Python package for building agentic
systems by **writing plain-English markdown, not code**. This page explains the
mental model: what the pieces are, how a request flows through them, and the
principles that hold everywhere.

New to EAR? Read this, then follow [Getting Started](GETTING_STARTED.md).

---

## The one idea

> **The model judges; code enforces and records.**

Every *judgment* — which capability fits, whether a passage is relevant, when to
call a tool, whether a decision is sound — is made by the language model at
runtime, against the plain-English stack you wrote. Every *guardrail* — policies,
budgets, authentication, sandbox confinement, the audit trail — is plain Python,
enforced deterministically. So what you stack is exactly what the runtime reasons
with, and everything it does is on the record.

When no model is configured, each reasoning step degrades to an honest
**deterministic fallback** and says so on the trail — a judgment nobody made is
never written down as one. This is why an EAR stack always runs and is always
testable offline.

---

## The stack

You build a runtime by stacking declarations, each one a plain-English markdown
file or object:

```
Intent → Skill → Persona → Workflow → Process → Policy → Runtime → Reasoner
```

| Piece | What it is |
|---|---|
| **Intent** | The request that starts a reasoning cycle — text plus a bag of context facts. |
| **Skill** | A stacked prompt: one capability, described in prose. A Python handler is optional, never required. |
| **Persona** | A set of skills plus standing instructions — the "who" that carries out a step. |
| **Workflow** | An ordered list of steps, each delegated to a persona, governed by its own policies. |
| **Process** | A set of workflows that performs an action end to end. |
| **Policy** | A governance rule — a statement the model judges, with an optional deterministic fallback. |
| **Runtime** | Orchestrates processes, holds memory and the audit trail, and reasons over an intent. |
| **Reasoner** | Assembles the relevant skills, personas and steps into one prompt and reasons with the model. |

For the intent at hand, the runtime composes the workflow's ordered steps, the
personas they delegate to, and the stacked skill prompts into **one assembled
block**, and reasons over it — enforcing the workflow's policies before it runs.

The whole stack is authored as seven markdown files in one directory; see the
[Authoring Guide](AUTHORING.md) for the file reference.

---

## The reasoning cycle

`runtime.reason(intent)` walks one intent through a fixed pipeline. Each stage is
a distinct, model-made judgment, and each writes a stage-labelled record to the
audit trail:

| Stage | What happens |
|---|---|
| **governance** | The `Governor` clears the intent against every applicable policy — the one gate a cycle passes before anything runs. A block is a recorded outcome, not a crash. |
| **discovery** | Which processes are relevant to this intent (model-ranked, keyword fallback). |
| **selection** | Which candidates to run when there is a choice. |
| **scheduling** | The execution order, when there is more than one workflow. |
| **delegation** | Which persona each undelegated step is assigned to, and why. |
| **deliberation** | The Reasoner reasons over the assembled capabilities and memory, producing the decision. (A `Panel` deliberates multi-voiced here when a workflow declares a pattern.) |
| **recall / retrieval** | What memory and which knowledge passages informed the decision, with citations. |
| **explanation / audit** | The prose rationale and a check that the evidence actually supports the decision. |
| **adaptation** | Any durable lesson distilled from the cycle. |

Every record carries the model that produced it (`deterministic-fallback` when no
model was active). See the [audit trail](OPERATIONS.md#the-audit-trail) for the
full stage list.

---

## Markdown in, markdown out

Markdown is EAR's native format on **both** sides of the boundary:

```
IN   intents/<name>.md     the request: #-title, prose, ## Context bullets (facts)
OUT  decisions/<name>.md    the decision, explanation, evidence, and every policy
                            judgment — a refusal is written as Status: BLOCKED,
                            never raised away
OUT  .ear/reasoning.md      the append-only reasoning audit trail
OUT  .ear/session.md        cross-session memory
```

The `Exchange` pairs an intent document with its decision document by filename and
is idempotent — an inbox, not a replay. A blocked cycle produces a decision
document that says it was blocked and why; a refusal is a first-class outcome on
the record.

---

## Memory, experience, adaptation

A runtime carries three distinct kinds of state into the next decision, folded
into the prompt as three separate sections so they never blur together:

- **Memory** — a rolling window of recent cycles kept verbatim, compressed to a
  summary when it overflows (deterministically, or with the model if you pass a
  summarizer).
- **Experience** — decision counts and evidence aggregated across cycles: the
  pattern, before any conclusion is drawn.
- **Adaptations** — durable lessons distilled from experience: standing
  impressions that bias future reasoning rather than raw history.

All three persist between sessions through a `SessionStore` (markdown by default),
restored on load and saved after every cycle.

---

## Zero dependencies

EAR is the Python standard library plus (optionally) a language model. There are
no third-party packages to install — a shipping invariant, verified by running
the full test suite with every third-party package uninstalled.

- LLM providers are spoken to directly over HTTPS (`ear/llm.py`) — Anthropic
  natively, or any OpenAI-compatible endpoint (Azure, Ollama, vLLM, OpenRouter, …)
  selected by a `ModelBinding`, never hardcoded.
- Structured prompting is native (`ear/judgment.py`): the model answers in
  markdown sections, parsed by the same `Section` codec the stack is authored with.
- The dashboard, the MCP client, the Kubernetes provider and the HTTP server are
  all built from the standard library the same way.

---

## Where to go next

- **[Getting Started](GETTING_STARTED.md)** — install to a running, governed agent.
- **[Authoring Guide](AUTHORING.md)** — the complete markdown stack reference.
- **[Governance](GOVERNANCE.md)** — policies, approval gates, budgets, the trail.
- **[Operations](OPERATIONS.md)** — the Exchange, kernel, server, Kubernetes, observability.
