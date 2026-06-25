# EAR — Enterprise Agentic Runtime

EAR is a Python package skeleton for building an enterprise agentic runtime.

Imagine a train: prompts are stacked inside skills, skills are stacked inside
a persona, a persona is stacked into a workflow, workflows are stacked into
processes, policies are mapped onto those processes, processes are
orchestrated by the runtime, and the runtime starts reasoning.

```text
Prompt → Skill → Persona → Workflow → Process → Policy → Runtime → Reasoning
```

Philosophical stack:

```text
Sankalpa  → Prompt / intent
Vidyā     → Skill / capability
Guna      → Persona / behavioural nature
Varna     → Workflow / role-ordering
Karma     → Process / action
Dharma    → Policy / governance
Ksetra    → Runtime battlefield / field of execution
Bhuddi    → Reasoning / discriminative intelligence
```

Manas (the mind) is the LLM provider binding -- model, credentials, call
parameters. It isn't a step in the stack so much as the current that runs
through it: Ksetra activates its Manas before handing the Sankalpa to
Bhuddi, so reasoning runs against a real model instead of the
dependency-free default.

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
from ear import Sankalpa, Vidya, Guna, Varna, Karma, Dharma, Ksetra

runtime = Ksetra(name="Procurement-Kurukshetra")

runtime.add_policy(Dharma(
    name="PO Approval Policy",
    rule="purchase_amount <= approval_limit",
))

process = Karma(name="Create Purchase Order")
process.add_workflow(Varna(name="Procurement Workflow"))
runtime.add_process(process)

result = runtime.reason(Sankalpa(text="Create PO for laptops under approved budget"))
print(result)
```

Without a Manas or DSPy program attached, `Ksetra.reason` falls back to a
deterministic summary of which processes the Sankalpa cleared, so the
example above runs with no LLM credentials. Give the runtime a real mind:

```python
from ear import Manas

runtime.manas = Manas(provider="openai", model="gpt-4o-mini", api_key="sk-...")

result = runtime.reason(Sankalpa(text="Create PO for laptops under approved budget"))
```

`Ksetra.reason` activates `runtime.manas` (configuring DSPy's LM) before
calling `Bhuddi.reason`. With no DSPy program attached, Bhuddi calls that
LM directly; attach a compiled DSPy program for structured reasoning
instead, and it'll run against the same activated Manas LM:

```python
import dspy
from ear.integrations.dspy_backend import make_reasoner

class Decide(dspy.Signature):
    sankalpa: str = dspy.InputField()
    context: dict = dspy.InputField()
    decision: str = dspy.OutputField()

runtime.reasoner = make_reasoner(Decide)
```

## Package layout

```text
ear/
  sankalpa.py   Sankalpa  — prompt / intent
  vidya.py      Vidya     — skill
  guna.py       Guna      — persona (a stack of Vidya skills)
  varna.py      Varna     — workflow (a stack of Guna personas)
  karma.py      Karma     — process (a stack of Varna workflows)
  dharma.py     Dharma    — policy (a guarded rule, safely evaluated — no eval/exec)
  ksetra.py     Ksetra    — runtime (orchestrates Karma processes, enforces Dharma, activates Manas, starts Bhuddi)
  manas.py      Manas     — LLM provider binding (model, credentials, params -> a DSPy LM)
  bhuddi.py     Bhuddi    — reasoning (DSPy-backed or raw Manas LM call, with a dependency-free default)
  integrations/
    dspy_backend.py      DSPy signature/program → Bhuddi
    evolve_backend.py    openevolve — evolve a Vidya's source against an evaluator
    skillopt_backend.py  skillopt   — train a Guna's skill document with ReflACT
```
