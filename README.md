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

Pramāṇa, Smṛti, Anubhava and Saṃskāra are the runtime's memory, wrapped
around every cycle as four distinct layers rather than one blurred together:

```text
Pramāṇa   → Evidence: why this decision was made
Smṛti     → Persistent memory: what happened
Anubhava  → Experience: the pattern across repeated Smriti entries
Saṃskāra  → Adaptation: how future behaviour should change
```

Ksetra writes a Pramana-evidenced entry to Smṛti after every `reason()`
call, folds it into Anubhava, and surfaces all three layers back to Bhuddi
on the next one, so memory actually compounds across cycles instead of
resetting each time.

`Ksetra.reason()` itself runs through one further named pipeline of
runtime operations -- discovery, selection, composition, scheduling,
orchestration, execution, deliberation, decision, validation, recall,
explanation, audit, learning and adaptation -- each its own class instead
of logic folded into one method:

```text
Discover     → Anveṣaṇa     (अन्वेषण) → find Karma processes relevant to a Sankalpa
Select       → Varaṇa       (वरण)    → choose among discovered processes
Compose      → Saṃyojana    (संयोजन)  → assemble their Varna workflows into a plan
Schedule     → Niyojana     (नियोजन)  → order the composed plan
Govern       → Niyamana     (नियमन)   → enforce Dharma policy gates
Initialize   → Ārambha      (आरम्भ)   → activate Manas
Orchestrate  → Samanvaya    (समन्वय)  → coordinate a cycle's execution end to end
Execute      → Anuṣṭhāna    (अनुष्ठान) → run the cycle's Kriya action
Perform      → Kriyā        (क्रिया)   → deliberate, decide, validate
Reason       → Vicāra       (विचार)   → deliberate via Bhuddi
Decide       → Nirṇaya      (निर्णय)   → commit to one decision
Validate     → Parīkṣā      (परीक्षा)  → reject a malformed decision
Remember     → Smaraṇa      (स्मरण)    → recall Smriti context as evidence
Store Memory → Smṛti        (स्मृति)   → what happened
Explain      → Vyākhyā      (व्याख्या) → render why a decision was reached
Audit        → Pariśodhana  (परिशोधन) → inspect evidence for compliance
Learn        → Adhyayana    (अध्ययन)  → fold a cycle into Anubhava experience
Adapt        → Anukūlana    (अनुकूलन) → periodically distill a new Samskara
Evolve       → Pariṇāma     (परिणाम)  → evolve a Vidya skill's source (openevolve)
Optimize     → Utkarṣa      (उत्कर्ष)  → refine a Guna skill document (SkillOpt)
```

Every `reason()` call runs:

```text
Niyamana → Arambha → Anveshana → Varana → Samyojana → Niyojana → Smarana
  → Samanvaya [→ Anushthana → Kriya [→ Vicara → Nirnaya → Pariksha]]
  → Vyakhya → Parishodhana → Smriti → Adhyayana → Anukulana
```

`Niyamana` raises `PermissionError` and stops the cycle before anything
else runs if a Dharma policy is violated. `Anukulana` only calls
`SamskaraBank.learn_from` every `adapt_every` observed cycles (default 5),
not on every single one, so adaptation doesn't spam one impression per
call. `Parinama` and `Utkarsha` are structural, dev-time operations on a
`Vidya` skill or `Guna` persona -- evolving or optimizing code isn't part
of running it, so they sit outside this per-cycle pipeline and are called
directly:

```python
runtime.parinama.evolve(skill, evaluator="path/to/evaluator.py")          # openevolve
trainer = runtime.utkarsha.optimize(config="skillopt.yaml", adapter=my_env)
runtime.utkarsha.apply(persona, "skill-name", trainer.train())
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

## Memory: Pramāṇa, Smṛti, Anubhava and Saṃskāra

AI systems routinely conflate four distinct concerns: *why* a decision was
made, *what* happened, the *pattern* across repeated executions, and *how*
behaviour should change as a result. EAR keeps each as its own layer so one
never silently stands in for another:

```text
Pramāṇa   → Evidence     -- why this decision was made
Smṛti     → Memory       -- what happened
Anubhava  → Experience   -- the pattern across repeated Smriti entries
Saṃskāra  → Adaptation   -- how future behaviour should change
```

Every `Ksetra.reason()` call builds a `Pramana` (which reasoning path was
used, which policies were checked, the context that applied), records it
alongside the decision in `runtime.smriti` as that entry's `evidence`, and
folds the entry into `runtime.anubhava` -- so "why" survives independently
of "what happened", and is never lost when memory compresses.

`runtime.smriti` is a two-layer memory that keeps context bounded as
history grows:

- **`working`** -- the most recent cycles, kept verbatim (bounded by
  `capacity`, default 20).
- **`compressed`** -- once `working` overflows, the oldest entries are
  rolled into one summary string per overflow event. This is the memory
  compression step. Pass an activated LM (`runtime.manas.lm`) to
  `smriti.compress(summarizer=...)` for an LLM-written summary instead of
  the deterministic digest used by default.

```python
runtime.smriti.capacity = 50          # raise/lower the verbatim window
runtime.smriti.compress(summarizer=runtime.manas.lm)  # force an early, LLM-written compression
print(runtime.smriti.context_window())  # compressed history + recent working entries
```

`runtime.anubhava` (an `Anubhava`) aggregates those Smriti entries into
decision counts and the evidence seen along the way -- the pattern a
Samskara is then distilled from, without yet drawing a conclusion:

```python
print(runtime.anubhava.summary())              # ranked decision counts across cycles
print(runtime.anubhava.most_common_decision())  # ("approved", 7), etc.
```

`runtime.samskara` (a `SamskaraBank`) distills durable lessons out of
Anubhava experience -- standing impressions that bias future reasoning
rather than raw history:

```python
learned = runtime.samskara.learn_from(runtime.anubhava, summarizer=runtime.manas.lm)
print(learned.insight)
```

On the next `reason()` call, `Bhuddi` (when running off the raw Manas LM
path) pulls in `runtime.smriti.context_window()`, `runtime.anubhava.summary()`
and any `runtime.samskara.relevant_to(sankalpa.text)` impressions and folds
them into the prompt as three distinct sections, so memory, experience and
learned adaptation each influence the next decision without blurring
together.

## Package layout

```text
ear/
  sankalpa.py   Sankalpa  — prompt / intent
  vidya.py      Vidya     — skill
  guna.py       Guna      — persona (a stack of Vidya skills)
  varna.py      Varna     — workflow (a stack of Guna personas)
  karma.py      Karma     — process (a stack of Varna workflows)
  dharma.py     Dharma    — policy (a guarded rule, safely evaluated — no eval/exec)
  ksetra.py     Ksetra    — runtime (runs every cycle through the full operation pipeline below)
  manas.py      Manas     — LLM provider binding (model, credentials, params -> a DSPy LM)
  pramana.py    Pramana   — evidence for one decision (why), distinct from Smriti (what) and Anubhava (the pattern)
  smriti.py     Smriti    — persistent memory (working + compressed layers; this is the compression step)
  anubhava.py   Anubhava  — experience aggregated from repeated Smriti entries, the missing step before Samskara
  samskara.py   Samskara  — learned adaptations distilled from Anubhava, surfaced back to Bhuddi
  bhuddi.py     Bhuddi    — reasoning (DSPy-backed or raw Manas LM call, fed Smriti/Anubhava/Samskara, with a dependency-free default)
  niyamana.py   Niyamana    — govern: enforce Dharma policy gates
  arambha.py    Arambha     — initialize: activate Manas
  anveshana.py  Anveshana   — discover: find Karma processes relevant to a Sankalpa
  varana.py     Varana      — select: choose among discovered Karma processes
  samyojana.py  Samyojana   — compose: assemble selected processes' Varna workflows into a plan
  niyojana.py   Niyojana    — schedule: order the composed plan
  samanvaya.py  Samanvaya   — orchestrate: coordinate a cycle's execution end to end
  anushthana.py Anushthana  — execute: run the cycle's Kriya action
  kriya.py      Kriya       — perform: chain Vicara -> Nirnaya -> Pariksha
  vicara.py     Vicara      — reason: deliberate via Bhuddi
  nirnaya.py    Nirnaya     — decide: commit to one final decision
  pariksha.py   Pariksha    — validate: reject a malformed decision
  smarana.py    Smarana     — remember: recall Smriti context as evidence
  vyakhya.py    Vyakhya     — explain: render why a decision was reached
  parishodhana.py Parishodhana — audit: inspect evidence for compliance
  adhyayana.py  Adhyayana   — learn: fold a cycle into Anubhava experience
  anukulana.py  Anukulana   — adapt: periodically distill a new Samskara
  parinama.py   Parinama    — evolve: transform a Vidya skill's source (openevolve, dev-time)
  utkarsha.py   Utkarsha    — optimize: refine a Guna skill document (SkillOpt, dev-time)
  integrations/
    dspy_backend.py      DSPy signature/program → Bhuddi
    evolve_backend.py    openevolve — evolve a Vidya's source against an evaluator
    skillopt_backend.py  skillopt   — train a Guna's skill document with ReflACT
```
