# EAR: An Enterprise Agentic Runtime with Decomposed Governance, Reasoning, and Memory (English-Terminology Edition)

*A technical whitepaper on the EAR (`ear`) Python package*

> **Note on terminology.** The underlying source code names every class
> after a Sanskrit term (`Ksetra`, `Dharma`, `Pariksha`, and so on) — see
> `docs/WHITEPAPER.md` for that original terminology and the full
> glossary. This edition describes the identical architecture using only
> English terms in prose, with the corresponding class name given once in
> backticks at first use per section purely so a reader can locate the
> implementation in the codebase.

## Glossary (English term → implementing class)

| English term | Implementing class |
|---|---|
| Intent | `Sankalpa` |
| Skill | `Vidya` |
| Persona | `Guna` |
| Workflow | `Varna` |
| Process | `Karma` |
| Policy | `Dharma` |
| Runtime | `Ksetra` |
| Reasoner | `Bhuddi` |
| Model binding | `Manas` |
| Evidence | `Pramana` |
| Memory | `Smriti` |
| Experience | `Anubhava` |
| Adaptation | `Samskara` |
| Governor (govern) | `Niyamana` |
| Initializer (initialize) | `Arambha` |
| Discoverer (discover) | `Anveshana` |
| Selector (select) | `Varana` |
| Composer (compose) | `Samyojana` |
| Scheduler (schedule) | `Niyojana` |
| Orchestrator (orchestrate) | `Samanvaya` |
| Executor (execute) | `Anushthana` |
| Performer (perform) | `Kriya` |
| Deliberator (deliberate) | `Vicara` |
| Decider (decide) | `Nirnaya` |
| Validator (validate) | `Pariksha` |
| Recaller (remember) | `Smarana` |
| Explainer (explain) | `Vyakhya` |
| Auditor (audit) | `Parishodhana` |
| Learner (learn) | `Adhyayana` |
| Adapter (adapt) | `Anukulana` |
| Evolver (evolve, dev-time only) | `Parinama` |
| Optimizer (optimize, dev-time only) | `Utkarsha` |

## Abstract

Production agentic systems routinely collapse several distinct
concerns — policy enforcement, candidate discovery, plan composition,
scheduling, deliberation, decision validation, evidentiary justification,
persistent memory, and learned adaptation — into a single opaque
"reason" or "run" call. This conflation makes such systems difficult to
audit, difficult to govern, and difficult to reason about formally,
precisely in domains — financial services, healthcare, public-sector
benefits determination — where *why* a decision was reached must be
reconstructable independently of *what* was decided. This paper presents
**EAR (Enterprise Agentic Runtime)**, a Python reference implementation
that decomposes an agentic decision cycle into nineteen named,
independently inspectable, independently swappable stages, organized
around an eight-layer stack (Intent → Skill → Persona → Workflow →
Process → Policy → Runtime → Reasoner) and a four-layer memory model
(Evidence → Memory → Experience → Adaptation) that keeps justification,
history, aggregated pattern, and learned change as separate artifacts
rather than one blended state. We describe the architecture, formalize
the per-cycle pipeline, detail the Validator layer that gates every
pipeline stage's output, and present a worked case study — a credit-risk
underwriting runtime — that exercises governance gates, checker-layer
interception, and multi-cycle memory consolidation against a
regulated-domain decision problem. We close with a discussion of the
architecture's auditability properties, its current limitations, and
directions for future work.

## 1. Introduction

### 1.1 Motivation

The dominant pattern in contemporary agent frameworks is to expose a
single high-level entry point — `agent.run(prompt)`,
`chain.invoke(input)`, `executor.call(task)` — behind which an arbitrary
amount of tool selection, planning, reasoning, and memory management
occurs without a stable, named seam between them. This is convenient for
rapid prototyping but creates three concrete problems once such a system
is deployed against a regulated or otherwise consequential decision:

1. **No fixed point for governance.** If policy enforcement is just
   another step the language model might or might not take, a policy
   violation can be silently reasoned around rather than being a hard gate
   the cycle cannot pass.
2. **No fixed point for evidence.** If "why" a decision was reached is
   recoverable only by re-reading a transcript of intermediate model
   outputs, there is no first-class, queryable artifact that an auditor,
   regulator, or downstream consumer can inspect independently of the
   decision text itself.
3. **No fixed point for validation between stages.** If a planning stage's
   malformed output is consumed directly by an execution stage, errors
   propagate silently rather than failing fast at the boundary where they
   were introduced.

EAR's design thesis is that each of these problems has the same root
cause — collapsing distinct operations into one undifferentiated
call — and the same remedy: give every operation its own named class,
its own single responsibility, and a fixed position in an explicit
pipeline.

### 1.2 Contribution

This paper documents:

- A **stack decomposition** (Section 3) separating intent, capability,
  behavioural nature, workflow ordering, process, policy, runtime, and
  reasoning into eight distinct classes.
- A **pipeline decomposition** (Section 4) of the runtime's per-cycle
  execution into nineteen named stages spanning governance,
  discovery/selection/composition/scheduling, orchestrated execution,
  deliberation/decision/validation, and post-hoc remembrance,
  explanation, audit, learning, and adaptation.
- A **maker–checker validation layer** (Section 5) that independently
  validates the output of every stage that produces structured data for a
  downstream stage to consume, rather than letting each stage
  re-implement its own ad hoc validation or, worse, omit it.
- A **four-layer memory model** (Section 6) — evidence, persistent
  history, aggregated experience, and distilled adaptation — that keeps
  "why", "what", "the pattern", and "the lesson" as four separate,
  independently consumable artifacts.
- A **worked case study** (Section 7) applying the runtime to personal-loan
  underwriting, a domain where the governance, evidence, and audit
  properties above are not academic conveniences but documented
  regulatory expectations.

## 2. Background and Related Work

Decomposing agent cognition into discrete stages is not itself novel.
ReAct-style prompting (Yao et al., 2022) already separates a model's
reasoning trace from its acting trace at the prompt level. Planner/executor
splits are common in classical AI planning and have re-appeared in LLM
agent frameworks as "plan-and-execute" patterns. What is comparatively rare
is making *every* stage between intent and decision — not just
reasoning-versus-acting — a first-class, independently named, independently
testable unit, and pairing that decomposition with an equally explicit
separation of memory concerns.

The memory side of EAR's design responds to a parallel observation from
the broader memory-augmented-agent literature: most systems that claim
"memory" conflate raw history (what happened), the pattern across repeated
history (experience), and the standing belief that pattern should produce
(adaptation) into one data structure — typically a vector store of past
transcripts that simultaneously serves as the system's record, its
statistics, and its bias. EAR instead treats these as a pipeline in their
own right: the Memory layer records, the Experience layer aggregates, the
Adaptation layer distills, and the Evidence layer separately preserves the
justification for each individual decision, independent of all three.

In regulated domains, the closest analogue to EAR's Validator layer is the
"maker–checker" (or "four-eyes") control already mandated by banking
operational-risk practice, in which one party's output (the "maker") is
independently reviewed by a second party (the "checker") before it is
acted upon. EAR applies that same control structure to *pipeline stages*
rather than to *human reviewers*: every maker stage's output passes
through a dedicated checker before the next stage is allowed to trust it.
Similarly, the evidentiary separation EAR enforces between a decision and
its justification mirrors the expectations articulated in interagency
model-risk-management guidance such as SR 11-7 (Board of Governors of the
Federal Reserve System / OCC, 2011), which requires that a model's outputs
be independently explainable and auditable, and in fair-lending statutes
such as the Equal Credit Opportunity Act, which require that adverse
credit decisions be accompanied by a specific, reconstructable reason.

## 3. The Eight-Layer Stack

EAR maps an eight-stage engineering stack onto eight classes, each named
so that the class's *role* is fixed independent of its implementation:

| Engineering role | English term | Implementing class | Responsibility |
|---|---|---|---|
| Prompt / intent | Intent | `Sankalpa` | The resolved intent that starts a cycle: free text plus a structured context dictionary. |
| Skill / capability | Skill | `Vidya` | One addressable capability with an optional handler — the unit prompts are "stacked inside." |
| Persona / behavioural nature | Persona | `Guna` | A stack of Skills plus standing instructions, giving the skills a coherent voice. |
| Workflow / role-ordering | Workflow | `Varna` | An ordered stack of Personas — *who* acts, in what sequence. |
| Process / action | Process | `Karma` | A stack of Workflows the runtime can discover and run. |
| Policy / governance | Policy | `Dharma` | A safely-evaluated rule (no `eval`/`exec`) that gates a Process's execution. |
| Runtime / field of execution | Runtime | `Ksetra` | The field of execution: holds processes, policies, and every pipeline stage; runs the full cycle. |
| Reasoning / discriminative intelligence | Reasoner | `Bhuddi` | Resolves an Intent, against a compiled DSPy program, an activated LLM, or a dependency-free deterministic fallback. |

The Model Binding (`Manas`) is deliberately *not* a ninth stack layer. It
is the LLM provider binding — model identifier, credentials, call
parameters — and is better understood as the current that runs through
the stack: the Runtime activates its Model Binding immediately before
handing an Intent to the Reasoner, so that whichever reasoning path the
Reasoner takes runs against a properly configured language model rather
than an unconfigured one. A Runtime with no Model Binding attached is not
a degraded configuration; it is a deliberately supported mode in which the
Reasoner falls back to a deterministic, dependency-free summary — meaning
the entire governance, validation, and audit apparatus described in this
paper is exercised whether or not a language model is present.

## 4. The Runtime Pipeline

A single reasoning call advances one cycle through nineteen named stages,
grouped here by function rather than by call order for exposition, with
their call order given in Section 4.2.

### 4.1 Stage catalogue

| Group | English term | Operation | Implementing class |
|---|---|---|---|
| Governance | Governor | Govern — enforce Policy gates | `Niyamana` |
| Initialization | Initializer | Initialize — activate the Model Binding | `Arambha` |
| Planning | Discoverer | Discover — find relevant Process candidates | `Anveshana` |
| Planning | Selector | Select — choose among discovered candidates | `Varana` |
| Planning | Composer | Compose — assemble selected Workflows into a plan | `Samyojana` |
| Planning | Scheduler | Schedule — order the composed plan | `Niyojana` |
| Coordination | Orchestrator | Orchestrate — coordinate a cycle's execution end to end | `Samanvaya` |
| Coordination | Executor | Execute — run the cycle's Performer action | `Anushthana` |
| Coordination | Performer | Perform — chain deliberate → decide → validate | `Kriya` |
| Core decision | Deliberator | Reason — deliberate via the Reasoner | `Vicara` |
| Core decision | Decider | Decide — commit to one decision | `Nirnaya` |
| Validation | Validator | Validate — checker layer for every maker stage's output | `Pariksha` |
| Memory recall | Recaller | Remember — recall Memory context as evidence | `Smarana` |
| Explanation | Explainer | Explain — render why a decision was reached | `Vyakhya` |
| Compliance | Auditor | Audit — inspect evidence for compliance | `Parishodhana` |
| Memory write | Memory | Store memory — record what happened | `Smriti` |
| Learning | Learner | Learn — fold a cycle into Experience | `Adhyayana` |
| Adaptation | Adapter | Adapt — periodically distill a new Adaptation impression | `Anukulana` |
| *(dev-time, out of cycle)* | Evolver | Evolve — transform a Skill's source (openevolve) | `Parinama` |
| *(dev-time, out of cycle)* | Optimizer | Optimize — refine a Persona's skill document (SkillOpt) | `Utkarsha` |

Each row is a separate class with one public method. None of them import
the Runtime class; the Runtime imports all of them. This is a deliberate
acyclic-dependency constraint: every stage is independently testable in
isolation, with the runtime as the only thing that knows how they compose.

### 4.2 Control flow

```text
Govern → Initialize → Discover → Validate → Select → Validate → Compose
  → Validate → Schedule → Validate → Remember
  → Orchestrate [→ Execute → Perform [→ Deliberate → Decide → Validate]]
  → Explain → Audit → Store memory → Learn → Adapt
```

Two properties of this control flow are load-bearing rather than
cosmetic:

**Governance is a hard gate, not a soft step.** The Governor's check runs
*before* the Initializer even activates the runtime's Model Binding. If
any Policy is violated, the reasoning call raises `PermissionError`
immediately; no candidate is discovered, no plan is composed, no Memory
entry is written. A policy violation cannot be reasoned around because
reasoning never starts.

**Adaptation is throttled, not continuous.** The Adapter only invokes the
Adaptation bank's distillation step every `adapt_every` *observed* cycles
(default five), guarded by a modulus check against the Experience layer's
observation count. This is an explicit design choice against the failure
mode of producing one fresh, noisy "lesson" per cycle; Adaptation entries
are meant to be durable impressions distilled from a body of experience,
not a running commentary on the most recent decision.

## 5. The Validator: A Maker–Checker Layer for Pipeline Stages

Four stages in the planning group — Discoverer, Selector, Composer,
Scheduler — each produce a typed list that the next stage consumes
without re-validating: the Discoverer and Selector each produce a list of
Processes; the Composer and Scheduler each produce a list of Workflows.
The Decider produces an arbitrary decision value. Letting each of these
five "maker" stages own its own validation invites two failure modes:
validation logic duplicated five times with five chances to diverge, or
validation omitted entirely on the stages whose authors judged it
unnecessary at the time.

The Validator (`ear/pariksha.py`) is instead the single checker every
maker stage's output passes through, exposing one method per maker stage:
validate the discovered candidates, validate the selection, validate the
composed plan, validate the schedule, and — separately — validate the
final decision. The four list-validating methods share one private
routine that raises a type error if the output is not a list, or if it
contains any element of the wrong type — naming the offending stage in
the error message so a failure is immediately attributable to the stage
that produced it. The decision-validating method instead rejects only a
blank string, since a structurally arbitrary decision value cannot be
type-checked the way a list of Processes or Workflows can.

Two design decisions here are deliberate rather than incidental:

- **Empty lists are valid.** A runtime with zero registered Processes is
  an explicitly supported configuration (the Discoverer returns the empty
  list it was given rather than raising), so the Validator checks *type*,
  not *non-emptiness*.
- **Two Validator instances exist by construction.** The Runtime holds one
  Validator that checks the four planning-stage outputs; the Performer
  (nested inside the Executor, inside the Orchestrator) holds a second
  Validator that checks the final decision. Since the Validator is
  stateless — every method is a pure function of its arguments — this
  duplication carries no behavioural risk, but it is worth surfacing
  explicitly: a future refactor collapsing both onto one shared instance
  would be safe, not merely convenient.

This is the architecture's maker–checker control made literal: every
producer of structured data for a downstream consumer has a corresponding,
independently invoked checker, and that checker's rejection — a type error
naming the offending stage — happens at the boundary where the malformed
data was introduced, not several stages later where its effects would
otherwise be much harder to trace back to a root cause.

## 6. Memory: Evidence, Memory, Experience, Adaptation

EAR treats four memory-adjacent concerns as four separate artifacts rather
than one blended state:

| Layer | Question answered | Implementing class |
|---|---|---|
| Evidence | *Why* was this particular decision made? | `Pramana` |
| Memory | *What* happened? | `Smriti` / `SmritiEntry` |
| Experience | What *pattern* holds across repeated Memory entries? | `Anubhava` |
| Adaptation | How should future behaviour *change* as a result? | `Samskara` / `SamskaraBank` |

Each reasoning call builds one Evidence record recording which reasoning
path resolved the decision (a compiled DSPy program, an activated Model
Binding, or the Reasoner's dependency-free default), which Policies were
checked, the Intent's input context, the composed plan, and what the
Recaller recalled from memory at decision time. The Explainer then renders
a human-readable explanation directly from that Evidence record, and the
Auditor marks the Evidence record as inspected — all *before* the Memory
layer writes the decision and its evidence to persistent memory as one
entry. Because the Evidence record is attached to the entry as a distinct
field rather than interpolated into the decision text, "why" remains
queryable independently of "what" indefinitely — including after the
Memory layer's own compression step (Section 6.1) has long since discarded
the verbatim entry.

### 6.1 Bounded memory via two-layer compression

The Memory layer keeps a "working" set of recent entries verbatim, bounded
by a capacity (default twenty). Once that set exceeds capacity, the oldest
overflow entries are rolled into one new summary string appended to a
"compressed" set — by default a deterministic digest of the overflowing
decisions, or, if an activated Model Binding's language model is supplied
as a summarizer, an LLM-written summary. A context-window method renders
the concatenation of compressed history and recent working entries as the
string the Reasoner's default path folds into its prompt. This keeps the
context a reasoning call sees bounded as history grows, without ever
discarding the existence of older cycles outright.

### 6.2 From memory to experience to adaptation

The Learner folds each newly written Memory entry into the runtime's
Experience layer, which maintains a running count of decisions seen and
the full list of Evidence records observed along the way — aggregation
without yet drawing a conclusion. The Adapter is the step that *does* draw
a conclusion, but only every `adapt_every` observed cycles: it invokes the
Adaptation bank's distillation method, which either reports the single
most frequent decision in the aggregated experience (the deterministic
default) or asks an LLM to state one durable lesson in a sentence (when a
summarizer is supplied), and appends the result as a new Adaptation
impression. On the *next* cycle, the Reasoner's default reasoning path
retrieves any Adaptation impressions relevant to the new intent — a
keyword-overlap lookup over the impression text — and folds any matching
impressions into its prompt as a third, clearly labelled section alongside
the Memory layer's history and the Experience layer's summary. Memory,
experience, and adaptation thus each influence subsequent reasoning as
three distinct, separately inspectable inputs, rather than one blurred
"context" blob.

## 7. Case Study: A Credit-Risk Underwriting Runtime

To validate the architecture against a realistic, regulation-adjacent
decision problem rather than a synthetic toy, we instantiated a complete
Runtime — "Credit-Risk-Guru" — for personal-loan underwriting and executed
it end to end (`examples/credit_risk_guru_ksetra.ipynb`).

### 7.1 Construction

Three Skills perform feature derivation: one bands a FICO score into
prime/near-prime/subprime/deep-subprime, one bands a debt-to-income ratio
into low/moderate/high, and one combines both bands into a letter grade
via a fixed lookup table. These are stacked into a Persona, "Credit Risk
Guru," with standing instructions to underwrite conservatively and band
every applicant before any approval reasoning happens. The Persona is
stacked into a single-persona Workflow ("Underwriting Workflow"), which is
in turn stacked into a Process ("Personal Loan Underwriting"). Four
Policies are registered directly on the runtime:

| Policy | Rule |
|---|---|
| Minimum Credit Score | `credit_score >= 620` |
| Debt-to-Income Ceiling | `debt_to_income_ratio <= 0.45` |
| No Active Defaults | `existing_defaults == 0` |
| Loan Amount Cap | `loan_amount <= 75000` |

Each policy rule is evaluated by a restricted AST-walking evaluator that
permits only literals, names, comparisons, boolean logic, and arithmetic —
explicitly forbidding `eval`/`exec` so a policy string supplied by
whoever configures the runtime can never execute arbitrary code.

### 7.2 Observed behaviour

The notebook exercises five distinct properties of the architecture
against this runtime, entirely on the Reasoner's dependency-free default
reasoning path (no LLM credentials required):

1. **A clean approval.** A prime-tier, low-DTI applicant clears all four
   Policy gates; the resulting Memory entry's Evidence record holds the
   composed plan, all four policies checked, and a rendered explanation.
2. **A governance rejection.** An applicant with a 0.52 debt-to-income
   ratio breaches the 0.45 ceiling. The Governor catches this before the
   Initializer runs, and the reasoning call raises `PermissionError`
   naming the violated policy — no decision is reasoned, no memory entry
   is written.
3. **A live Validator interception.** The Discoverer is temporarily
   monkey-patched to return a Workflow instance where the pipeline expects
   Process instances. The Validator raises a type error naming the
   Discoverer's output as the offending data — demonstrating that the
   checker layer documented in Section 5 is load-bearing, not merely
   declared.
4. **Multi-cycle memory consolidation.** A small six-applicant portfolio
   (one borderline applicant plus four further synthetic applicants) is
   run through the same runtime. By the fifth successful cycle, the
   Adapter's five-cycle threshold fires and the Adaptation bank distills
   one new impression from the accumulated Experience layer.
5. **A complete audit trail.** The final entry's Evidence record is
   rendered as a single JSON object containing the policies checked, the
   full applicant context (including derived score tier, DTI band, and
   risk grade), the composed plan, the recalled memory window, the
   rendered explanation, and the Auditor's audit flag — the artifact a
   model-governance review or an adverse-action-notice process would
   need.

### 7.3 Regulatory correspondence

This case study was chosen because personal-loan underwriting is a domain
in which the architectural properties above are not merely good
engineering practice but documented compliance expectations:

- The Policy/Governor gate corresponds to the hard underwriting and
  fair-lending floors a lender's policy already imposes on human
  underwriters — and, as in Section 7.2(2), a breach is structurally
  incapable of being reasoned around, since reasoning has not yet started
  when the gate is checked.
- The Validator checker layer corresponds to the maker–checker / four-eyes
  control already required of manual underwriting workflows, applied
  instead to every automated pipeline stage.
- The Evidence trail corresponds to the independent explainability and
  auditability interagency model-risk-management guidance (e.g., SR 11-7)
  expects of any model used in a credit decision, and to the specific,
  reconstructable reason fair-lending statutes such as the Equal Credit
  Opportunity Act require behind an adverse credit action.
- The Memory → Experience → Adaptation chain corresponds to
  portfolio-level monitoring for unmonitored model drift — precisely the
  failure mode model-risk-management governance exists to catch, and
  precisely what collapsing memory, pattern, and adaptation into one
  undifferentiated state would make invisible.

## 8. Discussion

### 8.1 Strengths

The central claim this architecture supports is **attributable
failure**: when something goes wrong, the named stage responsible is
visible in the stack trace, the exception message, or the Evidence
record, rather than buried inside one large "reason" call. Section 7.2's
Validator demonstration is the clearest instance of this: the failure is
not merely caught, it is caught *and attributed to the exact maker stage
that produced it*. Equally, every stage being its own class with one
public method makes substitution straightforward — swapping the
Discoverer's keyword-overlap discovery for an embedding-based retriever,
or the Scheduler's identity ordering for a priority/dependency scheduler,
requires touching only that one class, since every other stage interacts
with it only through its public method signature.

### 8.2 Limitations

Several aspects of the current implementation are intentionally minimal
and should not be mistaken for completeness:

- **Discovery and selection are keyword-overlap heuristics.** The
  Discoverer and the absence of any ranking in the Selector beyond
  deduplication are explicitly documented as a baseline to be replaced
  with embeddings or a learned retriever for production use.
- **Scheduling has no ordering signal to act on.** The Scheduler returns a
  defensive copy in discovery order because the Workflow class carries no
  priority, dependency, or cost field today; the seam exists, but nothing
  yet populates it.
- **No persistence layer.** The Memory, Experience, and Adaptation classes
  are in-memory; a deployment that needs cycles to survive process
  restarts, or needs Memory shared across concurrent runtime instances,
  must add that layer itself.
- **No concurrency model.** Nothing in the pipeline is documented as
  thread-safe, and the reasoning call is not designed for concurrent
  invocation against shared Memory/Experience/Adaptation state.
- **The dependency-free reasoning fallback is intentionally simple.** The
  Reasoner's default path is a deterministic summary string, sufficient to
  exercise every other stage of the pipeline without an LLM, but it is not
  itself a credit-risk decision engine — production use requires either a
  compiled DSPy program or an activated Model Binding pointed at a real
  model.
- **Two independent Validator instances exist by construction**, which is
  harmless only because the Validator is stateless; introducing any
  per-instance state to the Validator in the future would need to either
  justify the duplication or collapse it to one shared instance.

### 8.3 Future work

Natural extensions include: a learned (rather than keyword-overlap)
Discoverer/Selector pair; a priority- or dependency-aware Scheduler; a
pluggable persistence backend for the Memory/Experience/Adaptation stack;
and a more rigorous treatment of concurrent cycle execution against shared
runtime state. On the governance side, extending Policies beyond
single-expression rules toward composable rule sets with explicit
precedence, and extending the Validator with domain-specific schema
validation (e.g., bounds checking on the numeric fields a credit
decision's Evidence record holds) would both strengthen the architecture's
applicability to production regulated workloads without changing its
underlying decomposition.

## 9. Conclusion

EAR demonstrates that the conflation problem common to agentic
runtimes — blurring governance, planning, execution, validation, and
memory into one opaque call — has a tractable architectural remedy: name
every operation, give it one responsibility, fix its position in an
explicit pipeline, and validate every boundary between stages rather than
trusting it implicitly. The credit-risk underwriting case study shows
this decomposition holding up against a domain — regulated lending — whose
governance, auditability, and dual-control requirements are not abstract
design goals but documented external expectations, with the Validator
checker layer and the Evidence/Memory/Experience/Adaptation split each
demonstrably doing the specific job their roles commit them to.

## References

- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y.
  (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.*
- Board of Governors of the Federal Reserve System and Office of the
  Comptroller of the Currency. (2011). *SR 11-7: Guidance on Model Risk
  Management.*
- Equal Credit Opportunity Act, 15 U.S.C. § 1691 *et seq.*
- EAR source repository: `ear/` package (this work), particularly
  `ear/ksetra.py`, `ear/pariksha.py`, `ear/smriti.py`, `ear/anubhava.py`,
  `ear/samskara.py`, and `examples/credit_risk_guru_ksetra.ipynb`.
