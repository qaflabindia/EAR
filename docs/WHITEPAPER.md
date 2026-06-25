# EAR: An Enterprise Agentic Runtime with Decomposed Governance, Reasoning, and Memory

*A technical whitepaper on the EAR (`ear`) Python package*

## Abstract

Production agentic systems routinely collapse several distinct concerns —
policy enforcement, candidate discovery, plan composition, scheduling,
deliberation, decision validation, evidentiary justification, persistent
memory, and learned adaptation — into a single opaque `reason()` or
`agent.run()` call. This conflation makes such systems difficult to audit,
difficult to govern, and difficult to reason about formally, precisely in
domains — financial services, healthcare, public-sector benefits
determination — where *why* a decision was reached must be reconstructable
independently of *what* was decided. This paper presents **EAR (Enterprise
Agentic Runtime)**, a Python reference implementation that decomposes an
agentic decision cycle into nineteen named, independently inspectable,
independently swappable stages, organized around an eight-layer
"philosophical stack" (`Sankalpa → Vidya → Guna → Varna → Karma → Dharma →
Ksetra → Bhuddi`) and a four-layer memory model (`Pramana → Smriti →
Anubhava → Samskara`) that keeps evidence, history, aggregated experience,
and adaptation as separate artifacts rather than one blended state. We
describe the architecture, formalize the per-cycle pipeline, detail the
`Pariksha` maker–checker validation layer that gates every pipeline stage's
output, and present a worked case study — `Credit-Risk-Guru-Ksetra`, a
personal-loan underwriting runtime — that exercises governance gates,
checker-layer interception, and multi-cycle memory consolidation against
a regulated-domain decision problem. We close with a discussion of the
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
- A **maker–checker validation layer**, `Pariksha` (Section 5), that
  independently validates the output of every stage that produces
  structured data for a downstream stage to consume, rather than letting
  each stage re-implement its own ad hoc validation or, worse, omit it.
- A **four-layer memory model** (Section 6) — evidence (`Pramana`),
  persistent history (`Smriti`), aggregated experience (`Anubhava`), and
  distilled adaptation (`Samskara`) — that keeps "why", "what", "the
  pattern", and "the lesson" as four separate, independently consumable
  artifacts.
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
own right: `Smriti` records, `Anubhava` aggregates, `Samskara` distills,
and `Pramana` separately preserves the justification for each individual
decision, independent of all three.

In regulated domains, the closest analogue to EAR's `Pariksha` layer is the
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

## 3. The Philosophical Stack

EAR maps an eight-stage engineering stack onto eight Sanskrit-derived
class names, chosen so that the *name* of a class signals its *role*
independent of its implementation:

| Engineering role | Class | Sanskrit | Responsibility |
|---|---|---|---|
| Prompt / intent | `Sankalpa` | संकल्प | The resolved intent that starts a cycle: free text plus a structured context dictionary. |
| Skill / capability | `Vidya` | विद्या | One addressable capability with an optional handler — the unit prompts are "stacked inside." |
| Persona / behavioural nature | `Guna` | गुण | A stack of `Vidya` skills plus standing instructions, giving the skills a coherent voice. |
| Workflow / role-ordering | `Varna` | वर्ण | An ordered stack of `Guna` personas — *who* acts, in what sequence. |
| Process / action | `Karma` | कर्म | A stack of `Varna` workflows the runtime can discover and run. |
| Policy / governance | `Dharma` | धर्म | A safely-evaluated rule (no `eval`/`exec`) that gates a `Karma`'s execution. |
| Runtime / field of execution | `Ksetra` | क्षेत्र | The battlefield: holds processes, policies, and every pipeline stage; runs the full cycle. |
| Reasoning / discriminative intelligence | `Bhuddi` | बुद्धि | Resolves a `Sankalpa`, against a compiled DSPy program, an activated LLM, or a dependency-free deterministic fallback. |

`Manas` (मनस्, "the mind") is deliberately *not* a ninth stack layer. It is
the LLM provider binding — model identifier, credentials, call
parameters — and is better understood as the current that runs through
the stack: `Ksetra` activates a runtime's `Manas` immediately before
handing a `Sankalpa` to `Bhuddi`, so that whichever reasoning path `Bhuddi`
takes runs against a properly configured language model rather than an
unconfigured one. A `Ksetra` with no `Manas` attached is not a degraded
configuration; it is a deliberately supported mode in which `Bhuddi` falls
back to a deterministic, dependency-free summary — meaning the entire
governance, validation, and audit apparatus described in this paper is
exercised whether or not a language model is present.

## 4. The Ksetra Pipeline

A single call to `Ksetra.reason(sankalpa)` advances one cycle through
nineteen named stages, grouped here by function rather than by call order
for exposition, with their call order given in Section 4.2.

### 4.1 Stage catalogue

| Group | Sanskrit | Transliteration | Operation | Class |
|---|---|---|---|---|
| Governance | नियमन | Niyamana | Govern — enforce `Dharma` policy gates | `Niyamana` |
| Initialization | आरम्भ | Arambha | Initialize — activate `Manas` | `Arambha` |
| Planning | अन्वेषण | Anveshana | Discover — find relevant `Karma` candidates | `Anveshana` |
| Planning | वरण | Varana | Select — choose among discovered candidates | `Varana` |
| Planning | संयोजन | Samyojana | Compose — assemble selected `Varna` workflows into a plan | `Samyojana` |
| Planning | नियोजन | Niyojana | Schedule — order the composed plan | `Niyojana` |
| Coordination | समन्वय | Samanvaya | Orchestrate — coordinate a cycle's execution end to end | `Samanvaya` |
| Coordination | अनुष्ठान | Anushthana | Execute — run the cycle's `Kriya` action | `Anushthana` |
| Coordination | क्रिया | Kriya | Perform — chain deliberate → decide → validate | `Kriya` |
| Core decision | विचार | Vicara | Reason — deliberate via `Bhuddi` | `Vicara` |
| Core decision | निर्णय | Nirnaya | Decide — commit to one decision | `Nirnaya` |
| Validation | परीक्षा | Pariksha | Validate — checker layer for every maker stage's output | `Pariksha` |
| Memory recall | स्मरण | Smarana | Remember — recall `Smriti` context as evidence | `Smarana` |
| Explanation | व्याख्या | Vyakhya | Explain — render why a decision was reached | `Vyakhya` |
| Compliance | परिशोधन | Parishodhana | Audit — inspect evidence for compliance | `Parishodhana` |
| Memory write | स्मृति | Smriti | Store memory — record what happened | `Smriti` |
| Learning | अध्ययन | Adhyayana | Learn — fold a cycle into `Anubhava` experience | `Adhyayana` |
| Adaptation | अनुकूलन | Anukulana | Adapt — periodically distill a new `Samskara` | `Anukulana` |
| *(dev-time, out of cycle)* | परिणाम | Parinama | Evolve — transform a `Vidya` skill's source (openevolve) | `Parinama` |
| *(dev-time, out of cycle)* | उत्कर्ष | Utkarsha | Optimize — refine a `Guna` skill document (SkillOpt) | `Utkarsha` |

Each row is a separate `dataclass` with one public method. None of them
import `Ksetra`; `Ksetra` imports all of them. This is a deliberate
acyclic-dependency constraint: every stage is independently testable in
isolation, with the runtime as the only thing that knows how they compose.

### 4.2 Control flow

```text
Niyamana → Arambha → Anveshana → Pariksha → Varana → Pariksha → Samyojana
  → Pariksha → Niyojana → Pariksha → Smarana
  → Samanvaya [→ Anushthana → Kriya [→ Vicara → Nirnaya → Pariksha]]
  → Vyakhya → Parishodhana → Smriti → Adhyayana → Anukulana
```

Two properties of this control flow are load-bearing rather than
cosmetic:

**Governance is a hard gate, not a soft step.** `Niyamana.govern` runs
*before* `Arambha` even activates the runtime's `Manas`. If any `Dharma`
policy is violated, `Ksetra.reason` raises `PermissionError` immediately;
no candidate is discovered, no plan is composed, no `Smriti` entry is
written. A policy violation cannot be reasoned around because reasoning
never starts.

**Adaptation is throttled, not continuous.** `Anukulana.adapt` only
invokes `SamskaraBank.learn_from` every `adapt_every` *observed* cycles
(default five), guarded by `anubhava.observations % adapt_every == 0`.
This is a explicit design choice against the failure mode of producing one
fresh, noisy "lesson" per cycle; `Samskara` entries are meant to be
durable impressions distilled from a body of experience, not a running
commentary on the most recent decision.

## 5. Pariksha: A Maker–Checker Layer for Pipeline Stages

Four stages in the planning group — `Anveshana`, `Varana`, `Samyojana`,
`Niyojana` — each produce a typed list that the next stage consumes
without re-validating: `Anveshana` and `Varana` each produce
`list[Karma]`; `Samyojana` and `Niyojana` each produce `list[Varna]`.
`Nirnaya` produces an arbitrary decision value. Letting each of these five
"maker" stages own its own validation invites two failure modes: validation
logic duplicated five times with five chances to diverge, or validation
omitted entirely on the stages whose authors judged it unnecessary at the
time.

`Pariksha` (`ear/pariksha.py`) is instead the single checker every maker
stage's output passes through:

```python
@dataclass
class Pariksha:
    def validate_candidates(self, candidates: list[Karma]) -> list[Karma]: ...
    def validate_selection(self, selected: list[Karma]) -> list[Karma]: ...
    def validate_plan(self, plan: list[Varna]) -> list[Varna]: ...
    def validate_schedule(self, scheduled: list[Varna]) -> list[Varna]: ...
    def validate(self, decision: Any) -> Any: ...
```

Each `validate_*` method shares one private routine,
`_validate_list(items, item_type, label)`, which raises `TypeError` if
`items` is not a `list`, or if it contains any element that is not an
instance of `item_type` — naming the offending stage (`"Anveshana
candidates"`, `"Niyojana schedule"`, etc.) in the error so a failure is
immediately attributable to the stage that produced it. `validate` (the
original, decision-facing method, retained for `Kriya`'s internal
deliberate → decide → validate chain) instead rejects only a blank string,
since a structurally arbitrary `Any` decision cannot be type-checked the
way a list of `Karma`/`Varna` can.

Two design decisions here are deliberate rather than incidental:

- **Empty lists are valid.** A runtime with zero registered `Karma`
  processes is an explicitly supported configuration (`Anveshana.discover`
  returns the empty list it was given rather than raising), so `Pariksha`
  validates *type*, not *non-emptiness*.
- **`Ksetra` and `Kriya` each hold their own `Pariksha` instance.**
  `Ksetra.pariksha` checks the four planning-stage outputs;
  `Kriya.pariksha` (nested inside `Anushthana`, inside `Samanvaya`) checks
  the final decision. Since `Pariksha` is stateless — every method is a
  pure function of its arguments — this duplication carries no behavioural
  risk, but it is worth surfacing explicitly: a future refactor collapsing
  both onto one shared instance would be safe, not merely convenient.

This is the architecture's maker–checker control made literal: every
producer of structured data for a downstream consumer has a corresponding,
independently invoked checker, and that checker's rejection — a
`TypeError` naming the offending stage — happens at the boundary where the
malformed data was introduced, not several stages later where its effects
would otherwise be much harder to trace back to a root cause.

## 6. Memory: Pramana, Smriti, Anubhava, Samskara

EAR treats four memory-adjacent concerns as four separate artifacts rather
than one blended state:

| Layer | Sanskrit | Question answered | Class |
|---|---|---|---|
| Evidence | प्रमाण Pramana | *Why* was this particular decision made? | `Pramana` |
| Memory | स्मृति Smriti | *What* happened? | `Smriti` / `SmritiEntry` |
| Experience | अनुभव Anubhava | What *pattern* holds across repeated `Smriti` entries? | `Anubhava` |
| Adaptation | संस्कार Samskara | How should future behaviour *change* as a result? | `Samskara` / `SamskaraBank` |

Each `Ksetra.reason()` call builds one `Pramana` recording which reasoning
path resolved the decision (a compiled DSPy program, an activated `Manas`
LLM, or `Bhuddi`'s dependency-free default), which `Dharma` policies were
checked, the `Sankalpa`'s input context, the composed `Samyojana` plan, and
what `Smarana` recalled from memory at decision time. `Vyakhya.explain`
then renders a human-readable explanation directly from that `Pramana`,
and `Parishodhana.audit` marks the `Pramana` as inspected — all *before*
`Smriti.record` writes the decision and its evidence to persistent memory
as one `SmritiEntry`. Because the `Pramana` is attached to the entry as a
distinct field (`SmritiEntry.evidence`) rather than interpolated into the
decision text, "why" remains queryable independently of "what" indefinitely
— including after `Smriti`'s own compression step (Section 6.1) has long
since discarded the verbatim entry.

### 6.1 Bounded memory via two-layer compression

`Smriti` keeps a `working` layer of recent entries verbatim, bounded by a
`capacity` (default twenty). Once `working` exceeds `capacity`, the oldest
overflow entries are rolled into one new summary string appended to a
`compressed` layer — by default a deterministic digest of the overflowing
decisions, or, if an activated `Manas` LM is supplied as `summarizer`, an
LLM-written summary. `Smriti.context_window()` renders the concatenation
of `compressed` history and recent `working` entries as the string
`Bhuddi`'s default reasoning path folds into its prompt. This keeps the
context a reasoning call sees bounded as history grows, without ever
discarding the existence of older cycles outright.

### 6.2 From memory to experience to adaptation

`Adhyayana.learn` folds each newly written `SmritiEntry` into the
runtime's `Anubhava`, which maintains a running count of decisions seen
and the full list of `Pramana` evidence observed along the way —
aggregation without yet drawing a conclusion. `Anukulana.adapt` is the
step that *does* draw a conclusion, but only every `adapt_every` observed
cycles: it calls `SamskaraBank.learn_from(anubhava, summarizer=...)`, which
either reports the single most frequent decision in the aggregated
experience (the deterministic default) or asks an LLM to state one durable
lesson in a sentence (when a summarizer is supplied), and appends the
result as a new `Samskara` impression. On the *next* cycle, `Bhuddi`'s
default reasoning path retrieves `SamskaraBank.relevant_to(sankalpa.text)`
— a keyword-overlap lookup over the impression text — and folds any
matching impressions into its prompt as a third, clearly labelled section
alongside `Smriti`'s memory and `Anubhava`'s experience summary. Memory,
experience, and adaptation thus each influence subsequent reasoning as
three distinct, separately inspectable inputs, rather than one blurred
"context" blob.

## 7. Case Study: Credit-Risk-Guru-Ksetra

To validate the architecture against a realistic, regulation-adjacent
decision problem rather than a synthetic toy, we instantiated a complete
`Ksetra` — `Credit-Risk-Guru-Ksetra` — for personal-loan underwriting and
executed it end to end (`examples/credit_risk_guru_ksetra.ipynb`).

### 7.1 Construction

Three `Vidya` skills perform feature derivation: `credit_score_tier` bands
a FICO score into `prime`/`near-prime`/`subprime`/`deep-subprime`,
`dti_tier` bands a debt-to-income ratio into `low`/`moderate`/`high`, and
`risk_grade` combines both bands into a letter grade via a fixed lookup
table. These are stacked into a `Guna` persona, "Credit Risk Guru," with
standing instructions to underwrite conservatively and band every
applicant before any approval reasoning happens. The persona is stacked
into a single-persona `Varna` ("Underwriting Workflow"), which is in turn
stacked into a `Karma` process ("Personal Loan Underwriting"). Four
`Dharma` policies are registered directly on the runtime:

| Policy | Rule |
|---|---|
| Minimum Credit Score | `credit_score >= 620` |
| Debt-to-Income Ceiling | `debt_to_income_ratio <= 0.45` |
| No Active Defaults | `existing_defaults == 0` |
| Loan Amount Cap | `loan_amount <= 75000` |

Each `Dharma.rule` is evaluated by `ear/_safe_eval.py`, a restricted
AST-walking evaluator that permits only literals, names, comparisons,
boolean logic, and arithmetic — explicitly forbidding `eval`/`exec` so a
policy string supplied by whoever configures the runtime can never execute
arbitrary code.

### 7.2 Observed behaviour

The notebook exercises five distinct properties of the architecture
against this runtime, entirely on `Bhuddi`'s dependency-free default
reasoning path (no LLM credentials required):

1. **A clean approval.** A prime-tier, low-DTI applicant clears all four
   `Dharma` gates; the resulting `SmritiEntry`'s `Pramana` records the
   composed plan (`["Underwriting Workflow"]`), all four policies checked,
   and a rendered explanation.
2. **A governance rejection.** An applicant with a 0.52 debt-to-income
   ratio breaches the 0.45 ceiling. `Niyamana.govern` catches this before
   `Arambha` runs, and `Ksetra.reason` raises `PermissionError` naming the
   violated policy — no decision is reasoned, no memory entry is written.
3. **A live `Pariksha` interception.** `Anveshana.discover` is temporarily
   monkey-patched to return a `Varna` instance where the pipeline expects
   `Karma` instances. `Ksetra.pariksha.validate_candidates` raises
   `TypeError: Anveshana candidates must contain only Karma instances`
   before the malformed output ever reaches `Varana.select` — demonstrating
   that the checker layer documented in Section 5 is load-bearing, not
   merely declared.
4. **Multi-cycle memory consolidation.** A small six-applicant portfolio
   (one borderline applicant plus four further synthetic applicants) is
   run through the same runtime. By the fifth successful cycle,
   `Anukulana`'s `adapt_every=5` threshold fires and `SamskaraBank`
   distills one new `Samskara` impression from the accumulated `Anubhava`
   experience — observable directly via `runtime.samskara.impressions`.
5. **A complete audit trail.** The final entry's `Pramana.sources` is
   rendered as a single JSON object containing the policies checked, the
   full applicant context (including derived `score_tier`, `dti_band`, and
   `risk_grade`), the composed plan, the recalled memory window, the
   rendered explanation, and the `Parishodhana` audit flag — the artifact
   a model-governance review or an adverse-action-notice process would
   need.

### 7.3 Regulatory correspondence

This case study was chosen because personal-loan underwriting is a domain
in which the architectural properties above are not merely good
engineering practice but documented compliance expectations:

- The `Dharma`/`Niyamana` gate corresponds to the hard underwriting and
  fair-lending floors a lender's policy already imposes on human
  underwriters — and, as in Section 7.2(2), a breach is structurally
  incapable of being reasoned around, since reasoning has not yet started
  when the gate is checked.
- The `Pariksha` checker layer corresponds to the maker–checker / four-eyes
  control already required of manual underwriting workflows, applied
  instead to every automated pipeline stage.
- The `Pramana` evidence trail corresponds to the independent
  explainability and auditability interagency model-risk-management
  guidance (e.g., SR 11-7) expects of any model used in a credit decision,
  and to the specific, reconstructable reason fair-lending statutes such as
  the Equal Credit Opportunity Act require behind an adverse credit
  action.
- The `Smriti → Anubhava → Samskara` chain corresponds to portfolio-level
  monitoring for unmonitored model drift — precisely the failure mode
  model-risk-management governance exists to catch, and precisely what
  collapsing memory, pattern, and adaptation into one undifferentiated
  state would make invisible.

## 8. Discussion

### 8.1 Strengths

The central claim this architecture supports is **attributable
failure**: when something goes wrong, the named stage responsible is
visible in the stack trace, the exception message, or the `Pramana`
record, rather than buried inside one large `reason()` call. Section 7.2's
`Pariksha` demonstration is the clearest instance of this: the failure is
not merely caught, it is caught *and attributed to the exact maker stage
that produced it*. Equally, every stage being its own dataclass with one
public method makes substitution straightforward — swapping `Anveshana`'s
keyword-overlap discovery for an embedding-based retriever, or `Niyojana`'s
identity ordering for a priority/dependency scheduler, requires touching
only that one class, since every other stage interacts with it only
through its public method signature.

### 8.2 Limitations

Several aspects of the current implementation are intentionally minimal
and should not be mistaken for completeness:

- **Discovery and selection are keyword-overlap heuristics.**
  `Anveshana.discover` and the absence of any ranking in `Varana.select`
  beyond deduplication are explicitly documented as a baseline to be
  replaced with embeddings or a learned retriever for production use.
- **Scheduling has no ordering signal to act on.** `Niyojana.schedule`
  returns a defensive copy in discovery order because `Varna` carries no
  priority, dependency, or cost field today; the seam exists, but nothing
  yet populates it.
- **No persistence layer.** `Smriti`, `Anubhava`, and `SamskaraBank` are
  in-memory dataclasses; a deployment that needs cycles to survive process
  restarts, or needs `Smriti` shared across concurrent runtime instances,
  must add that layer itself.
- **No concurrency model.** Nothing in the pipeline is documented as
  thread-safe, and `Ksetra.reason` is not designed for concurrent
  invocation against shared `Smriti`/`Anubhava`/`Samskara` state.
- **The dependency-free reasoning fallback is intentionally simple.**
  `Bhuddi`'s default path is a deterministic summary string, sufficient to
  exercise every other stage of the pipeline without an LLM, but it is not
  itself a credit-risk decision engine — production use requires either a
  compiled DSPy program or an activated `Manas` pointed at a real model.
- **Two independent `Pariksha` instances exist by construction**
  (`Ksetra.pariksha` and `Kriya.pariksha`), which is harmless only because
  `Pariksha` is stateless; introducing any per-instance state to `Pariksha`
  in the future would need to either justify the duplication or collapse
  it to one shared instance.

### 8.3 Future work

Natural extensions include: a learned (rather than keyword-overlap)
`Anveshana`/`Varana` pair; a priority- or dependency-aware `Niyojana`;
a pluggable persistence backend for the `Smriti`/`Anubhava`/`Samskara`
stack; and a more rigorous treatment of concurrent cycle execution against
shared runtime state. On the governance side, extending `Dharma` beyond
single-expression rules toward composable rule sets with explicit
precedence, and extending `Pariksha` with domain-specific schema validation
(e.g., bounds checking on the numeric fields a credit decision's `Pramana`
records) would both strengthen the architecture's applicability to
production regulated workloads without changing its underlying
decomposition.

## 9. Conclusion

EAR demonstrates that the conflation problem common to agentic
runtimes — blurring governance, planning, execution, validation, and
memory into one opaque call — has a tractable architectural remedy: name
every operation, give it one responsibility, fix its position in an
explicit pipeline, and validate every boundary between stages rather than
trusting it implicitly. The `Credit-Risk-Guru-Ksetra` case study shows
this decomposition holding up against a domain — regulated lending — whose
governance, auditability, and dual-control requirements are not abstract
design goals but documented external expectations, with the
`Pariksha` checker layer and the `Pramana`/`Smriti`/`Anubhava`/`Samskara`
memory split each demonstrably doing the specific job their names commit
them to.

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
