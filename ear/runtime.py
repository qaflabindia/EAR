"""Runtime -- a cycle runs through its full, explicitly-named pipeline
rather than one opaque `reason()` call --

    Governor (govern) -> Initializer (initialize) -> Discoverer (discover)
    -> Selector (select) -> Composer (compose) -> Scheduler (schedule) ->
    Delegator (delegate) -> Orchestrator (orchestrate) -> [Executor
    (execute) -> Performer (perform) -> Deliberator (deliberate) ->
    Decider (decide) -> Validator (validate)] -> Recaller (remember) ->
    Explainer (explain) -> Auditor (audit) -> Memory (store) -> Learner
    (learn) -> Adapter (adapt)

so each operation that AI runtimes often blur together stays a separate,
inspectable, swappable step. Every judgment is made dynamically at
runtime against whichever ModelBinding (e.g. Claude) is active -- Policy
compliance, process Discovery, Selection among candidates, Scheduling
order, step Delegation, the Reasoner's decision, the Explainer's prose --
each with a deterministic fallback so the runtime stays usable offline,
and each judgment written to the ReasoningLog. Only the mechanics with no
judgment content (the Composer's flattening, the Validator's shape
checks, enforcement itself) stay plain Python: the LLM judges, code
enforces and records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .adaptation import Adaptation, AdaptationBank
from .adapter import Adapter
from .auditor import Auditor
from .composer import Composer
from .decider import Decider
from .delegator import Delegator
from .deliberator import Deliberator
from .discoverer import Discoverer
from .evidence import Evidence
from .executor import Executor
from .experience import Experience
from .explainer import Explainer
from .governor import Governor
from .initializer import Initializer
from .intent import Intent
from .learner import Learner
from .memory import Memory
from .model_binding import ModelBinding
from .orchestrator import Orchestrator
from .performer import Performer
from .policy import Policy
from .process import Process
from .reasoner import Reasoner
from .reasoning_log import ReasoningLog, model_name
from .recaller import Recaller
from .scheduler import Scheduler
from .selector import Selector
from .spawner import Spawner
from .validator import Validator
from .workflow import Workflow


@dataclass
class Runtime:
    """A Runtime is the battlefield: every cycle runs through the full
    Governor/Initializer/Discoverer/Selector/Composer/Scheduler/
    Orchestrator pipeline, and is recorded across the Evidence (why) /
    Memory (what) / Experience (pattern) / Adaptation (adaptation) layers."""

    name: str
    processes: list[Process] = field(default_factory=list)
    policies: list[Policy] = field(default_factory=list)
    reasoner: Reasoner = field(default_factory=Reasoner)
    model_binding: Optional[ModelBinding] = None
    memory: Memory = field(default_factory=Memory)
    experience: Experience = field(default_factory=Experience)
    adaptations: AdaptationBank = field(default_factory=AdaptationBank)

    # The operating strategy stacked in memory.md (context history,
    # cross-session data, subagent spawning, model selection, MCP, tools,
    # skills discovery, ontology), the cross-session store it declares, and
    # the Spawner it bounds.
    strategy: Optional[Any] = None
    session_store: Optional[Any] = None
    spawner: Spawner = field(default_factory=Spawner)

    # The audit trail of every reasoning step -- policy judgments with
    # their rationale, discovery, the deliberation with the full stacked
    # prompt material, and the explanation -- reviewable in memory and
    # flushed to JSONL after every cycle when a path is declared.
    reasoning_log: ReasoningLog = field(default_factory=ReasoningLog)

    # Per-cycle pipeline stages.
    governor: Governor = field(default_factory=Governor)
    initializer: Initializer = field(default_factory=Initializer)
    discoverer: Discoverer = field(default_factory=Discoverer)
    selector: Selector = field(default_factory=Selector)
    composer: Composer = field(default_factory=Composer)
    scheduler: Scheduler = field(default_factory=Scheduler)
    delegator: Delegator = field(default_factory=Delegator)
    validator: Validator = field(default_factory=Validator)
    orchestrator: Orchestrator = field(default_factory=Orchestrator)
    recaller: Recaller = field(default_factory=Recaller)
    explainer: Explainer = field(default_factory=Explainer)
    auditor: Auditor = field(default_factory=Auditor)
    learner: Learner = field(default_factory=Learner)
    adapter: Adapter = field(default_factory=Adapter)

    # Standalone, dev-time operations -- not part of the per-cycle pipeline.
    evolver: Any = None
    optimizer: Any = None

    def __post_init__(self) -> None:
        if self.evolver is None:
            from .evolver import Evolver

            self.evolver = Evolver()
        if self.optimizer is None:
            from .optimizer import Optimizer

            self.optimizer = Optimizer()

    def add_process(self, process: Process) -> "Runtime":
        self.processes.append(process)
        return self

    def add_policy(self, policy: Policy) -> "Runtime":
        self.policies.append(policy)
        return self

    def enforce_policies(self, **context: Any) -> list[Policy]:
        """Return the policies that are violated by the given context."""
        return [policy for policy in self.policies if not policy.evaluate(self.model_binding, **context)]

    def reason(self, intent: Intent) -> Any:
        self.reasoning_log.begin_cycle(intent)

        violations = self.governor.govern(self, intent)
        if violations:
            names = ", ".join(policy.name for policy in violations)
            # A blocked cycle is exactly what an auditor wants on record.
            self.reasoning_log.flush()
            raise PermissionError(f"Policy violated: {names}")

        self.initializer.initialize(self)

        candidates = self.validator.validate_candidates(self.discoverer.discover(self, intent))
        selected = self.validator.validate_selection(self.selector.select(self, candidates, intent=intent))
        plan = self.validator.validate_plan(self.composer.compose(selected))
        scheduled = self.validator.validate_schedule(self.scheduler.schedule(plan, runtime=self, intent=intent))

        workflow_violations = self.governor.govern_workflows(self, intent, scheduled)
        if workflow_violations:
            names = ", ".join(policy.name for policy in workflow_violations)
            self.reasoning_log.flush()
            raise PermissionError(f"Workflow policy violated: {names}")

        self.delegator.delegate(self, intent, scheduled)
        recalled = self.recaller.recall(self.memory, intent, runtime=self)

        decision = self.orchestrator.orchestrate(self, intent, plan=scheduled)

        data = self._formalize(intent, scheduled, decision)

        evidence = self._build_evidence(intent, scheduled, recalled)
        if data:
            evidence.sources["data"] = data
        explanation = self.explainer.explain(evidence, decision, model_binding=self.model_binding)
        evidence.sources["explanation"] = explanation
        self.reasoning_log.record(
            stage="explanation",
            inputs={"basis": evidence.basis, "decision": str(decision)},
            output=str(explanation),
            model=model_name(self.model_binding),
        )
        self.auditor.audit(evidence, runtime=self, decision=decision)

        active_lm = self.model_binding.lm if self.model_binding is not None else None
        entry = self.memory.record(
            intent.text, decision, context=intent.context, evidence=evidence, summarizer=active_lm
        )
        self.learner.learn(self.experience, entry)
        learned = self.adapter.adapt(self.adaptations, self.experience, summarizer=active_lm)
        if learned is not None:
            self.reasoning_log.record(
                stage="adaptation",
                inputs={"experience": self.experience.summary()},
                output=learned.insight,
                model=model_name(self.model_binding),
            )
        if self.session_store is not None:
            self.session_store.save(self)
        self.reasoning_log.flush()
        return decision

    def spawn(self, persona: Any, intent: Any) -> Any:
        """Spawn a subagent runtime scoped to one Persona and reason the
        given intent through it, within the spawning limits the strategy in
        memory.md declares."""
        return self.spawner.spawn(self, persona, intent)

    def _formalize(self, intent: Intent, plan: list[Workflow], decision: Any) -> dict[str, Any]:
        """Honor the plan's Contracts: extract each workflow's declared
        deliverable fields from the decision with the bound model, judge
        the extraction against the authored meanings (one hinted retry),
        and return only conformant data. With no model bound there is
        nothing honest to extract, so the skip itself goes on the record
        instead of fabricated values."""
        contracts = [workflow.contract for workflow in plan if workflow.contract is not None]
        if not contracts:
            return {}
        data: dict[str, Any] = {}
        model_active = self.model_binding is not None and self.model_binding.lm is not None
        for contract in contracts:
            if not model_active:
                self.reasoning_log.record(
                    stage="contract",
                    inputs={"contract": contract.name, "fields": contract.render_fields()},
                    output="skipped -- no model bound to extract the deliverable",
                    model=model_name(self.model_binding),
                )
                continue
            extracted = contract.extract(decision, intent, self.model_binding)
            conforms, rationale = contract.judge(extracted, self.model_binding)
            if not conforms:
                extracted = contract.extract(decision, intent, self.model_binding, hint=rationale)
                conforms, rationale = contract.judge(extracted, self.model_binding)
            self.reasoning_log.record(
                stage="contract",
                inputs={"contract": contract.name, "fields": contract.render_fields(), "data": extracted},
                output="conformant" if conforms else "NONCONFORMING -- data withheld from the decision",
                rationale=rationale,
                model=model_name(self.model_binding),
            )
            if conforms:
                data.update(extracted)
        return data

    def _build_evidence(self, intent: Intent, plan: list[Workflow], recalled: str) -> Evidence:
        """Capture why this decision was reached -- separately from what
        was decided (Memory) or any pattern drawn from repeating it
        (Experience)."""
        if self.reasoner.program is not None:
            basis = "Resolved via a compiled DSPy program"
        elif self.model_binding is not None and self.model_binding.lm is not None:
            basis = f"Resolved via ModelBinding LM '{self.model_binding.model_id}'"
        else:
            basis = "Resolved via the Reasoner's dependency-free default"
        return Evidence(
            basis=basis,
            sources={
                "policies_checked": [policy.name for policy in self.policies],
                "context": dict(intent.context),
                "plan": [workflow.name for workflow in plan],
                "recalled_memory": recalled,
            },
        )
