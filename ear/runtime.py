"""Runtime -- a cycle runs through its full, explicitly-named pipeline
rather than one opaque `reason()` call --

    Governor (govern) -> Initializer (initialize) -> Discoverer (discover)
    -> Selector (select) -> Composer (compose) -> Scheduler (schedule) ->
    Orchestrator (orchestrate) -> [Executor (execute) -> Performer
    (perform) -> Deliberator (deliberate) -> Decider (decide) -> Validator
    (validate)] -> Recaller (remember) -> Explainer (explain) -> Auditor
    (audit) -> Memory (store) -> Learner (learn) -> Adapter (adapt)

so each operation that AI runtimes often blur together stays a separate,
inspectable, swappable step. Judgment-laden steps (Policy compliance,
process Discovery, the Reasoner's decision, the Explainer's prose) reason
in natural language against whichever ModelBinding (e.g. Claude) is
active; structural steps (Selector, Composer, Scheduler) have no judgment
call to make, so they stay plain Python."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .adaptation import Adaptation, AdaptationBank
from .adapter import Adapter
from .auditor import Auditor
from .composer import Composer
from .decider import Decider
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
from .recaller import Recaller
from .scheduler import Scheduler
from .selector import Selector
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

    # Per-cycle pipeline stages.
    governor: Governor = field(default_factory=Governor)
    initializer: Initializer = field(default_factory=Initializer)
    discoverer: Discoverer = field(default_factory=Discoverer)
    selector: Selector = field(default_factory=Selector)
    composer: Composer = field(default_factory=Composer)
    scheduler: Scheduler = field(default_factory=Scheduler)
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
        violations = self.governor.govern(self, intent)
        if violations:
            names = ", ".join(policy.name for policy in violations)
            raise PermissionError(f"Policy violated: {names}")

        self.initializer.initialize(self)

        candidates = self.validator.validate_candidates(self.discoverer.discover(self, intent))
        selected = self.validator.validate_selection(self.selector.select(self, candidates))
        plan = self.validator.validate_plan(self.composer.compose(selected))
        scheduled = self.validator.validate_schedule(self.scheduler.schedule(plan))

        workflow_violations = self.governor.govern_workflows(self, intent, scheduled)
        if workflow_violations:
            names = ", ".join(policy.name for policy in workflow_violations)
            raise PermissionError(f"Workflow policy violated: {names}")

        recalled = self.recaller.recall(self.memory, intent)

        decision = self.orchestrator.orchestrate(self, intent, plan=scheduled)

        evidence = self._build_evidence(intent, scheduled, recalled)
        evidence.sources["explanation"] = self.explainer.explain(evidence, decision, model_binding=self.model_binding)
        self.auditor.audit(evidence)

        entry = self.memory.record(intent.text, decision, context=intent.context, evidence=evidence)
        self.learner.learn(self.experience, entry)
        self.adapter.adapt(self.adaptations, self.experience)
        return decision

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
