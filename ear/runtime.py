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

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .adaptation import AdaptationBank
from .adapter import Adapter
from .approval import ApprovalRequired
from .auditor import Auditor
from .composer import Composer
from .delegator import Delegator
from .discoverer import Discoverer
from .evidence import Evidence
from .experience import Experience
from .explainer import Explainer
from .goal import GoalKeeper, GoalOutcome
from .governor import Governor
from .initializer import Initializer
from .intent import Intent
from .learner import Learner
from .librarian import Librarian
from .memory import Memory
from .model_binding import ModelBinding
from .orchestrator import Orchestrator
from .panel import Panel
from .policy import Policy
from .process import Process
from .reasoner import Reasoner
from .reasoning_log import ReasoningLog, apply_retention, calls_so_far, model_name, usage_since
from .recaller import Recaller
from .scheduler import Scheduler
from .selector import Selector
from .spawner import Spawner
from .tool_binder import ToolBinder
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
    # Policies scoped to tool invocations (`Applies to: tools` in
    # policy.md): judged against a tool's name and arguments before the
    # call runs, so a violation blocks that one call rather than the cycle.
    tool_policies: list[Policy] = field(default_factory=list)
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
    # The named catalogue (Stores: SkillStore, PersonaStore, ...) this
    # runtime's stack was loaded from -- file-based under `store/` by
    # default (the fallback), or a PostgresAgeBackend if memory.md's
    # Catalogue Store section opted in. Loader wires this on every load;
    # a Runtime built by hand and never loaded from a directory leaves it
    # None.
    stores: Optional[Any] = None
    spawner: Spawner = field(default_factory=Spawner)
    tool_binder: ToolBinder = field(default_factory=ToolBinder)
    panel: Panel = field(default_factory=Panel)
    goal_keeper: GoalKeeper = field(default_factory=GoalKeeper)
    # The instance's isolated workspace and governed command runner, when a
    # Sandbox section is declared in memory.md (see ear/sandbox.py). None
    # means the runtime shares the host filesystem, as before.
    sandbox: Optional[Any] = None

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
    librarian: Librarian = field(default_factory=Librarian)
    explainer: Explainer = field(default_factory=Explainer)
    auditor: Auditor = field(default_factory=Auditor)
    learner: Learner = field(default_factory=Learner)
    adapter: Adapter = field(default_factory=Adapter)

    # Standalone, dev-time operation -- not part of the per-cycle pipeline.
    optimizer: Any = None

    # Live MCP connections opened by connect_mcp, closed by disconnect_mcp.
    _mcp_clients: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.optimizer is None:
            from .optimizer import Optimizer

            self.optimizer = Optimizer()

    def connect_mcp(self, name: Optional[str] = None, client: Any = None) -> list:
        """Connect a declared MCP server and bind its tools into the cycle's
        toolset. The server stays declared in memory.md; this reaches out to
        what the author named. `name` selects one declared server (all of
        them when omitted); pass `client` to supply an already-connected
        McpClient (a test double, say) instead of launching the command.
        Returns the BoundTools now available. Every MCP call then rides the
        same logged handler, budgets and tool-scoped policies as any tool."""
        from .mcp_client import McpClient, command_words
        from .tool_binder import BoundTool

        servers = list(getattr(self.strategy, "mcp_servers", None) or [])
        if name is not None:
            servers = [server for server in servers if server.name.lower() == name.lower()]
            if not servers:
                raise ValueError(f"No MCP server named '{name}' is declared in memory.md")

        bound: list = []
        for server in servers:
            if client is not None:
                connection = client
            elif server.command:
                connection = McpClient(command_words(server.command)).connect()
            else:
                raise ValueError(
                    f"MCP server '{server.name}' declares no launch command -- write it backticked in memory.md"
                )
            self._mcp_clients.append(connection)
            for tool in connection.list_tools():
                handler = _mcp_handler(connection, tool.name)
                bound_tool = BoundTool(
                    name=f"{server.name}: {tool.name}",
                    description=tool.description,
                    handler=handler,
                    parameter_names=tool.parameters,
                )
                self.tool_binder.mcp_tools.append(bound_tool)
                bound.append(bound_tool)
        return bound

    def disconnect_mcp(self) -> None:
        """Shut down every connected MCP server and drop its bound tools."""
        for connection in self._mcp_clients:
            close = getattr(connection, "close", None)
            if callable(close):
                close()
        self._mcp_clients = []
        self.tool_binder.mcp_tools = []

    def write_usage_report(self, path: Any) -> str:
        """Render the operational ledger from the trail -- cycles, model
        calls, tokens, dollars (when Pricing is declared), latency and tool
        calls -- and write it to `path`. Returns the markdown."""
        from pathlib import Path

        report = self.reasoning_log.usage_report(strategy=self.strategy)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report, encoding="utf-8")
        return report

    def pursue(self, goal: Any, intent: Any) -> "GoalOutcome":
        """Pursue a session goal: run the intent, then keep driving the
        runtime forward until the goal is met, genuinely blocked, or the
        continuation budget is spent -- the model judging satisfaction and
        the typed blocker, code enforcing the caps, all on the trail (see
        ear/goal.py)."""
        return self.goal_keeper.pursue(self, goal, intent)

    def add_process(self, process: Process) -> "Runtime":
        self.processes.append(process)
        return self

    def add_policy(self, policy: Policy) -> "Runtime":
        self.policies.append(policy)
        return self

    def enforce_policies(self, **context: Any) -> list[Policy]:
        """Return the policies that are violated by the given context."""
        return [policy for policy in self.policies if not policy.evaluate(self.model_binding, **context)]

    def reason(self, intent: Intent, approval: Any = None) -> Any:
        started = time.monotonic()
        calls_before = self._model_calls_so_far()
        self.reasoning_log.begin_cycle(intent)

        violations = self.governor.govern(self, intent, approval=approval)
        self._enforce(violations, approval, started, calls_before, scope="Policy")

        self.initializer.initialize(self)

        candidates = self.validator.validate_candidates(self.discoverer.discover(self, intent))
        selected = self.validator.validate_selection(self.selector.select(self, candidates, intent=intent))
        plan = self.validator.validate_plan(self.composer.compose(selected))
        scheduled = self.validator.validate_schedule(self.scheduler.schedule(plan, runtime=self, intent=intent))

        workflow_violations = self.governor.govern_workflows(self, intent, scheduled, approval=approval)
        self._enforce(workflow_violations, approval, started, calls_before, scope="Workflow policy")

        self.delegator.delegate(self, intent, scheduled)
        recalled = self.recaller.recall(self.memory, intent, runtime=self)
        research = self.librarian.research(self, intent)

        decision = self.orchestrator.orchestrate(self, intent, plan=scheduled, research=research)

        data = self._formalize(intent, scheduled, decision)

        evidence = self._build_evidence(intent, scheduled, recalled)
        if data:
            evidence.sources["data"] = data
        if research is not None and research.citations:
            evidence.sources["citations"] = list(research.citations)
        active_lm = self.model_binding.lm if self.model_binding is not None else None
        explanation_start = calls_so_far(active_lm)
        explanation = self.explainer.explain(evidence, decision, model_binding=self.model_binding)
        evidence.sources["explanation"] = explanation
        self.reasoning_log.record(
            stage="explanation",
            inputs={"basis": evidence.basis, "decision": str(decision)},
            output=str(explanation),
            model=model_name(self.model_binding),
            usage=usage_since(active_lm, explanation_start),
        )
        self.auditor.audit(evidence, runtime=self, decision=decision)

        entry = self.memory.record(
            intent.text, decision, context=intent.context, evidence=evidence, summarizer=active_lm
        )
        self.learner.learn(self.experience, entry)
        adaptation_start = calls_so_far(active_lm)
        learned = self.adapter.adapt(self.adaptations, self.experience, summarizer=active_lm)
        if learned is not None:
            self.reasoning_log.record(
                stage="adaptation",
                inputs={"experience": self.experience.summary()},
                output=learned.insight,
                model=model_name(self.model_binding),
                usage=usage_since(active_lm, adaptation_start),
            )
        if self.session_store is not None:
            self.session_store.save(self)
        self._record_usage(started, calls_before)
        apply_retention(self)
        self.reasoning_log.flush()
        return decision

    def _enforce(self, violations: list[Policy], approval: Any, started: float, calls_before: int, scope: str) -> None:
        """Turn the Governor's unresolved violations into a stop: a hard
        block when any non-gated policy is violated (or a human rejected
        the gate), a parked `ApprovalRequired` when only approval-gated
        policies remain. Both stops close the cycle's accounting and flush
        the trail -- a refusal and a parked cycle are records, not gaps."""
        if not violations:
            return
        rejected = approval is not None and approval.verdict is False
        blocking = [policy for policy in violations if not policy.approval_required or rejected]
        pending = [policy for policy in violations if policy.approval_required and not rejected]
        if blocking:
            names = ", ".join(policy.name for policy in blocking)
            self._record_usage(started, calls_before)
            self.reasoning_log.flush()
            raise PermissionError(f"{scope} violated: {names}")
        names = ", ".join(policy.name for policy in pending)
        self.reasoning_log.record(
            stage="approval",
            inputs={"policies": [policy.name for policy in pending]},
            output=f"PENDING -- human approval required for: {names}",
        )
        self._record_usage(started, calls_before)
        self.reasoning_log.flush()
        raise ApprovalRequired(pending)

    def _model_calls_so_far(self) -> int:
        """How many calls the bound LM's history holds before this cycle,
        so the cycle's usage is the delta. A binding not yet activated has
        no history -- its count starts at zero, which is exactly right."""
        lm = self.model_binding.lm if self.model_binding is not None else None
        history = getattr(lm, "history", None)
        return len(history) if history is not None else 0

    def _record_usage(self, started: float, calls_before: int) -> None:
        """Close the cycle's accounting: wall-clock latency always, and --
        when a model is bound -- the model calls, tokens and cost this
        cycle consumed, read from the LM's own call history. Written on
        blocked cycles too: a refusal costs whatever it cost."""
        latency_ms = int((time.monotonic() - started) * 1000)
        lm = self.model_binding.lm if self.model_binding is not None else None
        history = getattr(lm, "history", None) or []
        cycle_calls = history[calls_before:]
        input_tokens = output_tokens = retries = 0
        for call in cycle_calls:
            usage = call.get("usage") or {} if isinstance(call, dict) else {}
            input_tokens += int(usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("completion_tokens") or 0)
            retries += int(call.get("retries") or 0) if isinstance(call, dict) else 0
        # Dollars only when the author declared Pricing in memory.md -- a
        # figure nobody declared is never invented.
        cost = self.strategy.dollars(input_tokens, output_tokens) if self.strategy is not None else None
        if cycle_calls:
            summary = f"{len(cycle_calls)} model calls, {input_tokens}+{output_tokens} tokens"
            if cost is not None:
                summary += f", ~${cost:.6f}"
            if retries:
                summary += f", {retries} retried"
            summary += f", {latency_ms} ms"
        elif lm is not None:
            # A bound model with no new history entries means the calls were
            # answered from the LM's cache -- which costs nothing, and the
            # accounting says so rather than implying no model ran.
            summary = f"0 new model calls recorded (cached), {latency_ms} ms"
        else:
            summary = f"0 model calls (deterministic fallbacks), {latency_ms} ms"
        self.reasoning_log.record(
            stage="usage",
            inputs={
                "model_calls": len(cycle_calls),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "retries": retries,
                "cost": 0.0 if cost is None else cost,
                "latency_ms": latency_ms,
            },
            output=summary,
            model=model_name(self.model_binding),
        )

    def spawn(self, persona: Any, intent: Any) -> Any:
        """Spawn a subagent runtime scoped to one Persona and reason the
        given intent through it, within the spawning limits the strategy in
        memory.md declares."""
        return self.spawner.spawn(self, persona, intent)

    def bind_tool(self, name: str, handler: Any) -> "Runtime":
        """Attach the executable behind a tool the stack declares (a Tools
        bullet in memory.md, or a stacked skill). Binding an undeclared
        name fails loudly at reasoning time -- code never grows the runtime
        a capability the natural-language authoring doesn't show."""
        self.tool_binder.bind(name, handler)
        return self

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
            start = calls_so_far(self.model_binding.lm)
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
                usage=usage_since(self.model_binding.lm, start),
            )
            if conforms:
                data.update(extracted)
        return data

    def _build_evidence(self, intent: Intent, plan: list[Workflow], recalled: str) -> Evidence:
        """Capture why this decision was reached -- separately from what
        was decided (Memory) or any pattern drawn from repeating it
        (Experience)."""
        if self.model_binding is not None and self.model_binding.lm is not None:
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


def _mcp_handler(client: Any, tool_name: str):
    """A forwarding handler for one MCP tool: it takes the arguments the
    model chose and calls the server, returning the tool's text. Wrapped by
    the binder's logged_handler like any tool, so the call is on the trail
    and a server failure returns to the model as text."""

    def handler(**arguments: Any) -> str:
        return client.call_tool(tool_name, arguments)

    return handler
