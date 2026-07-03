"""Journey -- durable, resumable, step-wise execution of the stack,
native to the runtime: what workflow platforms provide as checkpointed
graphs, EAR provides as a markdown file.

A Journey walks the authored stack one step at a time for one intent.
Each leg is a **full governed cycle** on the runtime -- the Governor
(approval gates included), knowledge, tools, the trail and memory all
apply per leg -- carrying the overall intent and the earlier legs'
conclusions forward. After every leg the journey's state is written to
its markdown file, so a crash mid-journey loses at most the leg in
flight: a fresh runtime resumes exactly where the record ends. The state
file is the same natural language as everything else -- inspectable,
diffable, parsed back by the same Section codec.

Control flow is authored in prose, judged at runtime:

- **Routing** (`Routes:` on the workflow): after each leg of a routed
  workflow, a routing judgment reads the authored routes and the leg's
  outcome and chooses the next authored step -- jump, continue in order,
  or conclude. The model chooses **only among authored steps, never
  invents one** (the Delegator's rule, applied to control flow); loops
  are legal, and a per-step revisit budget in code refuses runaway ones.
  Every choice is a `routing` record; with no model bound the authored
  routes are not judged and the journey continues in order, saying so.
- **Retries** (`Retries:` on the workflow, or an execution/resilience
  section in memory.md): a leg whose cycle *raises* is retried within the
  declared budget, every attempt a `retry` record; exhaustion ends the
  journey `FAILED`, on the record. With no budget declared a crash keeps
  its plain crash-and-resume semantics: the exception propagates and the
  record resumes where it ends.
- **Events** (`events/<journey-stem>*.md` beside the record): external
  signals as markdown -- their Context bullets merge into the journey's
  context on resume, each consumption an `event` record and a line in the
  journey file, consumed exactly once.

Governance shapes the walk: a hard block ends the journey (`BLOCKED`), an
approval gate parks it (`PENDING APPROVAL`) -- stamping when it parked and
which policies it awaits -- until `run` is called again with the human's
`Approval`; a completed journey is settled -- running it again replays
nothing. And because the stack is the source of truth, a journey refuses
to resume over a stack whose steps no longer match the legs it already
walked: silently continuing a different plan would forge the record.

`Journeys.run_all(runtime, directory)` is the runner: one pass over every
journey record -- resume the resumable, release the approved
(`approvals/<journey-stem>.md`), and escalate the expired: a gated policy
may declare `Escalate: after 3 days`, and a parked journey found past
that deadline is marked `ESCALATED` with the reason in its record. No
daemon -- the runner is one call, and *when* it runs is the operator's
cron. Honest about that.
"""

from __future__ import annotations

import calendar
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from .approval import Approval, ApprovalRequired
from .intent import Intent
from .reasoning_log import calls_so_far, model_name, usage_since
from .section import Section, coerce, labelled_blocks, normalize, parse_document, quote, unquote

COMPLETED = "completed"
IN_PROGRESS = "in progress"
BLOCKED = "BLOCKED"
PENDING = "PENDING APPROVAL"
FAILED = "FAILED"
ESCALATED = "ESCALATED"

# How many times routing may walk any single authored step before a jump
# back to it is refused -- loops are legal, runaway loops are not.
# Execution mechanics, not judgment.
REVISIT_BUDGET = 3

_PARKED_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# The routing judgment's non-numeric answers, normalized.
_CONCLUDE_WORDS = {"conclude", "concluded", "done", "end", "finish", "finished", "stop", "complete"}
_CONTINUE_WORDS = {"next", "continue", "proceed", "order", "forward", ""}


@dataclass
class Leg:
    """One walked leg of a journey: which authored step it walked (a
    routed journey may walk a step more than once), what, and how it
    ended."""

    number: int
    workflow: str
    instruction: str
    decision: str = ""
    status: str = IN_PROGRESS
    step: int = 0

    def __post_init__(self) -> None:
        if not self.step:
            self.step = self.number


@dataclass
class Journey:
    """A durable walk of the authored stack, one governed cycle per leg,
    its state a markdown file."""

    path: Union[str, Path]
    intent_text: str = ""
    context: dict = field(default_factory=dict)
    legs: list[Leg] = field(default_factory=list)
    status: str = IN_PROGRESS
    # Which authored step the walk continues at -- persisted so a routing
    # choice survives a crash instead of being silently re-judged.
    next_step: Optional[int] = None
    # Park metadata: when the journey parked and which policies it awaits,
    # so the runner can hold a declared escalation deadline against it.
    parked_at: str = ""
    awaiting: str = ""
    escalation_note: str = ""
    # Event documents already folded into the context, consumed once each.
    events_consumed: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @property
    def decision(self) -> str:
        return self.legs[-1].decision if self.legs else ""

    def run(self, runtime: Any, intent: Optional[Intent] = None, approval: Any = None) -> str:
        """Walk the remaining legs. On the first run an `intent` is
        required; a resumed journey reads its own record. Returns the
        journey's status; the final decision is `journey.decision`."""
        authored = self._authored_steps(runtime)
        if self.path.exists():
            self._load()
            self._verify_record_matches(authored)
        elif intent is not None:
            self.intent_text, self.context = intent.text, dict(intent.context)
            # The record exists before the first leg is walked, so even a
            # crash on leg one leaves a resumable journey behind.
            self._save()
        else:
            raise ValueError(f"Journey '{self.path}' has no record yet -- the first run needs an intent")

        if self.status in (COMPLETED, BLOCKED, FAILED):
            return self.status
        if self.status in (PENDING, ESCALATED) and approval is None:
            return self.status

        self._consume_events(runtime)

        # A parked leg is retried with the human's verdict; decided legs
        # are settled and never replayed.
        self.legs = [leg for leg in self.legs if leg.status == "decided"]
        walked: dict[int, int] = {}
        for leg in self.legs:
            walked[leg.step] = walked.get(leg.step, 0) + 1
        position: Optional[int] = self.next_step or (self.legs[-1].step + 1 if self.legs else 1)

        while position is not None and position <= len(authored):
            workflow, step = authored[position - 1]
            leg = Leg(
                number=len(self.legs) + 1, workflow=workflow.name, instruction=step.instruction, step=position
            )
            leg_approval, approval = approval, None  # the verdict applies to one leg only
            try:
                leg.decision = self._walk_leg(runtime, step, leg_approval, workflow, leg)
            except ApprovalRequired as parked:
                leg.decision, leg.status = str(parked), PENDING
                self.legs.append(leg)
                self.status = PENDING
                self.parked_at = time.strftime(_PARKED_TIME_FORMAT, time.gmtime())
                self.awaiting = ", ".join(policy.name for policy in parked.policies)
                self._save()
                return self.status
            except PermissionError as blocked:
                leg.decision, leg.status = str(blocked), BLOCKED
                self.legs.append(leg)
                self.status = BLOCKED
                self._save()
                return self.status
            except _RetriesExhausted as exhausted:
                leg.decision, leg.status = str(exhausted), FAILED
                self.legs.append(leg)
                self.status = FAILED
                self._save()
                return self.status
            leg.status = "decided"
            self.legs.append(leg)
            walked[position] = walked.get(position, 0) + 1
            self.status = IN_PROGRESS
            position = self._route(runtime, workflow, authored, leg, walked)
            self.next_step = position
            self._save()
        self.status = COMPLETED
        self.next_step = None
        self._save()
        return self.status

    # -- walking ---------------------------------------------------------------

    def _walk_leg(self, runtime: Any, step: Any, leg_approval: Any, workflow: Any, leg: Leg) -> str:
        """One leg's governed cycle, retried within the declared budget
        when the cycle raises. Governance outcomes (blocks, approval
        parks) are never retried -- they are decisions, not failures; and
        with no budget declared, a crash propagates exactly as before, so
        crash-and-resume semantics stay intact."""
        budget = self._retry_budget(runtime, workflow)
        attempts = 1 + (budget or 0)
        for attempt in range(1, attempts + 1):
            try:
                return str(runtime.reason(self._leg_intent(step), approval=leg_approval))
            except PermissionError:
                raise
            except Exception as error:  # noqa: BLE001 -- the declared budget governs what happens next
                if not budget:
                    raise
                exhausted = attempt == attempts
                self._record(
                    runtime,
                    stage="retry",
                    inputs={
                        "journey": self.path.name,
                        "step": f"{leg.step}: {leg.instruction}",
                        "attempt": attempt,
                        "budget": budget,
                        "error": str(error),
                    },
                    output=(
                        f"retry budget exhausted after {attempts} attempts -- journey FAILED"
                        if exhausted
                        else f"leg failed (attempt {attempt} of {attempts}) -- retrying"
                    ),
                    rationale=f"the workflow declares {budget} retr{'y' if budget == 1 else 'ies'} for a failed leg",
                )
                if exhausted:
                    raise _RetriesExhausted(
                        f"leg failed {attempts} times, retry budget exhausted: {error}"
                    ) from error
        raise _RetriesExhausted("unreachable")  # pragma: no cover -- the loop always returns or raises

    @staticmethod
    def _retry_budget(runtime: Any, workflow: Any) -> Optional[int]:
        """The leg retry budget in force: the workflow's own declaration
        wins; otherwise the strategy's execution section; otherwise none."""
        budget = getattr(workflow, "retry_budget", None)
        if budget is not None:
            return budget
        strategy = getattr(runtime, "strategy", None)
        return getattr(strategy, "leg_retry_budget", None) if strategy is not None else None

    def _route(
        self, runtime: Any, workflow: Any, authored: list[tuple], leg: Leg, walked: dict[int, int]
    ) -> Optional[int]:
        """Where the walk goes after a decided leg. Without authored
        routes: the next step in order (mechanics, no record). With
        routes: the model judges among the authored steps -- jump,
        continue, or conclude -- and code enforces that only authored,
        budget-respecting steps are walked. Every judged choice is a
        `routing` record; so is the honest no-model fallback."""
        default = leg.step + 1 if leg.step < len(authored) else None
        routes = getattr(workflow, "routes", "")
        if not routes:
            return default

        model_binding = getattr(runtime, "model_binding", None)
        lm = getattr(model_binding, "lm", None)
        inputs = {
            "journey": self.path.name,
            "routes": routes,
            "completed_step": f"{leg.step}: {leg.instruction}",
            "outcome": leg.decision,
        }
        if lm is None:
            self._record(
                runtime,
                stage="routing",
                inputs=inputs,
                output=self._describe_position(default),
                rationale="no model bound -- the authored routes were not judged; continuing in order",
            )
            return default

        from .signatures import RouteAfterLeg

        start = calls_so_far(lm)
        numbered = "\n".join(f"{number}: {step.instruction}" for number, (_, step) in enumerate(authored, start=1))
        result = RouteAfterLeg.run(
            lm,
            routes=routes,
            completed_step=f"{leg.step}: {leg.instruction}",
            outcome=leg.decision,
            steps=numbered,
        )
        usage = usage_since(lm, start)
        choice, rationale = normalize(str(result.next_step)), str(result.rationale)

        if choice in _CONCLUDE_WORDS:
            chosen: Optional[int] = None
        elif choice in _CONTINUE_WORDS:
            chosen = default
        else:
            digits = "".join(ch for ch in choice if ch.isdigit())
            number = int(digits) if digits else 0
            if not 1 <= number <= len(authored):
                # The model named no authored step: the route is refused,
                # never improvised -- and the record says so.
                rationale = f"'{result.next_step}' names no authored step -- continuing in order; {rationale}"
                chosen = default
            elif walked.get(number, 0) >= REVISIT_BUDGET:
                rationale = (
                    f"step {number} has already been walked {walked[number]} times -- the revisit "
                    f"budget ({REVISIT_BUDGET}) refuses the loop; continuing in order; {rationale}"
                )
                chosen = default
            else:
                chosen = number
        self._record(
            runtime,
            stage="routing",
            inputs=inputs,
            output=self._describe_position(chosen),
            rationale=rationale,
            model=model_name(model_binding),
            usage=usage,
        )
        return chosen

    @staticmethod
    def _describe_position(position: Optional[int]) -> str:
        return f"walk step {position} next" if position is not None else "conclude the journey"

    def _consume_events(self, runtime: Any) -> None:
        """Fold pending event documents into the journey's context --
        external signals as markdown, the same way approvals work. An
        event is `events/<journey-stem>*.md` beside the record, its facts
        ordinary Context bullets; each is consumed exactly once, on the
        record and in the journey file."""
        events_dir = self.path.parent / "events"
        if not events_dir.is_dir():
            return
        stem = self.path.stem
        consumed_any = False
        for event_path in sorted(events_dir.glob("*.md")):
            if not (event_path.stem == stem or event_path.stem.startswith(stem + "-")):
                continue
            if event_path.name in self.events_consumed:
                continue
            facts = Intent.from_markdown(event_path.read_text(encoding="utf-8")).context
            self.context.update(facts)
            self.events_consumed.append(event_path.name)
            consumed_any = True
            self._record(
                runtime,
                stage="event",
                inputs={"journey": self.path.name, "event": event_path.name, "facts": dict(facts)},
                output=f"consumed {len(facts)} fact(s) into the journey context",
                rationale="an external signal arrived as a markdown event document",
            )
        if consumed_any:
            self._save()

    @staticmethod
    def _record(runtime: Any, **kwargs: Any) -> None:
        log = getattr(runtime, "reasoning_log", None)
        if log is not None:
            log.record(**kwargs)
            log.flush()

    @staticmethod
    def _authored_steps(runtime: Any) -> list[tuple]:
        authored = [
            (workflow, step)
            for process in getattr(runtime, "processes", [])
            for workflow in process.workflows
            for step in workflow.steps
        ]
        if not authored:
            raise ValueError(f"Runtime '{runtime.name}' has no workflow steps for a journey to walk")
        return authored

    def _leg_intent(self, step: Any) -> Intent:
        text = step.instruction
        if self.intent_text:
            text += f"\n\nThe overall intent this step serves: {self.intent_text}"
        decided = [leg for leg in self.legs if leg.status == "decided"]
        if decided:
            transcript = "\n".join(f"- {leg.instruction} -> {leg.decision}" for leg in decided)
            text += f"\n\nEarlier legs concluded:\n{transcript}"
        return Intent(text=text, context=dict(self.context))

    def _verify_record_matches(self, authored: list[tuple]) -> None:
        for leg in self.legs:
            if leg.step > len(authored):
                raise ValueError(
                    f"Journey '{self.path}' walked authored step {leg.step} but the stack now authors "
                    f"only {len(authored)} steps -- the record and the stack no longer match"
                )
            _, step = authored[leg.step - 1]
            if leg.instruction != step.instruction:
                raise ValueError(
                    f"Journey '{self.path}' leg {leg.number} walked '{leg.instruction}' but the stack "
                    f"now authors '{step.instruction}' there -- resuming over a changed plan would "
                    "forge the record; start a new journey instead"
                )

    # -- the markdown record ----------------------------------------------------

    def _save(self) -> None:
        title = self.intent_text.partition("\n")[0].strip()
        lines = [
            f"# Journey -- {title}",
            "",
            f"Status: {self.status}",
            f"Legs walked: {len(self.legs)}",
        ]
        if self.next_step is not None:
            lines.append(f"Next step: {self.next_step}")
        if self.parked_at:
            lines.append(f"Parked: {self.parked_at}")
        if self.awaiting:
            lines.append(f"Awaiting: {self.awaiting}")
        lines += ["", "## Intent", "", quote(self.intent_text)]
        if self.context:
            lines += ["", "Context:", ""]
            lines += [f"- {key}: {value}" for key, value in self.context.items()]
        if self.events_consumed:
            lines += ["", "## Events", ""]
            lines += [f"- {name}" for name in self.events_consumed]
        for leg in self.legs:
            lines += [
                "",
                f"## Leg {leg.number} -- {leg.workflow} ({leg.status})",
                "",
                f"Step: {leg.step}",
                f"Instruction: {leg.instruction}",
                "",
                "Decision:",
                quote(leg.decision),
            ]
        if self.escalation_note:
            lines += ["", "## Escalation", "", quote(self.escalation_note)]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _load(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        all_lines = text.replace("\r\n", "\n").split("\n")
        header = Section(name="journey", lines=all_lines).body(
            field_keys=("status", "next step", "parked", "awaiting")
        )
        self.status = header.field("status") or IN_PROGRESS
        next_digits = "".join(ch for ch in header.field("next step") if ch.isdigit())
        self.next_step = int(next_digits) if next_digits else None
        self.parked_at = header.field("parked")
        self.awaiting = header.field("awaiting")

        self.legs = []
        self.events_consumed = []
        self.escalation_note = ""
        for section in parse_document(text).sections:
            key = normalize(section.name)
            if key == "intent":
                self.intent_text = unquote(section.lines)
                self.context = {}
                for bullet in section.body().bullets:
                    name, separator, value = bullet.partition(": ")
                    if not separator:
                        name, separator, value = bullet.partition(":")
                    if separator:
                        self.context[name.strip()] = coerce(value)
            elif key == "events":
                self.events_consumed = list(section.body().bullets)
            elif key == "escalation":
                self.escalation_note = unquote(section.lines)
            elif key.startswith("leg"):
                self.legs.append(self._leg_from_section(section))

    @staticmethod
    def _leg_from_section(section: Section) -> Leg:
        # "Leg 3 -- Underwriting Workflow (decided)"
        head, _, status = section.name.rpartition(" (")
        number_part, _, workflow = head.partition(" -- ")
        number_text = "".join(ch for ch in number_part if ch.isdigit())
        body = section.body(field_keys=("instruction", "step"))
        step_digits = "".join(ch for ch in body.field("step") if ch.isdigit())
        return Leg(
            number=int(number_text or 0),
            workflow=workflow.strip(),
            instruction=body.field("instruction"),
            decision=labelled_blocks(section.lines).get("decision", ""),
            status=status.rstrip(")").strip() or IN_PROGRESS,
            step=int(step_digits) if step_digits else 0,
        )


class _RetriesExhausted(RuntimeError):
    """A leg failed past its declared retry budget -- the journey gives
    up, on the record. Internal to the walk; the journey surfaces it as
    status FAILED, never as an exception."""


@dataclass
class Journeys:
    """The journey runner: one pass over every journey record in a
    directory -- resume the resumable, release the approved, escalate the
    expired. No daemon: the runner is one call, and *when* it runs is the
    operator's cron. Returns each record's resulting status by filename."""

    def run_all(self, runtime: Any, directory: Union[str, Path], now: Optional[float] = None) -> dict[str, str]:
        directory = Path(directory)
        outcomes: dict[str, str] = {}
        for path in sorted(directory.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            if not text.lstrip().startswith("# Journey"):
                continue
            journey = Journey(path)
            journey._load()
            if journey.status in (COMPLETED, BLOCKED, FAILED):
                outcomes[path.name] = journey.status
            elif journey.status in (PENDING, ESCALATED):
                outcomes[path.name] = self._release_or_escalate(runtime, journey, directory, now)
            else:
                outcomes[path.name] = Journey(path).run(runtime)
        self._apply_retention(runtime, now)
        return outcomes

    @staticmethod
    def _apply_retention(runtime: Any, now: Optional[float]) -> None:
        """Rotate the reasoning trail down to the declared retention window
        -- the runner is where retention is applied (no daemon), on the
        record, never a silent purge."""
        import datetime as _datetime

        strategy = getattr(runtime, "strategy", None)
        retention_days = getattr(strategy, "retention_days", None) if strategy is not None else None
        log = getattr(runtime, "reasoning_log", None)
        if retention_days and log is not None:
            moment = (
                _datetime.datetime.fromtimestamp(now, _datetime.timezone.utc) if now is not None else None
            )
            log.rotate(retention_days, now=moment)

    def _release_or_escalate(
        self, runtime: Any, journey: Journey, directory: Path, now: Optional[float]
    ) -> str:
        approval = self._approval_for(runtime, journey, directory)
        if approval is not None:
            return Journey(journey.path).run(runtime, approval=approval)
        return self._maybe_escalate(runtime, journey, now)

    @staticmethod
    def _approval_for(runtime: Any, journey: Journey, directory: Path) -> Optional[Approval]:
        """The human verdict for a parked journey, read from
        `approvals/<journey-stem>.md`. An unreadable verdict is refused
        loudly on the record -- the journey stays parked, never guessed
        through the gate."""
        path = directory / "approvals" / f"{Path(journey.path).stem}.md"
        if not path.exists():
            return None
        approval = Approval.from_markdown(path.read_text(encoding="utf-8"))
        if approval.verdict is None:
            Journey._record(
                runtime,
                stage="approval",
                inputs={"journey": Path(journey.path).name, "approval": str(path)},
                output=f"unreadable verdict in {path.name} -- the journey stays parked",
                rationale="a verdict outside the vocabulary is excluded, never guessed",
            )
            return None
        return approval

    def _maybe_escalate(self, runtime: Any, journey: Journey, now: Optional[float]) -> str:
        """Hold the awaited policies' declared escalation deadline against
        a parked journey. Past the deadline the journey is marked
        ESCALATED with the reason in its record -- still releasable by an
        approval, but no longer quietly waiting."""
        if journey.status == ESCALATED:
            return ESCALATED
        policy = self._earliest_escalating_policy(runtime, journey.awaiting)
        if policy is None or not journey.parked_at:
            return journey.status
        try:
            parked_epoch = calendar.timegm(time.strptime(journey.parked_at, _PARKED_TIME_FORMAT))
        except ValueError:
            return journey.status
        moment = now if now is not None else time.time()
        if moment < parked_epoch + policy.escalation_days * 86400:
            return journey.status
        journey.status = ESCALATED
        journey.escalation_note = (
            f"Parked since {journey.parked_at} awaiting {journey.awaiting}; policy "
            f"'{policy.name}' declares escalation '{policy.escalation}' and the deadline has passed."
        )
        journey._save()
        Journey._record(
            runtime,
            stage="escalation",
            inputs={
                "journey": Path(journey.path).name,
                "parked": journey.parked_at,
                "awaiting": journey.awaiting,
                "declared": policy.escalation,
            },
            output=f"journey ESCALATED -- {journey.escalation_note}",
            rationale="the policy's declared escalation period passed with no approval",
        )
        return ESCALATED

    @staticmethod
    def _earliest_escalating_policy(runtime: Any, awaiting: str) -> Optional[Any]:
        """Among the policies a journey awaits, the one whose declared
        escalation period is shortest -- or None when none declares one."""
        names = {normalize(name) for name in awaiting.split(",") if name.strip()}
        if not names:
            return None
        candidates = list(getattr(runtime, "policies", []) or [])
        for process in getattr(runtime, "processes", []) or []:
            for workflow in process.workflows:
                candidates.extend(workflow.policies)
        matching = [
            policy
            for policy in candidates
            if normalize(policy.name) in names and getattr(policy, "escalation_days", None) is not None
        ]
        return min(matching, key=lambda policy: policy.escalation_days) if matching else None
