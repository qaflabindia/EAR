"""Goal -- a session goal: one completion condition that drives the runtime
forward, cycle after cycle, until it is met, genuinely blocked, or the
budget is spent.

This is the autonomous-continuation loop a heavyweight harness gives you,
built the EAR way -- the model judges, code enforces, everything records.
A `Goal` is a plain-English completion condition ("finish the analysis and
state a recommendation"). `GoalKeeper.pursue(runtime, goal, intent)` runs
the first cycle, then after each cycle asks the model, through
`JudgeGoalProgress`, one question: is the goal met -- and if not, *why
not*, as exactly one **typed blocker**:

    goal_not_met_yet   more work would help -> continue autonomously
    needs_user_input   a human must supply something -> stop, surface it
    external_wait      waiting on an outside event/system -> stop, surface it
    missing_evidence   the work cannot be verified -> stop, surface it
    run_failed         it went wrong and cannot recover -> stop, surface it

Only `goal_not_met_yet` earns a continuation: the keeper derives the next
step from the evaluator's own `next_step` and drives another cycle. The
loop is bounded in code -- a maximum number of continuations (default 8)
and a no-progress breaker that stops after the same non-progress verdict
repeats (default 2 times), so an autonomous loop can never run away. A
governance stop maps to a blocker without special-casing: an approval gate
(`ApprovalRequired`) is `needs_user_input`; any other `PermissionError` is
`run_failed`.

Every evaluation lands on the trail (stage `goal`) with its blocker and
evidence. With no model bound the goal cannot be judged, so the keeper
stops at `ungraded` after the first cycle and never fabricates satisfaction
or a continuation -- a judgment nobody made is never written down as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .intent import Intent
from .reasoning_log import calls_so_far, model_name, usage_since
from .section import normalize, parse_document

# The blocker taxonomy. Only GOAL_NOT_MET continues the loop; the rest are
# terminal and surface their reason. SATISFIED and UNGRADED are the
# keeper's own outcomes, not model-named blockers.
GOAL_NOT_MET = "goal_not_met_yet"
NEEDS_INPUT = "needs_user_input"
EXTERNAL_WAIT = "external_wait"
MISSING_EVIDENCE = "missing_evidence"
RUN_FAILED = "run_failed"
SATISFIED = "satisfied"
UNGRADED = "ungraded"

_STOP_BLOCKERS = frozenset({NEEDS_INPUT, EXTERNAL_WAIT, MISSING_EVIDENCE, RUN_FAILED})
_KNOWN_BLOCKERS = frozenset({GOAL_NOT_MET, NEEDS_INPUT, EXTERNAL_WAIT, MISSING_EVIDENCE, RUN_FAILED})

DEFAULT_MAX_CONTINUATIONS = 8
DEFAULT_NO_PROGRESS_CAP = 2


@dataclass
class Goal:
    """One session goal: a completion condition in plain English, and how
    many autonomous continuations it may drive before the keeper gives up."""

    condition: str
    max_continuations: int = DEFAULT_MAX_CONTINUATIONS
    active: bool = True

    @classmethod
    def from_markdown(cls, text: str) -> "Goal":
        """Read a goal from a markdown document: a `## Goal` section's prose
        if present, otherwise the whole document as the condition."""
        for section in parse_document(text).sections:
            if "goal" in normalize(section.name):
                prose = section.body().prose.strip()
                if prose:
                    return cls(condition=prose)
        stripped = "\n".join(line for line in text.splitlines() if not line.startswith("#")).strip()
        return cls(condition=stripped or text.strip())


@dataclass
class GoalEvaluation:
    """One judgment of the goal after a cycle: met or not, the single typed
    blocker if not, the visible evidence, and the next step to try."""

    satisfied: bool
    blocker: str
    evidence: str = ""
    next_step: str = ""


@dataclass
class GoalOutcome:
    """Where the pursuit ended: satisfied, blocked (a terminal blocker),
    exhausted (a cap tripped), or ungraded (no model). Carries the final
    decision, the deciding evidence, and every evaluation made."""

    goal: str
    status: str  # "satisfied" | "blocked" | "exhausted" | "ungraded"
    blocker: str
    decision: str = ""
    evidence: str = ""
    continuations: int = 0
    history: list[GoalEvaluation] = field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return self.status == "satisfied"


@dataclass
class GoalKeeper:
    """Pursues a Goal to a terminal outcome, bounded in code and judged by
    the model."""

    no_progress_cap: int = DEFAULT_NO_PROGRESS_CAP

    def pursue(self, runtime: Any, goal: Union[Goal, str], intent: Union[Intent, str]) -> GoalOutcome:
        goal = goal if isinstance(goal, Goal) else Goal(condition=str(goal))
        intent = intent if isinstance(intent, Intent) else Intent(text=str(intent))

        decision, terminal = self._drive(runtime, intent)
        if terminal is not None:
            return self._blocked(runtime, goal, decision, *terminal, continuations=0, history=[])

        history: list[GoalEvaluation] = []
        continuations = 0
        repeats = 0
        last_signature: Optional[str] = None

        while True:
            evaluation, usage, model = self._evaluate(runtime, goal, decision)
            self._record(runtime, goal, evaluation, continuations, usage, model)
            history.append(evaluation)

            if evaluation.blocker == UNGRADED:
                return GoalOutcome(
                    goal=goal.condition,
                    status="ungraded",
                    blocker=UNGRADED,
                    decision=decision,
                    evidence=evaluation.evidence,
                    continuations=continuations,
                    history=history,
                )
            if evaluation.satisfied:
                goal.active = False
                return GoalOutcome(
                    goal=goal.condition,
                    status="satisfied",
                    blocker=SATISFIED,
                    decision=decision,
                    evidence=evaluation.evidence,
                    continuations=continuations,
                    history=history,
                )
            if evaluation.blocker in _STOP_BLOCKERS:
                return GoalOutcome(
                    goal=goal.condition,
                    status="blocked",
                    blocker=evaluation.blocker,
                    decision=decision,
                    evidence=evaluation.evidence,
                    continuations=continuations,
                    history=history,
                )

            # goal_not_met_yet: continue, if the budget and progress allow.
            signature = normalize(evaluation.evidence + " | " + evaluation.next_step)
            repeats = repeats + 1 if signature == last_signature else 0
            last_signature = signature
            if repeats >= self.no_progress_cap:
                return self._exhausted(goal, "no_progress", decision, evaluation, continuations, history)
            if continuations >= goal.max_continuations:
                return self._exhausted(goal, "max_continuations", decision, evaluation, continuations, history)

            continuations += 1
            follow_up = Intent(
                text=evaluation.next_step.strip() or goal.condition,
                context=dict(intent.context),
            )
            decision, terminal = self._drive(runtime, follow_up)
            if terminal is not None:
                return self._blocked(runtime, goal, decision, *terminal, continuations=continuations, history=history)

    # -- driving a cycle -------------------------------------------------------

    @staticmethod
    def _drive(runtime: Any, intent: Intent) -> tuple[str, Optional[tuple[str, str]]]:
        """Run one cycle. A governance stop is not an exception to swallow
        -- it is a typed blocker: an approval gate is `needs_user_input`,
        any other refusal is `run_failed`."""
        from .approval import ApprovalRequired

        try:
            return str(runtime.reason(intent)), None
        except ApprovalRequired as parked:
            return "", (NEEDS_INPUT, str(parked))
        except PermissionError as blocked:
            return "", (RUN_FAILED, str(blocked))

    # -- judging the goal ------------------------------------------------------

    def _evaluate(self, runtime: Any, goal: Goal, decision: str) -> tuple[GoalEvaluation, Any, str]:
        model_binding = getattr(runtime, "model_binding", None)
        lm = getattr(model_binding, "lm", None)
        if lm is None:
            return (
                GoalEvaluation(
                    satisfied=False,
                    blocker=UNGRADED,
                    evidence="no model bound -- a goal is a judgment, and none was made",
                ),
                None,
                "",
            )
        from .signatures import JudgeGoalProgress

        start = calls_so_far(lm)
        result = JudgeGoalProgress.run(lm, goal=goal.condition, progress=self._progress(runtime, decision))
        satisfied = bool(result.satisfied)
        blocker = SATISFIED if satisfied else _read_blocker(result.blocker)
        evaluation = GoalEvaluation(
            satisfied=satisfied,
            blocker=blocker,
            evidence=str(result.evidence).strip(),
            next_step=str(result.next_step).strip(),
        )
        return evaluation, usage_since(lm, start), model_name(model_binding)

    @staticmethod
    def _progress(runtime: Any, decision: str) -> str:
        """What the evaluator sees: the latest decision, plus the runtime's
        recent remembered context so it judges the whole thread, not one
        turn in isolation."""
        parts = [f"Most recent outcome:\n{decision}"]
        memory = getattr(runtime, "memory", None)
        if memory is not None and len(memory):
            window = memory.context_window()
            if window:
                parts.append(f"Recent history:\n{window}")
        return "\n\n".join(parts)

    # -- terminal outcomes -----------------------------------------------------

    def _blocked(
        self,
        runtime: Any,
        goal: Goal,
        decision: str,
        blocker: str,
        evidence: str,
        continuations: int,
        history: list,
    ) -> GoalOutcome:
        evaluation = GoalEvaluation(satisfied=False, blocker=blocker, evidence=evidence)
        self._record(runtime, goal, evaluation, continuations, None, model_name(getattr(runtime, "model_binding", None)))
        history = history + [evaluation]
        return GoalOutcome(
            goal=goal.condition,
            status="blocked",
            blocker=blocker,
            decision=decision,
            evidence=evidence,
            continuations=continuations,
            history=history,
        )

    @staticmethod
    def _exhausted(
        goal: Goal, reason: str, decision: str, evaluation: GoalEvaluation, continuations: int, history: list
    ) -> GoalOutcome:
        return GoalOutcome(
            goal=goal.condition,
            status="exhausted",
            blocker=reason,
            decision=decision,
            evidence=evaluation.evidence,
            continuations=continuations,
            history=history,
        )

    @staticmethod
    def _record(
        runtime: Any, goal: Goal, evaluation: GoalEvaluation, continuation: int, usage: Any, model: str
    ) -> None:
        log = getattr(runtime, "reasoning_log", None)
        if log is None:
            return
        output = SATISFIED if evaluation.satisfied else evaluation.blocker
        log.record(
            stage="goal",
            inputs={
                "goal": goal.condition,
                "continuation": continuation,
                "next_step": evaluation.next_step,
            },
            output=output,
            rationale=evaluation.evidence,
            model="" if evaluation.blocker == UNGRADED else model,
            usage=usage,
        )
        log.flush()


def _read_blocker(value: Any) -> str:
    """Map the model's blocker word onto the taxonomy. An unrecognized but
    not-satisfied answer defaults to `goal_not_met_yet` -- the only reading
    that keeps the loop honest without inventing a stop the model did not
    name."""
    key = normalize(str(value)).replace(" ", "_")
    for blocker in _KNOWN_BLOCKERS:
        if blocker in key or key in blocker:
            return blocker
    return GOAL_NOT_MET
