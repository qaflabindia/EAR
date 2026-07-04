"""Assessor -- assess whether a Goal is met after a cycle. The "you are
done when" counterpart to the Governor's "you may not": the Governor gates
a cycle before it runs, the Assessor judges the cycle's decision after it
runs and tells the Runtime whether to stop or iterate again.

When a ModelBinding is active, completion is judged by an LLM reading the
Goal statement against the decision and the run's history -- genuine
natural-language judgment, not a hardcoded rule. Without one, it falls
back to the Goal's safe `fallback_expression` so goal-driven iteration
stays usable and testable offline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .goal import Goal
from .intent import Intent
from .safe_evaluator import MissingVariableError, safe_eval

# Blocker strings that mean "nothing is actually blocking" -- normalized to "".
_NO_BLOCKER = {"", "none", "n/a", "na", "no", "null", "false"}


@dataclass
class Assessor:
    """An Assessor assesses a Goal against a cycle's decision, returning
    `(done, blocker)`. `done` is True once the Goal is satisfied; `blocker`
    is a short reason the loop cannot make further progress (e.g.
    "needs_input", "blocked", "failed") or "" when none. The Runtime stops
    iterating when either is set."""

    def assess(self, runtime: Any, intent: Intent, goal: Goal, decision: Any) -> tuple[bool, str]:
        model_binding = getattr(runtime, "model_binding", None)
        if model_binding is not None and getattr(model_binding, "lm", None) is not None:
            history = self._history(runtime)
            return self._assess_with_llm(model_binding.lm, goal, decision, history)
        return self._assess_by_fallback(goal, intent, decision), ""

    @staticmethod
    def _assess_by_fallback(goal: Goal, intent: Intent, decision: Any) -> bool:
        # No expression to check offline -> nothing to iterate on, so the
        # Goal is treated as met after one cycle rather than looping to the
        # cap for no reason.
        if not goal.fallback_expression:
            return True
        variables = {**intent.context, "decision": str(decision)}
        try:
            return bool(safe_eval(goal.fallback_expression, variables))
        except MissingVariableError:
            # A variable the Goal expects isn't set yet -> not done, keep
            # iterating (still bounded by max_cycles). Genuinely unsafe
            # expressions raise, exactly as they do for Policy.
            return False

    def _assess_with_llm(self, lm: Any, goal: Goal, decision: Any, history: str) -> tuple[bool, str]:
        if not goal.statement:
            return True, ""
        import dspy

        from .signatures import AssessGoalCompletion

        assessor = dspy.Predict(AssessGoalCompletion)
        with dspy.context(lm=lm):
            result = assessor(goal_statement=goal.statement, decision=str(decision), history=history)
        blocker = str(getattr(result, "blocker", "") or "").strip()
        if blocker.lower() in _NO_BLOCKER:
            blocker = ""
        return bool(result.complete), blocker

    @staticmethod
    def _history(runtime: Any) -> str:
        memory = getattr(runtime, "memory", None)
        if memory is not None and len(memory):
            return memory.context_window()
        return "no prior cycles"
