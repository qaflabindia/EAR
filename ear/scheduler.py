"""Scheduler -- order the Composer's composed plan before execution.

Ordering is a runtime judgment: when a ModelBinding is active and the plan
holds more than one workflow, the LLM orders them for the intent at hand --
prerequisites and information-producing workflows first. With no model (or
a single workflow) it falls back to a defensive copy in composition order,
so the package stays fully usable offline. Either way the ordering -- and
which mind chose it -- is written to the runtime's ReasoningLog.

Ordering is the only judgment delegated: every composed workflow stays in
the schedule. Dropping one would be a selection decision made in the wrong
place, and the Governor still has to see the whole plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .intent import Intent
from .reasoning_log import model_name
from .workflow import Workflow


@dataclass
class Scheduler:
    """A Scheduler orders a composed plan: an LLM judgment when a model is
    active, composition order otherwise."""

    def schedule(self, plan: list[Workflow], runtime: Any = None, intent: Optional[Intent] = None) -> list[Workflow]:
        scheduled = list(plan)
        model_binding = getattr(runtime, "model_binding", None)
        if len(scheduled) > 1 and intent is not None and model_binding is not None and model_binding.lm is not None:
            scheduled = self._order_with_llm(scheduled, intent, model_binding.lm)
        log = getattr(runtime, "reasoning_log", None)
        if log is not None and len(scheduled) > 1:
            log.record(
                stage="scheduling",
                inputs={
                    "intent": intent.text if intent is not None else "",
                    "composed_order": [workflow.name for workflow in plan],
                },
                output=", ".join(workflow.name for workflow in scheduled),
                model=model_name(model_binding),
            )
        return scheduled

    @staticmethod
    def _order_with_llm(plan: list[Workflow], intent: Intent, lm: Any) -> list[Workflow]:
        import dspy

        from .signatures import ScheduleWorkflows

        summaries = "\n".join(
            f"{workflow.name}: " + ("; ".join(step.instruction for step in workflow.steps[:4]) or "no steps")
            for workflow in plan
        )
        orderer = dspy.Predict(ScheduleWorkflows)
        with dspy.context(lm=lm):
            result = orderer(intent_text=intent.text, workflows=summaries)
        by_name = {workflow.name: workflow for workflow in plan}
        ordered: list[Workflow] = []
        for name in result.ordered_workflow_names:
            workflow = by_name.get(name)
            if workflow is not None and all(workflow is not placed for placed in ordered):
                ordered.append(workflow)
        # Every workflow the Composer produced stays in the schedule: any
        # the LLM forgot keep their composition order at the end.
        placed_ids = {id(workflow) for workflow in ordered}
        ordered.extend(workflow for workflow in plan if id(workflow) not in placed_ids)
        return ordered