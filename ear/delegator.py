"""Delegator -- delegate: assign undelegated workflow steps to personas at
runtime.

Authors delegate steps explicitly in workflow.md by naming the persona;
any step they leave undelegated is completed dynamically here. When a
ModelBinding is active, the LLM reads each undelegated step's instruction
against the workflow's available personas -- their standing instructions
and stacked skills -- and assigns the best-suited one. With no model the
step simply stays undelegated, exactly as authored, and the Reasoner still
reasons it with the workflow's other capabilities.

Two deliberate boundaries: the pool is the workflow's own personas (a step
is never handed to a persona its workflow never mentions), and an explicit
delegation written by the author is never overridden. Each inferred
assignment is written to the ReasoningLog, so the completed authoring is
on the audit trail, not silent. Assignments persist on the Step -- the
runtime completes the stack once, it does not re-deal the work every
cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .intent import Intent
from .persona import Persona
from .reasoning_log import calls_so_far, model_name, usage_since
from .section import normalize
from .step import Step
from .workflow import Workflow


@dataclass
class Delegator:
    """A Delegator completes the delegation of a plan's steps: undelegated
    steps are assigned to the workflow's best-suited persona by LLM
    judgment; authored delegations are never touched."""

    def delegate(self, runtime: Any, intent: Intent, plan: list[Workflow]) -> list[Workflow]:
        model_binding = getattr(runtime, "model_binding", None)
        if model_binding is None or model_binding.lm is None:
            return plan
        log = getattr(runtime, "reasoning_log", None)
        for workflow in plan:
            undelegated = [(number, step) for number, step in enumerate(workflow.steps, start=1) if step.persona is None]
            pool = workflow.delegated_personas()
            if not undelegated or not pool:
                continue
            start = calls_so_far(model_binding.lm)
            assignments = self._infer_with_llm(undelegated, pool, model_binding.lm)
            applied = self._apply(undelegated, pool, assignments)
            if applied and log is not None:
                log.record(
                    stage="delegation",
                    inputs={
                        "workflow": workflow.name,
                        "undelegated_steps": [step.instruction for _, step in undelegated],
                        "available_personas": [persona.name for persona in pool],
                    },
                    output="; ".join(f"step {number} -> {persona.name}" for number, persona in applied),
                    model=model_name(model_binding),
                    usage=usage_since(model_binding.lm, start),
                )
        return plan

    @staticmethod
    def _infer_with_llm(undelegated: list[tuple[int, Step]], pool: list[Persona], lm: Any) -> list[str]:
        from .signatures import DelegateSteps

        steps = "\n".join(f"{number}: {step.instruction}" for number, step in undelegated)
        personas = "\n".join(
            f"{persona.name}: {persona.instructions or 'no standing instructions'}"
            + (f" -- skills: {', '.join(skill.name for skill in persona.skills)}" if persona.skills else "")
            for persona in pool
        )
        result = DelegateSteps.run(lm, steps=steps, personas=personas)
        return list(result.assignments)

    @staticmethod
    def _apply(
        undelegated: list[tuple[int, Step]],
        pool: list[Persona],
        assignments: list[str],
    ) -> list[tuple[int, Persona]]:
        """Apply only assignments that resolve to a real step number and a
        real persona -- an unusable answer leaves the step as authored."""
        steps_by_number = {number: step for number, step in undelegated}
        personas_by_name = {normalize(persona.name): persona for persona in pool}
        applied: list[tuple[int, Persona]] = []
        for assignment in assignments:
            number_text, separator, persona_name = str(assignment).partition(":")
            if not separator:
                continue
            try:
                number = int(number_text.strip())
            except ValueError:
                continue
            step = steps_by_number.get(number)
            persona = personas_by_name.get(normalize(persona_name))
            if step is not None and persona is not None and step.persona is None:
                step.persona = persona
                applied.append((number, persona))
        return applied