"""Deliberator -- deliberate via the runtime's Reasoner given its current
state. Kept distinct from Decider (committing to a decision) and Validator
(validating it) so deliberation, decision and validation are three
separate, inspectable steps rather than one opaque call.

A Workflow marked `parallel=True` is deliberated differently: instead of
stacking its delegated Personas into one shared reasoning call, each is
delegated as its own isolated sub-agent (`Delegator`), and their results
are folded into one decision (`Synthesizer`) -- EAR's declarative take on
sub-agent fan-out. Workflows left sequential (the default) keep reasoning
exactly as they always have, in one shared call; a plan with no parallel
workflow at all is unaffected by any of this."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .intent import Intent


@dataclass
class Deliberator:
    """A Deliberator deliberates by handing the intent to the runtime's
    Reasoner -- or, for any `parallel` Workflow in the plan, by fanning out
    to each of its delegated Personas as an isolated sub-agent and
    synthesizing their results."""

    def deliberate(self, runtime: Any, intent: Intent, plan: Any = None) -> Any:
        plan = plan or []
        parallel_workflows = [workflow for workflow in plan if getattr(workflow, "parallel", False)]
        if not parallel_workflows:
            return runtime.reasoner.reason(intent, runtime=runtime, plan=plan)

        sequential_workflows = [workflow for workflow in plan if not getattr(workflow, "parallel", False)]
        sub_decisions: list[tuple[str, Any]] = []
        for workflow in parallel_workflows:
            for persona in workflow.delegated_personas():
                decision = runtime.delegator.delegate(runtime, intent, persona)
                sub_decisions.append((persona.name, decision))
        if sequential_workflows:
            shared = runtime.reasoner.reason(intent, runtime=runtime, plan=sequential_workflows)
            sub_decisions.append(("(shared workflow reasoning)", shared))

        if not sub_decisions:
            # A parallel workflow with no delegated personas -- nothing to
            # fan out, so fall back to reasoning the plan as given.
            return runtime.reasoner.reason(intent, runtime=runtime, plan=plan)

        # Recorded here (rather than returned) so it survives into this
        # cycle's Evidence alongside tool calls -- sub-agent provenance
        # audited the same way everything else in a cycle is.
        runtime._cycle_sub_agent_decisions = list(sub_decisions)
        return runtime.synthesizer.synthesize(runtime, intent, sub_decisions)
