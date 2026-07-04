"""Delegator -- run a delegated Persona as its own isolated sub-agent
cycle. EAR's declarative take on sub-agent fan-out.

DeerFlow's lead agent spawns sub-agents dynamically, each in its own
isolated context, and reports their results back for synthesis. EAR keeps
the fan-out *declared* rather than emergent: a `Workflow` marked
`parallel=True` dispatches each of its delegated Personas through this
Delegator instead of stacking every persona into one shared prompt.
"Isolated context" means exactly this, mechanically: each delegated
Persona reasons alone, in its own call to the Reasoner, seeing only its
own instructions and skills -- never another persona's.

`Synthesizer` is the other half: it folds what several Delegator calls
return into one final decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .intent import Intent
from .workflow import Workflow


@dataclass
class Delegator:
    """A Delegator delegates one Persona to reason alone. The parent
    declares *who* runs (the Workflow's delegated personas); each delegate
    still reasons through the ordinary Reasoner -- LLM-judged, with the
    same dependency-free fallback as every other stage -- just scoped to a
    fresh, single-persona capabilities block instead of the shared plan."""

    def delegate(self, runtime: Any, intent: Intent, persona: Any) -> Any:
        isolated = Workflow(name=f"{persona.name} (sub-agent)")
        isolated.add_persona(persona)
        return runtime.reasoner.reason(intent, runtime=runtime, plan=[isolated])
