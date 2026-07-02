"""Reasoner -- the discriminative intelligence the runtime starts.

This is the one place DSPy and GEPA are used deliberately, not sprinkled
across every class: the Reasoner's core judgment is always a DSPy
`Predict(ReasonAboutIntent)` call against whichever ModelBinding is active
-- a natural-language prompt, not a hardcoded decision tree -- and GEPA is
the single sanctioned optimizer for reflectively improving that one
program's prompt against a labelled trainset. Every other stage in this
package either has no judgment call to make (Selector, Composer,
Scheduler) or uses a DSPy signature directly (Policy, Discoverer,
Explainer) without needing GEPA optimization of its own.

With no ModelBinding active at all, reasoning falls back to a
deterministic summary, so the runtime is usable -- and testable -- with no
LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .intent import Intent


@dataclass
class Reasoner:
    """The reasoning layer a Runtime invokes once an Intent has cleared
    every Policy gate. Runtime activates its ModelBinding (LLM provider)
    first; Reasoner then reasons with a compiled DSPy program if one is
    attached, or by predicting directly against the activated
    ModelBinding's LM otherwise. Call `compile_with_dspy` to attach a
    custom DSPy program, or `optimize_with_gepa` to reflectively improve
    the default one against examples."""

    program: Optional[Any] = None

    def reason(self, intent: Intent, runtime: Any = None, plan: Any = None) -> Any:
        capabilities = self._render_capabilities(plan)
        if self.program is not None:
            return self._run_program(intent, capabilities)
        model_binding = getattr(runtime, "model_binding", None)
        if model_binding is not None and model_binding.lm is not None:
            return self._reason_with_llm(intent, runtime, model_binding.lm, capabilities)
        return self._default_reasoning(intent, runtime, capabilities)

    def compile_with_dspy(self, signature: Optional[Any] = None, **predict_kwargs: Any) -> "Reasoner":
        """Attach a DSPy signature or program as this Reasoner's reasoning
        core. With no `signature`, compiles the package's default
        `ReasonAboutIntent` signature."""
        import dspy

        from .signatures import ReasonAboutIntent

        signature = signature or ReasonAboutIntent
        if isinstance(signature, type) and issubclass(signature, dspy.Signature):
            self.program = dspy.Predict(signature, **predict_kwargs)
        else:
            self.program = signature
        return self

    def optimize_with_gepa(self, trainset: list[Any], metric: Any, **gepa_kwargs: Any) -> "Reasoner":
        """Reflectively optimize this Reasoner's compiled DSPy program with
        GEPA -- this package's one sanctioned use of GEPA, kept on the
        single most judgment-laden stage rather than spread across every
        module. Compiles the default `ReasonAboutIntent` program first if
        none is attached yet."""
        import dspy

        if self.program is None:
            self.compile_with_dspy()
        optimizer = dspy.GEPA(metric=metric, **gepa_kwargs)
        self.program = optimizer.compile(self.program, trainset=trainset)
        return self

    def _run_program(self, intent: Intent, capabilities: str = "") -> Any:
        return self.program(intent=str(intent), context=intent.context, capabilities=capabilities)

    @staticmethod
    def _reason_with_llm(intent: Intent, runtime: Any, lm: Any, capabilities: str = "") -> str:
        import dspy

        from .signatures import ReasonAboutIntent

        runtime_name = getattr(runtime, "name", "Runtime")
        process_names = [process.name for process in getattr(runtime, "processes", [])]
        context = dict(intent.context)
        context["_runtime_name"] = runtime_name
        context["_available_processes"] = ", ".join(process_names) if process_names else "none"
        context["_remembered_context"] = Reasoner._memory_block(intent, runtime)
        strategy_narrative = Reasoner._strategy_block(runtime)
        if strategy_narrative:
            context["_operating_strategy"] = strategy_narrative

        reasoner = dspy.Predict(ReasonAboutIntent)
        with dspy.context(lm=lm):
            result = reasoner(
                intent=intent.text,
                context=context,
                capabilities=capabilities or "none",
            )
        return result.decision

    @staticmethod
    def _default_reasoning(intent: Intent, runtime: Any, capabilities: str = "") -> str:
        process_names = [process.name for process in getattr(runtime, "processes", [])]
        runtime_name = getattr(runtime, "name", "Runtime")
        processes = ", ".join(process_names) if process_names else "none"
        memory = getattr(runtime, "memory", None)
        memory_note = f", drawing on {len(memory)} remembered cycles" if memory and len(memory) else ""
        capability_note = ""
        if capabilities:
            names = [line.split(":", 1)[0].strip(" -") for line in capabilities.splitlines() if line.strip()]
            if names:
                capability_note = f", applying capabilities: {', '.join(names)}"
        return (
            f"[{runtime_name}] resolved intent '{intent.text}' across processes: "
            f"{processes}{capability_note}{memory_note}"
        )

    @staticmethod
    def _render_capabilities(plan: Any) -> str:
        """Flatten the scheduled plan (Workflows -> ordered Steps delegated to
        Personas -> stacked Skill prompts) into a natural-language block the
        reasoner can act on, in order. This is what makes the user's stacking
        matter: the narrated steps, the personas they delegate to and the
        stacked skill prompts are what the LLM reasons with and the order it
        works them in, rather than the bare intent. Returns "" when no plan
        is threaded through, so reasoning stays valid in that case."""
        if not plan:
            return ""
        lines: list[str] = []
        for workflow in plan:
            workflow_name = getattr(workflow, "name", "")
            if workflow_name:
                lines.append(f"Workflow {workflow_name}:")
            steps = getattr(workflow, "steps", [])
            for number, step in enumerate(steps, start=1):
                delegate = ""
                if step.persona is not None:
                    delegate = f" [delegated to Persona {step.persona.name}]"
                lines.append(f"  Step {number}: {step.instruction}{delegate}")
                Reasoner._render_persona(step.persona, lines, indent="      ")
            # Personas stacked directly on the workflow (no per-step narration).
            for persona in getattr(workflow, "personas", []):
                Reasoner._render_persona(persona, lines, indent="  ", header=True)
        return "\n".join(lines)

    @staticmethod
    def _render_persona(persona: Any, lines: list[str], indent: str, header: bool = False) -> None:
        if persona is None:
            return
        instructions = getattr(persona, "instructions", "")
        if header:
            line = f"{indent}Persona {persona.name}"
            if instructions:
                line += f": {instructions}"
            lines.append(line)
        elif instructions:
            lines.append(f"{indent}Persona {persona.name}: {instructions}")
        for skill in getattr(persona, "skills", []):
            instruction = skill.instruction() if hasattr(skill, "instruction") else getattr(skill, "name", "")
            lines.append(f"{indent}  - Skill {skill.name}: {instruction}")

    @staticmethod
    def _strategy_block(runtime: Any) -> str:
        """Render the operating strategy stacked in memory.md -- the
        ontology's vocabulary, the declared tools and MCP servers, and the
        discovery guidance -- so the model reasons with the enterprise's own
        terms and knows what capabilities it has."""
        strategy = getattr(runtime, "strategy", None)
        if strategy is None:
            return ""
        narrative = getattr(strategy, "narrative", None)
        return narrative() if callable(narrative) else ""

    @staticmethod
    def _memory_block(intent: Intent, runtime: Any) -> str:
        """Render Memory history, Experience and any relevant Adaptation
        insights for the prompt -- this is how persistent memory,
        aggregated experience and learned adaptations feed back into
        reasoning, kept as three distinct layers rather than one blob."""
        block = ""
        memory = getattr(runtime, "memory", None)
        if memory is not None and len(memory):
            block += f"\n\nMemory:\n{memory.context_window()}"
        experience = getattr(runtime, "experience", None)
        if experience is not None and experience.observations:
            block += f"\n\nExperience:\n{experience.summary()}"
        adaptations = getattr(runtime, "adaptations", None)
        if adaptations is not None:
            relevant = adaptations.relevant_to(intent.text)
            if relevant:
                insights = "\n".join(f"- {a.insight}" for a in relevant)
                block += f"\n\nLearned adaptations:\n{insights}"
        return block
