"""Panel -- multi-persona deliberation, native to the runtime.

A workflow authored with a `Pattern:` line in workflow.md convenes its
personas as a panel instead of reasoning single-voiced -- and the pattern
is prose, not an enum: `Pattern: adversarial debate, the risk officer has
the last word` goes into the prompt verbatim, so the deliberation *style*
is itself a natural-language instruction the model follows, never a
hardcoded protocol.

Who speaks next is a judgment, not a rotation: each turn, a moderator
judgment reads the pattern and the transcript and chooses the next
speaker -- or concludes the panel early when it has genuinely converged,
so consensus ends the deliberation before the budget does. Code guards
what the model may not decide: only listed personas speak (an unreadable
choice falls back to rotation, on the record), conclusion is honored only
once every persona has spoken at least once, and the turn budget still
caps the whole conversation regardless. Personas whose skills carry bound
tools may use the native tool loop inside their turns -- get the facts,
then speak -- with every invocation a `tool` record exactly as in
deliberation.

Each turn, one persona speaks -- its standing instructions and stacked
skills in hand, the transcript so far in view -- and a synthesis concludes
the panel into the single decision the pipeline continues with. Everything
around the conversation stays the runtime's: the Governor gated the cycle
before the panel sat, the Decider/Validator check the synthesis, Contracts
still judge the deliverable, and every turn lands on the trail (stage
`conversation`) with the synthesis as the cycle's `deliberation` record.

Budgets are code-enforced: `rounds` sets how many times the table goes
around and `max_turns` caps the total regardless -- a conversation is a
judgment, its cost is a control. With no model bound the panel does not
fake a debate: it rotates deterministically, reports who would have
deliberated, and says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .intent import Intent
from .persona import Persona
from .reasoning_log import calls_so_far, model_name, usage_since
from .section import normalize

# How many tool calls one panel turn may make before the persona must
# speak with what it has -- execution mechanics, not judgment.
TURN_TOOL_BUDGET = 3

_CONCLUDE_WORDS = {"conclude", "concluded", "consensus", "done", "end", "finish", "finished"}


@dataclass
class Turn:
    """One panel turn: who spoke, and what they said."""

    speaker: str
    statement: str


@dataclass
class Panel:
    """A Panel convenes personas over an intent and synthesizes their
    deliberation into one decision, on the record."""

    rounds: int = 2
    max_turns: int = 12

    def convene(
        self, runtime: Any, personas: list[Persona], intent: Intent, style: str = "", plan: Any = None
    ) -> str:
        model_binding = getattr(runtime, "model_binding", None)
        log = getattr(runtime, "reasoning_log", None)
        lm = getattr(model_binding, "lm", None)
        live = lm is not None
        binder = getattr(runtime, "tool_binder", None)
        tools = binder.bound_tools(runtime, plan) if (live and binder is not None) else []

        transcript: list[Turn] = []
        budget = min(self.rounds * len(personas), self.max_turns)
        spoken: set[str] = set()
        last_index = -1
        conclusion = ""
        while len(transcript) < budget:
            start = calls_so_far(lm)
            persona = personas[(last_index + 1) % len(personas)]
            chosen_by, choice_rationale = "rotation", ""
            if live:
                choice, rationale = self._choose_speaker_with_llm(intent, personas, transcript, style, lm)
                key = normalize(choice)
                if key in _CONCLUDE_WORDS:
                    if spoken >= {p.name for p in personas}:
                        conclusion = rationale
                        break
                    # A panel cannot converge before everyone has spoken --
                    # rotation continues, and the record says why.
                    chosen_by = "rotation (conclusion refused: not every persona has spoken yet)"
                else:
                    named = next((p for p in personas if normalize(p.name) == key), None)
                    if named is not None:
                        persona, chosen_by, choice_rationale = named, "model", rationale
                    else:
                        chosen_by = f"rotation (the choice '{choice}' names no listed persona)"
            if live and tools:
                statement = self._speak_with_tools(runtime, intent, persona, transcript, style, lm, tools)
            elif live:
                statement = self._speak_with_llm(intent, persona, transcript, style, lm)
            else:
                statement = (
                    f"(no model bound) {persona.name} would deliberate here as: "
                    f"{persona.instructions or 'no standing instructions'}"
                )
            last_index = personas.index(persona)
            spoken.add(persona.name)
            transcript.append(Turn(speaker=persona.name, statement=statement))
            if log is not None:
                log.record(
                    stage="conversation",
                    inputs={
                        "speaker": persona.name,
                        "turn": len(transcript),
                        "style": style,
                        "intent": intent.text,
                        "chosen_by": chosen_by,
                        "choice_rationale": choice_rationale,
                    },
                    output=statement,
                    model=model_name(model_binding),
                    usage=usage_since(lm, start),
                )

        synthesis_start = calls_so_far(lm)
        if live:
            decision = self._synthesize_with_llm(intent, transcript, style, lm)
        else:
            names = ", ".join(persona.name for persona in personas)
            decision = (
                f"Panel of {names} deliberated '{intent.text}'"
                + (f" in the style '{style}'" if style else "")
                + " -- no model bound, so no judgment was synthesized."
            )
        if log is not None:
            inputs = {
                "intent": intent.text,
                "context": dict(intent.context),
                "panel": [persona.name for persona in personas],
                "style": style,
                "transcript": self._render_transcript(transcript),
            }
            if conclusion:
                inputs["concluded_early"] = conclusion
            log.record(
                stage="deliberation",
                inputs=inputs,
                output=decision,
                model=model_name(model_binding),
                usage=usage_since(lm, synthesis_start),
            )
        return decision

    @staticmethod
    def _render_persona(persona: Persona) -> str:
        lines = [f"{persona.name}: {persona.instructions or 'no standing instructions'}"]
        lines += [f"  - Skill {skill.name}: {skill.instruction()}" for skill in persona.skills]
        return "\n".join(lines)

    @staticmethod
    def _render_transcript(transcript: list[Turn]) -> str:
        return "\n\n".join(f"[{turn.speaker}]\n{turn.statement}" for turn in transcript) or "no turns yet"

    def _choose_speaker_with_llm(
        self, intent: Intent, personas: list[Persona], transcript: list[Turn], style: str, lm: Any
    ) -> tuple[str, str]:
        from .signatures import ChooseNextSpeaker

        table = "\n".join(
            f"{persona.name}: {persona.instructions or 'no standing instructions'}" for persona in personas
        )
        result = ChooseNextSpeaker.run(
            lm,
            intent_text=intent.text,
            pattern=style or "an open deliberation",
            personas=table,
            transcript=self._render_transcript(transcript),
        )
        return str(result.speaker).strip(), str(result.rationale).strip()

    def _speak_with_llm(self, intent: Intent, persona: Persona, transcript: list[Turn], style: str, lm: Any) -> str:
        from .signatures import SpeakInPanel

        result = SpeakInPanel.run(
            lm,
            intent_text=intent.text,
            persona=self._render_persona(persona),
            pattern=style or "an open deliberation",
            transcript=self._render_transcript(transcript),
        )
        return str(result.statement).strip()

    def _speak_with_tools(
        self,
        runtime: Any,
        intent: Intent,
        persona: Persona,
        transcript: list[Turn],
        style: str,
        lm: Any,
        tools: list,
    ) -> str:
        """One turn with the native tool loop inside it: the persona may
        call the bound tools for facts -- each invocation a `tool` record
        through the binder, exactly as in deliberation -- then speaks.
        Budget spent or no statement given: the persona speaks plainly
        with the gathered facts in view."""
        from .signatures import SpeakOrUseTool
        from .tool_binder import ToolBinder

        binder = runtime.tool_binder
        by_key = {ToolBinder.tool_key(tool.name): tool for tool in tools}
        catalogue = "\n".join(f"{tool.name}({', '.join(tool.parameters)}): {tool.description}" for tool in tools)
        gathered: list[str] = []
        for _ in range(TURN_TOOL_BUDGET):
            action = SpeakOrUseTool.run(
                lm,
                intent_text=intent.text,
                persona=self._render_persona(persona),
                pattern=style or "an open deliberation",
                transcript=self._render_transcript(transcript),
                tools=catalogue,
                gathered="\n".join(gathered) or "none yet",
            )
            chosen = by_key.get(ToolBinder.tool_key(str(action.tool)))
            if chosen is None:
                statement = str(action.statement).strip()
                if statement:
                    return statement
                break  # neither a tool nor a statement -- speak plainly below
            arguments = ToolBinder.parse_arguments(action.arguments)
            result = binder.logged_handler(runtime, chosen)(**arguments)
            from .reasoner import Reasoner

            fed_back = Reasoner._compress_tool_result(runtime, chosen.name, arguments, result)
            gathered.append(f"{chosen.name}({arguments}) -> {fed_back}")
        rendered = self._render_transcript(transcript)
        if gathered:
            rendered += f"\n\n[facts {persona.name} gathered with tools]\n" + "\n".join(gathered)
        from .signatures import SpeakInPanel

        result = SpeakInPanel.run(
            lm,
            intent_text=intent.text,
            persona=self._render_persona(persona),
            pattern=style or "an open deliberation",
            transcript=rendered,
        )
        return str(result.statement).strip()

    def _synthesize_with_llm(self, intent: Intent, transcript: list[Turn], style: str, lm: Any) -> str:
        from .signatures import SynthesizePanel

        result = SynthesizePanel.run(
            lm,
            intent_text=intent.text,
            pattern=style or "an open deliberation",
            transcript=self._render_transcript(transcript),
        )
        return str(result.decision).strip()
