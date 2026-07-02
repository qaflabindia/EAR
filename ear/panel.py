"""Panel -- multi-persona deliberation, native to the runtime.

A workflow authored with a `Pattern:` line in workflow.md convenes its
personas as a panel instead of reasoning single-voiced -- and the pattern
is prose, not an enum: `Pattern: adversarial debate, the risk officer has
the last word` goes into the prompt verbatim, so the deliberation *style*
is itself a natural-language instruction the model follows, never a
hardcoded protocol.

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
fake a debate: it reports deterministically who would have deliberated,
and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .intent import Intent
from .persona import Persona
from .reasoning_log import model_name


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

    def convene(self, runtime: Any, personas: list[Persona], intent: Intent, style: str = "") -> str:
        model_binding = getattr(runtime, "model_binding", None)
        log = getattr(runtime, "reasoning_log", None)
        live = model_binding is not None and getattr(model_binding, "lm", None) is not None

        transcript: list[Turn] = []
        budget = min(self.rounds * len(personas), self.max_turns)
        for turn_number in range(budget):
            persona = personas[turn_number % len(personas)]
            if live:
                statement = self._speak_with_llm(intent, persona, transcript, style, model_binding.lm)
            else:
                statement = (
                    f"(no model bound) {persona.name} would deliberate here as: "
                    f"{persona.instructions or 'no standing instructions'}"
                )
            transcript.append(Turn(speaker=persona.name, statement=statement))
            if log is not None:
                log.record(
                    stage="conversation",
                    inputs={
                        "speaker": persona.name,
                        "round": turn_number // len(personas) + 1,
                        "style": style,
                        "intent": intent.text,
                    },
                    output=statement,
                    model=model_name(model_binding),
                )

        if live:
            decision = self._synthesize_with_llm(intent, transcript, style, model_binding.lm)
        else:
            names = ", ".join(persona.name for persona in personas)
            decision = (
                f"Panel of {names} deliberated '{intent.text}'"
                + (f" in the style '{style}'" if style else "")
                + " -- no model bound, so no judgment was synthesized."
            )
        if log is not None:
            log.record(
                stage="deliberation",
                inputs={
                    "intent": intent.text,
                    "context": dict(intent.context),
                    "panel": [persona.name for persona in personas],
                    "style": style,
                    "transcript": self._render_transcript(transcript),
                },
                output=decision,
                model=model_name(model_binding),
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

    def _synthesize_with_llm(self, intent: Intent, transcript: list[Turn], style: str, lm: Any) -> str:
        from .signatures import SynthesizePanel

        result = SynthesizePanel.run(
            lm,
            intent_text=intent.text,
            pattern=style or "an open deliberation",
            transcript=self._render_transcript(transcript),
        )
        return str(result.decision).strip()