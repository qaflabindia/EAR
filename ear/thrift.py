"""Thrift -- spend the smallest adequate model.

Most enterprise cycles are light work: extraction, routing, classification,
short summaries. Sending every one to the largest model burns compute,
dollars and energy for no better answer. `ModelThrift` holds a two-tier
ladder -- a light binding and a heavy one -- and routes each intent by
*judged* complexity: the **light** model itself judges whether the task
needs the heavy one (`JudgeTaskComplexity`), so the routing decision costs a
cheap call, never an expensive one. Escalation is honest by instruction:
when the judge is uncertain, it says heavy -- a wasted large call costs
money; a botched hard task costs more.

Offline (no light model bound), the fallback is deterministic and labeled:
route by the intent's sheer size (a long, context-heavy request goes heavy;
a short one goes light). A routing judgment nobody made is never written
down as one.

Every choice lands on the one audit spine (stage `thrift`), with the tier,
the basis, and the rationale -- so the efficiency win is measurable from the
trail (pair it with Pricing dollars and the `## Energy` rate to see exactly
what thrift saved).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .intent import Intent
from .llm import LMError

LIGHT = "light"
HEAVY = "heavy"

# The deterministic fallback's size threshold: an intent whose text plus
# rendered context exceeds this many words routes heavy. Mechanics, not
# judgment -- and only ever used when no model is bound to judge.
FALLBACK_WORD_THRESHOLD = 120


@dataclass
class ThriftChoice:
    """One routing decision: which tier, on what basis, and why."""

    tier: str
    binding: Any
    basis: str
    rationale: str


@dataclass
class ModelThrift:
    """A two-tier model ladder routed by judged complexity. `choose(intent)`
    returns the binding to reason with; `bind(runtime, intent)` applies it
    to the runtime for the coming cycle and records the choice."""

    light: Any
    heavy: Any

    def choose(self, intent: Intent, runtime: Any = None) -> ThriftChoice:
        choice = self._decide(intent)
        log = getattr(runtime, "reasoning_log", None)
        if log is not None:
            log.record(
                stage="thrift",
                inputs={"intent": intent.text, "context": dict(intent.context)},
                output=f"{choice.tier} model ({self._model_name(choice.binding)})",
                rationale=f"{choice.rationale} [{choice.basis}]",
            )
        return choice

    def bind(self, runtime: Any, intent: Intent) -> ThriftChoice:
        """Choose a tier for `intent` and set it as the runtime's model
        binding for the coming cycle -- the one-line way to make a whole
        stack thrift-routed per intent. When the chosen tier is not reachable
        (no resolvable credential and no local endpoint), the runtime is left
        on its deterministic fallback rather than pointed at an
        unconfigured endpoint -- the same posture the Loader takes."""
        choice = self.choose(intent, runtime=runtime)
        runtime.model_binding = choice.binding if self._reachable(choice.binding) else None
        return choice

    @staticmethod
    def _reachable(binding: Any) -> bool:
        if binding is None:
            return False
        resolve = getattr(binding, "resolve_api_key", None)
        if callable(resolve) and resolve() is not None:
            return True
        return bool(getattr(binding, "api_base", ""))

    # -- the two paths --------------------------------------------------------

    def _decide(self, intent: Intent) -> ThriftChoice:
        lm = self._light_lm()
        if lm is not None:
            try:
                return self._judge(lm, intent)
            except LMError:
                # A routing call that fails (a dead endpoint, a provider
                # error) degrades to the deterministic fallback -- routing
                # must never take the cycle down, and the fallback says it
                # was the model that was unavailable, not that none existed.
                return self._fallback(intent, unavailable=True)
        return self._fallback(intent)

    def _light_lm(self) -> Optional[Any]:
        """The light model's live `LM`, but only when it is actually
        reachable -- a resolvable credential or a local endpoint -- the same
        guard the Loader applies before attaching a binding. Absent, the
        thrift router degrades to its deterministic fallback rather than
        firing a doomed network call, so an unconfigured stack still routes."""
        if self.light is None:
            return None
        resolve = getattr(self.light, "resolve_api_key", None)
        has_key = callable(resolve) and resolve() is not None
        if not has_key and not getattr(self.light, "api_base", ""):
            return None
        activate = getattr(self.light, "activate", None)
        if callable(activate):
            activate()
        return getattr(self.light, "lm", None)

    def _judge(self, lm: Any, intent: Intent) -> ThriftChoice:
        from .signatures import JudgeTaskComplexity

        result = JudgeTaskComplexity.run(lm, intent_text=intent.text, context=dict(intent.context))
        heavy = bool(getattr(result, "heavy", True))
        rationale = str(getattr(result, "rationale", "") or "judged by the light model")
        if heavy:
            return ThriftChoice(HEAVY, self.heavy, "judged by the light model", rationale)
        return ThriftChoice(LIGHT, self.light, "judged by the light model", rationale)

    def _fallback(self, intent: Intent, unavailable: bool = False) -> ThriftChoice:
        basis = (
            "deterministic fallback (light model unavailable)"
            if unavailable
            else "deterministic fallback (no model to judge)"
        )
        words = len(intent.text.split()) + sum(
            len(str(key).split()) + len(str(value).split()) for key, value in intent.context.items()
        )
        if words > FALLBACK_WORD_THRESHOLD:
            return ThriftChoice(
                HEAVY,
                self.heavy,
                basis,
                f"{words} words of intent and context exceed the {FALLBACK_WORD_THRESHOLD}-word threshold",
            )
        return ThriftChoice(
            LIGHT,
            self.light,
            basis,
            f"{words} words of intent and context fit the light tier",
        )

    @staticmethod
    def _model_name(binding: Any) -> str:
        return str(getattr(binding, "model", "") or getattr(binding, "name", "") or "unbound")
