"""Adversary -- the ATC adversarial-deliberation hook.

On the intent path (framework architecture §5), a *flagged* intent takes an
adversarial pass before it executes: ATC (the Adversarial Testing command
centre) argues the strongest case against the action, argues the defense,
and returns a verdict -- uphold, escalate, or overturn. It is deliberation
*when triggered*, not on every cycle: most intents are not flagged, and an
unflagged intent is never delayed.

Two parts:

* `is_flagged(intent, registry=...)` -- the trigger. An intent is flagged
  when its context declares high stakes or an explicit adversarial request,
  or when its acting agent is on probation in the AECC registry (Phase 3's
  `EnvelopeRegistry`). The predicate reads plainly and returns *why* it
  flagged, so the reason rides the record.

* `AdversarialReview` -- the pass itself. With a model bound it runs the
  native `AdversarialChallenge` judgment (structured prompting, no framework
  underneath); offline it degrades to a deterministic, conservative
  fallback -- a flagged action with low confidence or high stakes escalates
  rather than being silently upheld. Either way the challenge, defense and
  verdict land on the runtime's one audit spine.

Standard library only. The review never *executes* anything: it returns a
verdict a caller (an Exchange, an operator, the governance plane) acts on,
the same division of labour as every other judgment in the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .intent import Intent
from .reasoning_log import calls_so_far, model_name, usage_since
from .section import normalize

# Verdicts an adversarial review returns.
UPHOLD = "uphold"
ESCALATE = "escalate"
OVERTURN = "overturn"
_VERDICTS = {UPHOLD, ESCALATE, OVERTURN}

# Context keys that ask for, or raise the stakes warranting, an adversarial
# pass -- read plainly, coerced to bool by the same codec as everything else.
_STAKES_KEYS = ("high_stakes", "adversarial", "adversarial_review", "irreversible", "flagged")
_CONFIDENCE_KEYS = ("confidence", "production_confidence", "certainty")
_ACTOR_KEYS = ("agent", "actor", "agent_id", "acting_agent")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    return normalize(str(value)) in {"true", "yes", "y", "1", "high", "critical"}


def _deterministic_floor(intent: Intent, registry: Optional[Any]) -> tuple[bool, str]:
    """The factual signals that flag an intent regardless of any judgment: an
    explicit high-stakes/irreversible/adversarial context value, or an acting
    agent AECC has placed on probation (authorized-but-watched). These are
    facts, not opinions, so they flag on their own."""
    context = intent.context
    for key in _STAKES_KEYS:
        if key in context and _truthy(context[key]):
            return True, f"context flags '{key}'"
    if registry is not None:
        for key in _ACTOR_KEYS:
            agent = context.get(key)
            if agent:
                envelope = registry.get(str(agent))
                if envelope is not None and normalize(envelope.status) == "probation":
                    return True, f"acting agent '{agent}' is on probation"
    return False, ""


def is_flagged(
    intent: Intent,
    registry: Optional[Any] = None,
    model_binding: Any = None,
    threshold: float = 0.6,
) -> tuple[bool, str]:
    """Whether `intent` warrants an adversarial pass, and why -- judged, with
    a deterministic fallback.

    A factual floor always flags (an explicit high-stakes/irreversible value,
    or a probationary acting agent). Otherwise the decision -- *does this
    intent warrant scrutiny?* -- is a judgment: with a model bound, the model
    reads the intent and decides (the `FlagForAdversarialReview` signature);
    offline, an unremarkable intent is simply not flagged, and says so. So
    flagging is model-judged when it can be, never silently hardcoded to
    keywords, and never fabricates a judgment it did not make."""
    floored, reason = _deterministic_floor(intent, registry)
    if floored:
        return True, reason
    lm = getattr(model_binding, "lm", None) if model_binding is not None else None
    if lm is not None:
        from .signatures import FlagForAdversarialReview

        result = FlagForAdversarialReview.run(lm, intent_text=intent.text, context=dict(intent.context))
        flag = bool(getattr(result, "flag", False))
        rationale = str(getattr(result, "reason", "") or ("model flagged for review" if flag else "model saw no need for review"))
        return flag, rationale
    return False, "no high-stakes signal and no model to judge -- not flagged"


@dataclass
class ReviewOutcome:
    """The result of an adversarial pass: the verdict, the two sides argued,
    and why it was reviewed. `passed` is True only on an outright uphold --
    an escalate or overturn stops the action pending a human or a rethink."""

    verdict: str
    concern: str = ""
    challenge: str = ""
    defense: str = ""
    rationale: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == UPHOLD

    @property
    def escalated(self) -> bool:
        return self.verdict == ESCALATE


@dataclass
class AdversarialReview:
    """ATC's adversarial pass over a flagged intent. LLM-judged when a model
    is bound (the `AdversarialChallenge` signature), deterministic and
    conservative offline."""

    threshold: float = 0.6

    def review(
        self,
        runtime: Any,
        intent: Intent,
        decision: Any = None,
        concern: str = "",
        model_binding: Any = None,
    ) -> ReviewOutcome:
        binding = model_binding if model_binding is not None else getattr(runtime, "model_binding", None)
        lm = getattr(binding, "lm", None) if binding is not None else None
        if binding is not None:
            binding.activate()
            lm = getattr(binding, "lm", None)

        start = calls_so_far(lm)
        if lm is not None:
            outcome = self._judge(lm, intent, decision, concern)
        else:
            outcome = self._fallback(intent, concern)

        log = getattr(runtime, "reasoning_log", None)
        if log is not None:
            log.record(
                stage="adversarial",
                inputs={
                    "intent": intent.text,
                    "context": dict(intent.context),
                    "concern": concern,
                },
                output=f"{outcome.verdict.upper()} -- {outcome.rationale}",
                rationale=f"challenge: {outcome.challenge} | defense: {outcome.defense}",
                model=model_name(binding),
                usage=usage_since(lm, start),
            )
        return outcome

    def review_flagged(
        self,
        runtime: Any,
        intent: Intent,
        decision: Any = None,
        registry: Optional[Any] = None,
        model_binding: Any = None,
    ) -> Optional[ReviewOutcome]:
        """Review `intent` only if it is flagged; otherwise return None so
        the caller proceeds without a pass. The hook the intent path calls.

        The flag decision itself is judged (with a deterministic fallback,
        see `is_flagged`) and recorded on the runtime's one audit spine -- so
        a decision *not* to review, when a model made it, is on the record
        too, never a silent skip."""
        binding = model_binding if model_binding is not None else getattr(runtime, "model_binding", None)
        flagged, concern = is_flagged(
            intent, registry=registry, model_binding=binding, threshold=self.threshold
        )
        log = getattr(runtime, "reasoning_log", None)
        if log is not None:
            log.record(
                stage="flag",
                inputs={"intent": intent.text, "context": dict(intent.context)},
                output="flagged for adversarial review" if flagged else "not flagged",
                rationale=concern,
            )
        if not flagged:
            return None
        return self.review(runtime, intent, decision=decision, concern=concern, model_binding=binding)

    # -- the two paths ------------------------------------------------------

    def _judge(self, lm: Any, intent: Intent, decision: Any, concern: str) -> ReviewOutcome:
        from .signatures import AdversarialChallenge

        result = AdversarialChallenge.run(
            lm,
            intent_text=intent.text,
            context=dict(intent.context),
            concern=concern or "flagged for adversarial review",
            decision=str(decision) if decision is not None else "none reached yet",
        )
        verdict = normalize(str(getattr(result, "verdict", "") or "")).split()
        verdict_word = verdict[0] if verdict else ""
        if verdict_word not in _VERDICTS:
            # An unparseable verdict is never guessed into an uphold -- an
            # adversarial pass that cannot conclude escalates to a human.
            verdict_word = ESCALATE
        return ReviewOutcome(
            verdict=verdict_word,
            concern=concern,
            challenge=str(getattr(result, "challenge", "") or ""),
            defense=str(getattr(result, "defense", "") or ""),
            rationale=str(getattr(result, "rationale", "") or "adversarial judgment"),
        )

    def _fallback(self, intent: Intent, concern: str) -> ReviewOutcome:
        """Offline: no adversary to argue, so decide conservatively from the
        facts. A flagged action escalates unless its own confidence is high;
        it is never silently upheld, because no judgment was actually made."""
        confidence = None
        for key in _CONFIDENCE_KEYS:
            if key in intent.context:
                try:
                    confidence = float(intent.context[key])
                except (TypeError, ValueError):
                    confidence = None
                break
        if confidence is not None and confidence >= self.threshold:
            return ReviewOutcome(
                verdict=UPHOLD,
                concern=concern,
                rationale=(
                    f"offline fallback: flagged action carries confidence {confidence} "
                    f">= {self.threshold}, upheld deterministically"
                ),
            )
        return ReviewOutcome(
            verdict=ESCALATE,
            concern=concern,
            rationale=(
                "offline fallback: flagged action lacks sufficient confidence to uphold "
                "without an adversarial judgment -- escalated to a human"
            ),
        )
