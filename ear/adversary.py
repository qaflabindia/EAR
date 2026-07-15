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


def is_flagged(
    intent: Intent,
    registry: Optional[Any] = None,
    threshold: float = 0.6,
) -> tuple[bool, str]:
    """Whether `intent` warrants an adversarial pass, and why. Flags on an
    explicit high-stakes/adversarial context value, on a low-confidence
    high-impact action, or on an acting agent that AECC has placed on
    probation. Returns (False, "") when nothing triggers."""
    context = intent.context
    for key in _STAKES_KEYS:
        if key in context and _truthy(context[key]):
            return True, f"context flags '{key}'"
    # A probationary agent's actions are flagged for review (the AECC
    # standing that authorizes-but-watches, see authority.py).
    if registry is not None:
        for key in _ACTOR_KEYS:
            agent = context.get(key)
            if agent:
                envelope = registry.get(str(agent))
                if envelope is not None and normalize(envelope.status) == "probation":
                    return True, f"acting agent '{agent}' is on probation"
    return False, ""


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
        the caller proceeds without a pass. The hook the intent path calls."""
        flagged, concern = is_flagged(intent, registry=registry, threshold=self.threshold)
        if not flagged:
            return None
        return self.review(runtime, intent, decision=decision, concern=concern, model_binding=model_binding)

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
