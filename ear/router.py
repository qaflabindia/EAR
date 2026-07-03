"""Router -- the omni-route agent: one provider-agnostic binding that
routes a reasoning call across a whole stack of providers.

A single `ModelBinding` speaks to one provider. Real deployments want the
OmniRoute pattern instead: many providers behind one endpoint, a routing
strategy that decides which to try first, automatic fallback to the next
when one errors or is rate-limited, and a circuit breaker that benches a
failing provider for a cooldown window so the next call skips it. That is
what a `Router` is.

Crucially, a Router is a *drop-in ModelBinding*: it exposes the same
`activate()`, `lm` and `model_id` surface the rest of the runtime already
duck-types against, so every judgment-laden stage (Governor, Discoverer,
Policy, Reasoner, Explainer) routes across all providers without any of
them knowing a router is there. Set `runtime.model_binding = router` and
the whole pipeline becomes provider-agnostic.

The providers themselves are ordinary `ModelBinding`s -- any LiteLLM
provider (Anthropic, OpenAI, Gemini, Bedrock, Groq, Ollama, ... 250+),
each reading its own key from the environment, never hardcoded -- so the
same code fans out across whatever providers a deployment has wired up.

Nothing here talks to a network on its own: the selection order, the
fallback walk and the cooldown breaker are plain, deterministic Python,
fully testable with fake per-provider callables and no LLM configured at
all -- the same two-tier testability the rest of the package keeps."""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .model_binding import ModelBinding


class RoutingStrategy(str, Enum):
    """How a Router orders its providers for a given call. Whatever the
    strategy, the Router still walks the ordered list and falls back to the
    next provider on failure -- the strategy only decides *who goes first*."""

    PRIORITY = "priority"          # lowest `priority` number first (ordered fallback)
    CHEAPEST = "cheapest"          # lowest `cost_per_1k` first
    FREE_FIRST = "free_first"      # free providers first, then by priority
    ROUND_ROBIN = "round_robin"    # rotate the starting provider each call
    WEIGHTED = "weighted"          # random order, biased by `weight`
    RANDOM = "random"              # uniform random order


class AllProvidersFailed(RuntimeError):
    """Raised when every provider in a Router's stack failed (or was on
    cooldown and then failed) for a single call. Carries the (model_id,
    error) pairs it collected along the way so the failure is inspectable
    rather than swallowed."""

    def __init__(self, attempts: list[tuple[str, BaseException]]) -> None:
        self.attempts = attempts
        tried = ", ".join(model_id for model_id, _ in attempts) or "no providers"
        super().__init__(f"All providers failed to serve the call (tried: {tried})")


@dataclass
class Router:
    """A Router routes a reasoning call across a stack of provider
    `ModelBinding`s. It is a drop-in ModelBinding: `activate()` builds a
    routing LM that, on every call, walks the providers in strategy order,
    skips any that a recent failure benched for `cooldown_seconds`, tries
    each in turn, and returns the first success -- tripping the circuit
    breaker on the ones that failed so the next call routes around them."""

    providers: list[ModelBinding] = field(default_factory=list)
    strategy: RoutingStrategy = RoutingStrategy.PRIORITY
    cooldown_seconds: float = 30.0
    max_failures: int = 1
    rng: random.Random = field(default_factory=random.Random)
    clock: Callable[[], float] = time.monotonic

    lm: Optional[Any] = None
    last_served: Optional[ModelBinding] = None

    _failures: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _available_at: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _rr_index: int = field(default=0, init=False, repr=False)

    # -- construction -------------------------------------------------------

    def add_provider(self, binding: ModelBinding) -> "Router":
        self.providers.append(binding)
        return self

    @classmethod
    def across(cls, *providers: ModelBinding, **kwargs: Any) -> "Router":
        """Build a Router across the given provider bindings."""
        return cls(providers=list(providers), **kwargs)

    @classmethod
    def from_env(
        cls,
        var: str = "EAR_ROUTER",
        strategy: RoutingStrategy = RoutingStrategy.PRIORITY,
        **kwargs: Any,
    ) -> "Router":
        """Build a Router from an environment variable -- the config lives
        in the environment, never hardcoded, matching the rest of the
        package. See `from_spec` for the accepted formats."""
        spec = os.environ.get(var)
        if not spec or not spec.strip():
            raise ValueError(f"No router spec found in environment variable {var!r}")
        return cls.from_spec(spec, strategy=strategy, **kwargs)

    @classmethod
    def from_spec(
        cls,
        spec: str,
        strategy: RoutingStrategy = RoutingStrategy.PRIORITY,
        **kwargs: Any,
    ) -> "Router":
        """Parse a router spec into a Router. Two formats are accepted:

        * A JSON array of provider objects, e.g.
          ``[{"provider": "anthropic", "model": "claude-opus-4-8",
             "priority": 10}, {"provider": "groq", "model": "llama-3.3-70b",
             "is_free": true}]``
        * A shorthand list of ``provider/model`` tokens separated by
          commas, semicolons or newlines, e.g.
          ``anthropic/claude-opus-4-8, openai/gpt-4o, groq/llama-3.3-70b``
          (priority follows list order)."""
        spec = spec.strip()
        if spec.startswith("["):
            entries = json.loads(spec)
            providers = [cls._binding_from_dict(entry, index) for index, entry in enumerate(entries)]
        else:
            tokens = [token.strip() for token in re.split(r"[,\n;]+", spec) if token.strip()]
            providers = []
            for index, token in enumerate(tokens):
                provider, _, model = token.partition("/")
                providers.append(ModelBinding(provider=provider, model=model or provider, priority=index))
        return cls(providers=providers, strategy=strategy, **kwargs)

    @staticmethod
    def _binding_from_dict(entry: dict[str, Any], index: int) -> ModelBinding:
        return ModelBinding(
            provider=entry["provider"],
            model=entry.get("model", entry["provider"]),
            api_key_env_var=entry.get("api_key_env_var"),
            api_base=entry.get("api_base"),
            params=entry.get("params", {}),
            priority=entry.get("priority", index),
            cost_per_1k=entry.get("cost_per_1k", entry.get("cost", 0.0)),
            weight=entry.get("weight", 1.0),
            is_free=entry.get("is_free", entry.get("free", False)),
            label=entry.get("label"),
        )

    # -- ModelBinding-compatible surface -----------------------------------

    @property
    def model_id(self) -> str:
        return f"omni-route:{self.strategy.value}({len(self.providers)} providers)"

    def activate(self) -> Any:
        """Build (once) this Router's routing LM and configure it as DSPy's
        active LM. From here the whole pipeline calls `self.lm`, which fans
        every call out across the providers with fallback and cooldown."""
        import dspy

        if self.lm is None:
            self.lm = _build_routing_lm(self)
        dspy.configure(lm=self.lm)
        return self.lm

    # -- routing core (LLM-free, fully testable) ---------------------------

    def order(self) -> list[ModelBinding]:
        """Return the providers to try for this call, in the order the
        current strategy prefers, with providers on cooldown filtered out.
        If every provider is on cooldown, they are all offered anyway --
        better to attempt a benched provider than to fail without trying."""
        candidates = [binding for binding in self.providers if self._available(binding)]
        if not candidates:
            candidates = list(self.providers)
        return self._order_by_strategy(candidates)

    def dispatch(self, caller: Callable[[Any], Any]) -> Any:
        """Walk the ordered providers and hand each one's built LM to
        `caller` until one returns without raising. Record a success on the
        server, trip the breaker on each failure, and raise
        `AllProvidersFailed` if the whole stack is exhausted. This is the
        one seam the routing LM (and tests) drive -- `caller` is the actual
        LLM call, or a fake in tests."""
        attempts: list[tuple[str, BaseException]] = []
        last_error: Optional[BaseException] = None
        for binding in self.order():
            lm = binding.build()
            try:
                result = caller(lm)
            except Exception as error:  # any provider error triggers fallback to the next
                self._record_failure(binding)
                attempts.append((binding.model_id, error))
                last_error = error
                continue
            self._record_success(binding)
            self.last_served = binding
            return result
        raise AllProvidersFailed(attempts) from last_error

    def reset(self) -> None:
        """Clear every provider's failure count and cooldown."""
        self._failures.clear()
        self._available_at.clear()

    # -- internals ----------------------------------------------------------

    def _order_by_strategy(self, candidates: list[ModelBinding]) -> list[ModelBinding]:
        strategy = self.strategy
        if strategy is RoutingStrategy.PRIORITY:
            return sorted(candidates, key=lambda binding: binding.priority)
        if strategy is RoutingStrategy.CHEAPEST:
            return sorted(candidates, key=lambda binding: (binding.cost_per_1k, binding.priority))
        if strategy is RoutingStrategy.FREE_FIRST:
            return sorted(candidates, key=lambda binding: (not binding.is_free, binding.priority, binding.cost_per_1k))
        if strategy is RoutingStrategy.ROUND_ROBIN:
            if not candidates:
                return []
            start = self._rr_index % len(candidates)
            self._rr_index += 1
            return candidates[start:] + candidates[:start]
        if strategy is RoutingStrategy.WEIGHTED:
            return self._weighted_order(candidates)
        if strategy is RoutingStrategy.RANDOM:
            shuffled = list(candidates)
            self.rng.shuffle(shuffled)
            return shuffled
        return list(candidates)

    def _weighted_order(self, candidates: list[ModelBinding]) -> list[ModelBinding]:
        # Efraimidis-Spirakis weighted sampling without replacement: draw a
        # key u ** (1 / weight) per provider and sort descending, so higher
        # `weight` tends to sort earlier while every provider still appears.
        def key(binding: ModelBinding) -> float:
            weight = binding.weight if binding.weight > 0 else 1e-9
            return self.rng.random() ** (1.0 / weight)

        return sorted(candidates, key=key, reverse=True)

    def _available(self, binding: ModelBinding) -> bool:
        return self.clock() >= self._available_at.get(binding.model_id, 0.0)

    def _record_failure(self, binding: ModelBinding) -> None:
        model_id = binding.model_id
        self._failures[model_id] = self._failures.get(model_id, 0) + 1
        if self._failures[model_id] >= self.max_failures:
            self._available_at[model_id] = self.clock() + self.cooldown_seconds
            self._failures[model_id] = 0

    def _record_success(self, binding: ModelBinding) -> None:
        model_id = binding.model_id
        self._failures.pop(model_id, None)
        self._available_at.pop(model_id, None)


def _build_routing_lm(router: Router) -> Any:
    """Construct a `dspy.LM` whose every call is routed through the Router's
    fallback-and-cooldown dispatch. Defined lazily so importing `ear` never
    imports dspy."""
    import dspy

    class RoutingLM(dspy.LM):
        """A dspy.LM facade over a Router: each call is dispatched across
        the Router's providers, returning the first provider's response."""

        def __init__(self, router: Router) -> None:
            super().__init__(model="ear-router/omni-route")
            self._router = router

        def __call__(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
            return self._router.dispatch(
                lambda lm: lm(prompt=prompt, messages=messages, **kwargs)
            )

        def forward(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
            return self._router.dispatch(
                lambda lm: lm.forward(prompt=prompt, messages=messages, **kwargs)
            )

    return RoutingLM(router)
