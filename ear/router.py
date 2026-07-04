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
Policy, Reasoner, Explainer, ...) routes across all providers without any
of them knowing a router is there. Set `runtime.model_binding = router` and
the whole pipeline becomes provider-agnostic. It builds no framework of its
own: once a provider's turn comes, `Router` just calls that provider's own
dependency-free `LM.complete(prompt, system=...)` (see `ear/llm.py`) --
routing is a seam in front of EAR's native LLM client, not a replacement
for it.

The providers themselves are ordinary `ModelBinding`s -- any provider
`ear/llm.py` speaks to (Anthropic natively, or anything OpenAI-compatible
via `api_base`: OpenAI, Azure, Ollama, Together, vLLM, ... 250+), each
reading its own key from the environment, never hardcoded -- so the same
code fans out across whatever providers a deployment has wired up. Routing
metadata (`priority`, `cost_per_1k`, `weight`, `is_free`) lives on a small
`RouterProvider` wrapper here, not on the shared `ModelBinding` class
itself, since a binding's priority is a property of *this router*, not of
the provider.

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
class RouterProvider:
    """One provider in a Router's stack: a `ModelBinding` plus the routing
    metadata a Router reads to decide when to try it. The metadata lives
    here, not on `ModelBinding` itself, because priority/cost/weight are
    properties of *this router*, not of the provider binding."""

    binding: ModelBinding
    priority: int = 100
    cost_per_1k: float = 0.0
    weight: float = 1.0
    is_free: bool = False
    label: str = ""

    @property
    def model_id(self) -> str:
        return self.binding.model_id


@dataclass
class Router:
    """A Router routes a reasoning call across a stack of provider
    `ModelBinding`s. It is a drop-in ModelBinding: `activate()` builds a
    routing `LM` facade that, on every call, walks the providers in
    strategy order, skips any that a recent failure benched for
    `cooldown_seconds`, tries each in turn via that provider's own native
    `LM.complete()`, and returns the first success -- tripping the circuit
    breaker on the ones that failed so the next call routes around them."""

    providers: list[RouterProvider] = field(default_factory=list)
    strategy: RoutingStrategy = RoutingStrategy.PRIORITY
    cooldown_seconds: float = 30.0
    max_failures: int = 1
    rng: random.Random = field(default_factory=random.Random)
    clock: Callable[[], float] = time.monotonic

    lm: Optional[Any] = None
    last_served: Optional[RouterProvider] = None

    _failures: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _available_at: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _rr_index: int = field(default=0, init=False, repr=False)

    # -- construction -------------------------------------------------------

    def add_provider(
        self,
        binding: ModelBinding,
        priority: int = 100,
        cost_per_1k: float = 0.0,
        weight: float = 1.0,
        is_free: bool = False,
        label: str = "",
    ) -> "Router":
        self.providers.append(
            RouterProvider(
                binding=binding, priority=priority, cost_per_1k=cost_per_1k, weight=weight, is_free=is_free, label=label
            )
        )
        return self

    @classmethod
    def across(cls, *bindings: ModelBinding, **kwargs: Any) -> "Router":
        """Build a Router across the given provider bindings, prioritized
        in the order given (first = tried first under the default PRIORITY
        strategy)."""
        providers = [RouterProvider(binding=binding, priority=index) for index, binding in enumerate(bindings)]
        return cls(providers=providers, **kwargs)

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
            providers = [cls._provider_from_dict(entry, index) for index, entry in enumerate(entries)]
        else:
            tokens = [token.strip() for token in re.split(r"[,\n;]+", spec) if token.strip()]
            providers = []
            for index, token in enumerate(tokens):
                provider, _, model = token.partition("/")
                binding = ModelBinding(provider=provider, model=model or provider)
                providers.append(RouterProvider(binding=binding, priority=index))
        return cls(providers=providers, strategy=strategy, **kwargs)

    @staticmethod
    def _provider_from_dict(entry: dict[str, Any], index: int) -> RouterProvider:
        binding = ModelBinding(
            provider=entry["provider"],
            model=entry.get("model", entry["provider"]),
            api_key_env_var=entry.get("api_key_env_var"),
            api_base=entry.get("api_base"),
            params=entry.get("params", {}),
        )
        return RouterProvider(
            binding=binding,
            priority=entry.get("priority", index),
            cost_per_1k=entry.get("cost_per_1k", entry.get("cost", 0.0)),
            weight=entry.get("weight", 1.0),
            is_free=entry.get("is_free", entry.get("free", False)),
            label=entry.get("label", ""),
        )

    # -- ModelBinding-compatible surface -----------------------------------

    @property
    def model_id(self) -> str:
        return f"omni-route:{self.strategy.value}({len(self.providers)} providers)"

    def activate(self) -> Any:
        """Build (once) this Router's routing `LM` facade -- from here the
        rest of the runtime calls `self.lm.complete(...)`, which fans every
        call out across the providers with fallback and cooldown."""
        if self.lm is None:
            self.lm = _RoutingLM(router=self)
        return self.lm

    # -- routing core (LLM-free, fully testable) ---------------------------

    def order(self) -> list[RouterProvider]:
        """Return the providers to try for this call, in the order the
        current strategy prefers, with providers on cooldown filtered out.
        If every provider is on cooldown, they are all offered anyway --
        better to attempt a benched provider than to fail without trying."""
        candidates = [provider for provider in self.providers if self._available(provider)]
        if not candidates:
            candidates = list(self.providers)
        return self._order_by_strategy(candidates)

    def dispatch(self, caller: Callable[[Any], Any]) -> Any:
        """Walk the ordered providers and hand each one's activated `LM` to
        `caller` until one returns without raising. Record a success on the
        server, trip the breaker on each failure, and raise
        `AllProvidersFailed` if the whole stack is exhausted. This is the
        one seam the routing LM (and tests) drive -- `caller` is the actual
        LLM call, or a fake in tests."""
        attempts: list[tuple[str, BaseException]] = []
        last_error: Optional[BaseException] = None
        for provider in self.order():
            lm = provider.binding.activate()
            try:
                result = caller(lm)
            except Exception as error:  # any provider error triggers fallback to the next
                self._record_failure(provider)
                attempts.append((provider.model_id, error))
                last_error = error
                continue
            self._record_success(provider)
            self.last_served = provider
            return result
        raise AllProvidersFailed(attempts) from last_error

    def reset(self) -> None:
        """Clear every provider's failure count and cooldown."""
        self._failures.clear()
        self._available_at.clear()

    # -- internals ----------------------------------------------------------

    def _order_by_strategy(self, candidates: list[RouterProvider]) -> list[RouterProvider]:
        strategy = self.strategy
        if strategy is RoutingStrategy.PRIORITY:
            return sorted(candidates, key=lambda provider: provider.priority)
        if strategy is RoutingStrategy.CHEAPEST:
            return sorted(candidates, key=lambda provider: (provider.cost_per_1k, provider.priority))
        if strategy is RoutingStrategy.FREE_FIRST:
            return sorted(candidates, key=lambda provider: (not provider.is_free, provider.priority, provider.cost_per_1k))
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

    def _weighted_order(self, candidates: list[RouterProvider]) -> list[RouterProvider]:
        # Efraimidis-Spirakis weighted sampling without replacement: draw a
        # key u ** (1 / weight) per provider and sort descending, so higher
        # `weight` tends to sort earlier while every provider still appears.
        def key(provider: RouterProvider) -> float:
            weight = provider.weight if provider.weight > 0 else 1e-9
            return self.rng.random() ** (1.0 / weight)

        return sorted(candidates, key=key, reverse=True)

    def _available(self, provider: RouterProvider) -> bool:
        return self.clock() >= self._available_at.get(provider.model_id, 0.0)

    def _record_failure(self, provider: RouterProvider) -> None:
        model_id = provider.model_id
        self._failures[model_id] = self._failures.get(model_id, 0) + 1
        if self._failures[model_id] >= self.max_failures:
            self._available_at[model_id] = self.clock() + self.cooldown_seconds
            self._failures[model_id] = 0

    def _record_success(self, provider: RouterProvider) -> None:
        model_id = provider.model_id
        self._failures.pop(model_id, None)
        self._available_at.pop(model_id, None)


@dataclass
class _RoutingLM:
    """The `LM`-shaped facade a Router installs as `model_binding.lm`.
    Every `.complete()` call is dispatched across the Router's providers
    with fallback and cooldown; a successful call's usage entry is copied
    from the serving provider's own `LM.history` into this facade's
    `history`, so the rest of the package's usage accounting
    (`calls_so_far`/`usage_since` in `ear/reasoning_log.py`) keeps working
    transparently -- a Router is a real drop-in, not a special case other
    modules need to know about."""

    router: Router
    history: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, prompt: str, system: str = "") -> str:
        def caller(lm: Any) -> str:
            before = len(lm.history)
            text = lm.complete(prompt, system=system)
            self.history.extend(lm.history[before:])
            return text

        return self.router.dispatch(caller)
