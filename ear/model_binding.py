"""ModelBinding -- the LLM provider binding that the Runtime activates to
power every natural-language reasoning step (the Reasoner's decision,
Policy judgment, Discoverer relevance ranking, Explainer's explanation,
and Memory/Experience summarization).

A single ModelBinding names one provider and model. To route across many
providers -- OmniRoute-style, provider-agnostic, with fallback -- stack
several bindings into a `Router` (see `ear/router.py`), which is itself a
drop-in ModelBinding. The routing metadata a Router reads (`priority`,
`cost_per_1k`, `weight`, `is_free`) lives here on the binding, because
those are properties of the provider, not of the router."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelBinding:
    """A ModelBinding binds a provider and model (e.g. "anthropic" /
    "claude-opus-4-8") plus credentials and call parameters into a DSPy LM.
    Credentials are read from an environment variable -- never hardcoded
    and never written back to disk -- so the same code runs against
    whichever key the deployment environment provides.

    The `priority`, `cost_per_1k`, `weight` and `is_free` fields carry no
    weight on their own; they are the knobs a `Router` reads when it routes
    across a stack of bindings (ordered fallback, cheapest-first,
    free-first, weighted). `label` is an optional friendly name for
    dashboards and logs."""

    provider: str
    model: str
    api_key: Optional[str] = None
    api_key_env_var: Optional[str] = None
    api_base: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)

    # Routing metadata -- read by a Router, ignored when the binding is used
    # on its own.
    priority: int = 100
    cost_per_1k: float = 0.0
    weight: float = 1.0
    is_free: bool = False
    label: Optional[str] = None

    lm: Optional[Any] = None

    @property
    def model_id(self) -> str:
        return self.model if "/" in self.model else f"{self.provider}/{self.model}"

    def resolve_api_key(self) -> Optional[str]:
        if self.api_key is not None:
            return self.api_key
        env_var = self.api_key_env_var or f"{self.provider.upper()}_API_KEY"
        return os.environ.get(env_var)

    def build(self) -> Any:
        """Build (once) this ModelBinding's `dspy.LM` and return it, without
        making it DSPy's globally-active LM. A `Router` uses this to hold
        several providers' LMs side by side and pick between them per call;
        `activate` is the single-provider path that also configures DSPy."""
        if self.lm is None:
            import dspy

            kwargs: dict[str, Any] = dict(self.params)
            api_key = self.resolve_api_key()
            if api_key is not None:
                kwargs.setdefault("api_key", api_key)
            if self.api_base is not None:
                kwargs.setdefault("api_base", self.api_base)
            self.lm = dspy.LM(self.model_id, **kwargs)
        return self.lm

    def activate(self) -> Any:
        """Build this ModelBinding's `dspy.LM` and configure it as DSPy's
        active LM, then return it."""
        import dspy

        lm = self.build()
        dspy.configure(lm=lm)
        return lm
