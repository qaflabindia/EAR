"""ModelBinding -- the LLM provider binding that the Runtime activates to
power every natural-language reasoning step (the Reasoner's decision,
Policy judgment, Discoverer relevance ranking, Explainer's explanation,
and Memory/Experience summarization)."""

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
    whichever key the deployment environment provides."""

    provider: str
    model: str
    api_key: Optional[str] = None
    api_key_env_var: Optional[str] = None
    api_base: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    lm: Optional[Any] = None

    @property
    def model_id(self) -> str:
        return self.model if "/" in self.model else f"{self.provider}/{self.model}"

    def resolve_api_key(self) -> Optional[str]:
        if self.api_key is not None:
            return self.api_key
        env_var = self.api_key_env_var or f"{self.provider.upper()}_API_KEY"
        return os.environ.get(env_var)

    def activate(self) -> Any:
        """Build (once) and configure this ModelBinding's `dspy.LM` as
        DSPy's active LM, then return it."""
        import dspy

        if self.lm is None:
            kwargs: dict[str, Any] = dict(self.params)
            api_key = self.resolve_api_key()
            if api_key is not None:
                kwargs.setdefault("api_key", api_key)
            if self.api_base is not None:
                kwargs.setdefault("api_base", self.api_base)
            self.lm = dspy.LM(self.model_id, **kwargs)
        dspy.configure(lm=self.lm)
        return self.lm
