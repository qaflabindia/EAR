"""ModelBinding -- the LLM provider binding the Runtime activates to power
every natural-language reasoning step (the Reasoner's decision, Policy
judgment, Discoverer relevance, Selection, Scheduling, Delegation, Recall,
Audit, Explanation, Panel deliberation, and Memory/Experience
summarisation).

It builds EAR's own dependency-free `LM` (see `ear/llm.py`) -- no DSPy, no
LiteLLM, no provider SDK. Credentials are read from an environment variable,
never hardcoded and never written back to disk, so the same code runs
against whichever key the deployment environment provides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .llm import LM


@dataclass
class ModelBinding:
    """A ModelBinding binds a provider and model (e.g. "anthropic" /
    "claude-opus-4-8") plus credentials and call parameters into an `LM`."""

    provider: str
    model: str
    api_key: Optional[str] = field(default=None, repr=False)  # a credential -- never shown by repr/str
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
        """Build (once) this ModelBinding's `LM` and return it. Idempotent
        and cheap -- no network call happens until a judgment stage asks
        the model to reason."""
        if self.lm is None:
            self.lm = LM(
                model=self.model_id,
                provider=self.provider,
                api_key=self.resolve_api_key(),
                api_base=self.api_base,
                params=dict(self.params),
            )
        return self.lm
