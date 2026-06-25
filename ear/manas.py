"""Manas -- the mind: the LLM provider binding that Ksetra activates to
power Bhuddi reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Manas:
    """A Manas binds a provider and model (e.g. "openai" / "gpt-4o-mini")
    plus credentials and call parameters into a DSPy LM. A Ksetra activates
    its Manas before handing the Sankalpa to Bhuddi, so reasoning -- whether
    a compiled DSPy program or Bhuddi's raw fallback -- runs against it."""

    provider: str
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    lm: Optional[Any] = None

    @property
    def model_id(self) -> str:
        return self.model if "/" in self.model else f"{self.provider}/{self.model}"

    def activate(self) -> Any:
        """Build (once) and configure this Manas's `dspy.LM` as DSPy's
        active LM, then return it."""
        import dspy

        if self.lm is None:
            kwargs: dict[str, Any] = dict(self.params)
            if self.api_key is not None:
                kwargs.setdefault("api_key", self.api_key)
            if self.api_base is not None:
                kwargs.setdefault("api_base", self.api_base)
            self.lm = dspy.LM(self.model_id, **kwargs)
        dspy.configure(lm=self.lm)
        return self.lm
