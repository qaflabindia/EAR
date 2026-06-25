"""Bhuddi -- reasoning: the discriminative intelligence the runtime starts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .sankalpa import Sankalpa


@dataclass
class Bhuddi:
    """Bhuddi is the reasoning layer a Ksetra runtime invokes once a Sankalpa
    has cleared every Dharma policy gate. With no program attached it falls
    back to a deterministic summary so the runtime is usable without an LLM;
    call `compile_with_dspy` to back it with a real DSPy program."""

    program: Optional[Any] = None

    def reason(self, sankalpa: Sankalpa, runtime: Any = None) -> Any:
        if self.program is not None:
            return self.program(sankalpa=str(sankalpa), context=sankalpa.context)
        return self._default_reasoning(sankalpa, runtime)

    def compile_with_dspy(self, signature: Any, **predict_kwargs: Any) -> "Bhuddi":
        """Attach a DSPy signature or program as this Bhuddi's reasoning core."""
        import dspy  # local import keeps dspy an opt-in runtime dependency

        if isinstance(signature, type) and issubclass(signature, dspy.Signature):
            self.program = dspy.Predict(signature, **predict_kwargs)
        else:
            self.program = signature
        return self

    @staticmethod
    def _default_reasoning(sankalpa: Sankalpa, runtime: Any) -> str:
        process_names = [process.name for process in getattr(runtime, "processes", [])]
        runtime_name = getattr(runtime, "name", "Ksetra")
        processes = ", ".join(process_names) if process_names else "none"
        return f"[{runtime_name}] resolved Sankalpa '{sankalpa.text}' across processes: {processes}"
