"""Bhuddi -- reasoning: the discriminative intelligence the runtime starts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .sankalpa import Sankalpa


@dataclass
class Bhuddi:
    """Bhuddi is the reasoning layer a Ksetra runtime invokes once a Sankalpa
    has cleared every Dharma policy gate. Ksetra activates its Manas (LLM
    provider) first; Bhuddi then reasons with a compiled DSPy program if one
    is attached, or by calling the activated Manas LM directly. With no
    program and no Manas it falls back to a deterministic summary, so the
    runtime is usable with no LLM at all. Call `compile_with_dspy` to attach
    a real DSPy program."""

    program: Optional[Any] = None

    def reason(self, sankalpa: Sankalpa, runtime: Any = None) -> Any:
        if self.program is not None:
            return self.program(sankalpa=str(sankalpa), context=sankalpa.context)
        manas = getattr(runtime, "manas", None)
        if manas is not None and manas.lm is not None:
            return self._manas_reasoning(sankalpa, runtime, manas.lm)
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
    def _manas_reasoning(sankalpa: Sankalpa, runtime: Any, lm: Any) -> str:
        process_names = [process.name for process in getattr(runtime, "processes", [])]
        runtime_name = getattr(runtime, "name", "Ksetra")
        processes = ", ".join(process_names) if process_names else "none"
        prompt = (
            f"You are Bhuddi, the reasoning layer of the '{runtime_name}' agentic "
            f"runtime. Resolve this Sankalpa (intent) given the active processes: "
            f"{processes}.\n\nSankalpa: {sankalpa.text}\nContext: {sankalpa.context}"
        )
        completions = lm(prompt=prompt)
        return completions[0] if completions else ""

    @staticmethod
    def _default_reasoning(sankalpa: Sankalpa, runtime: Any) -> str:
        process_names = [process.name for process in getattr(runtime, "processes", [])]
        runtime_name = getattr(runtime, "name", "Ksetra")
        processes = ", ".join(process_names) if process_names else "none"
        return f"[{runtime_name}] resolved Sankalpa '{sankalpa.text}' across processes: {processes}"
