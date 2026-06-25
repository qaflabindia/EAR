"""DSPy backend -- turns a DSPy signature/program into a Bhuddi reasoner."""

from __future__ import annotations

from typing import Any

from ..bhuddi import Bhuddi


def make_reasoner(signature: Any, **predict_kwargs: Any) -> Bhuddi:
    """Build a Bhuddi backed by a DSPy `Signature` (wrapped in `dspy.Predict`)
    or by an already-built DSPy program/module."""
    return Bhuddi().compile_with_dspy(signature, **predict_kwargs)
