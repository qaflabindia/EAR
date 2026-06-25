"""Dharma -- the policy: governance mapped onto one or more processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ._safe_eval import MissingVariableError, safe_eval


@dataclass
class Dharma:
    """A Dharma is a policy: a governance rule that a Ksetra enforces before
    it lets a Karma process run."""

    name: str
    rule: str = ""
    check: Optional[Callable[..., bool]] = None

    def evaluate(self, **context: Any) -> bool:
        """Return True when the policy is satisfied (or not applicable)."""
        if self.check is not None:
            return bool(self.check(**context))
        if not self.rule:
            return True
        try:
            return bool(safe_eval(self.rule, context))
        except MissingVariableError:
            # The rule references a variable this Sankalpa's context doesn't
            # carry, so the policy doesn't apply to this particular intent.
            return True
