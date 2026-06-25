"""Ksetra -- the runtime: the field where processes are orchestrated, policies
are enforced, and reasoning (Bhuddi) is finally started."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bhuddi import Bhuddi
from .dharma import Dharma
from .karma import Karma
from .sankalpa import Sankalpa


@dataclass
class Ksetra:
    """A Ksetra is the runtime battlefield: processes orchestrated, policies
    enforced, reasoning started."""

    name: str
    processes: list[Karma] = field(default_factory=list)
    policies: list[Dharma] = field(default_factory=list)
    reasoner: Bhuddi = field(default_factory=Bhuddi)

    def add_process(self, process: Karma) -> "Ksetra":
        self.processes.append(process)
        return self

    def add_policy(self, policy: Dharma) -> "Ksetra":
        self.policies.append(policy)
        return self

    def enforce_policies(self, **context: Any) -> list[Dharma]:
        """Return the policies that are violated by the given context."""
        return [policy for policy in self.policies if not policy.evaluate(**context)]

    def reason(self, sankalpa: Sankalpa) -> Any:
        violations = self.enforce_policies(**sankalpa.context)
        if violations:
            names = ", ".join(policy.name for policy in violations)
            raise PermissionError(f"Dharma violated: {names}")
        return self.reasoner.reason(sankalpa, runtime=self)
