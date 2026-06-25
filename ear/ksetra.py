"""Ksetra -- the runtime: the field where processes are orchestrated, policies
are enforced, Manas (the LLM provider) is activated, reasoning (Bhuddi) is
started, and the cycle's evidence (Pramana), memory (Smṛti) and experience
(Anubhava) are all committed -- three layers AI systems often conflate into
one."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .anubhava import Anubhava
from .bhuddi import Bhuddi
from .dharma import Dharma
from .karma import Karma
from .manas import Manas
from .pramana import Pramana
from .samskara import SamskaraBank
from .sankalpa import Sankalpa
from .smriti import Smriti


@dataclass
class Ksetra:
    """A Ksetra is the runtime battlefield: processes orchestrated, policies
    enforced, Manas activated, reasoning started, and the outcome recorded
    across the Pramana (why) / Smriti (what) / Anubhava (pattern across
    repetition) / Samskara (adaptation) layers."""

    name: str
    processes: list[Karma] = field(default_factory=list)
    policies: list[Dharma] = field(default_factory=list)
    reasoner: Bhuddi = field(default_factory=Bhuddi)
    manas: Optional[Manas] = None
    smriti: Smriti = field(default_factory=Smriti)
    anubhava: Anubhava = field(default_factory=Anubhava)
    samskara: SamskaraBank = field(default_factory=SamskaraBank)

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
        if self.manas is not None:
            self.manas.activate()
        result = self.reasoner.reason(sankalpa, runtime=self)
        pramana = self._build_pramana(sankalpa)
        entry = self.smriti.record(sankalpa.text, result, context=sankalpa.context, evidence=pramana)
        self.anubhava.observe_entry(entry)
        return result

    def _build_pramana(self, sankalpa: Sankalpa) -> Pramana:
        """Capture why this decision was reached -- separately from what was
        decided (Smriti) or any pattern drawn from repeating it (Anubhava)."""
        if self.reasoner.program is not None:
            basis = "Resolved via a compiled DSPy program"
        elif self.manas is not None and self.manas.lm is not None:
            basis = f"Resolved via Manas LM '{self.manas.model_id}'"
        else:
            basis = "Resolved via Bhuddi's dependency-free default"
        return Pramana(
            basis=basis,
            sources={
                "policies_checked": [policy.name for policy in self.policies],
                "context": dict(sankalpa.context),
            },
        )
