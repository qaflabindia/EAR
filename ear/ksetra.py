"""Ksetra -- the runtime: the field where a cycle runs through its full,
explicitly-named pipeline rather than one opaque `reason()` call --

    Niyamana (govern) -> Arambha (initialize) -> Anveshana (discover) ->
    Varana (select) -> Samyojana (compose) -> Niyojana (schedule) ->
    Samanvaya (orchestrate) -> [Anushthana (execute) -> Kriya (perform) ->
    Vicara (reason) -> Nirnaya (decide) -> Pariksha (validate)] ->
    Smarana (remember) -> Vyakhya (explain) -> Parishodhana (audit) ->
    Smriti (store memory) -> Adhyayana (learn) -> Anukulana (adapt)

so each operation that AI runtimes often blur together stays a separate,
inspectable, swappable step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .adhyayana import Adhyayana
from .anubhava import Anubhava
from .anukulana import Anukulana
from .anveshana import Anveshana
from .arambha import Arambha
from .bhuddi import Bhuddi
from .dharma import Dharma
from .karma import Karma
from .manas import Manas
from .niyamana import Niyamana
from .niyojana import Niyojana
from .parinama import Parinama
from .parishodhana import Parishodhana
from .pariksha import Pariksha
from .pramana import Pramana
from .samanvaya import Samanvaya
from .samskara import SamskaraBank
from .samyojana import Samyojana
from .sankalpa import Sankalpa
from .smarana import Smarana
from .smriti import Smriti
from .utkarsha import Utkarsha
from .varana import Varana
from .varna import Varna
from .vyakhya import Vyakhya


@dataclass
class Ksetra:
    """A Ksetra is the runtime battlefield: every cycle runs through the
    full Niyamana/Arambha/Anveshana/Varana/Samyojana/Niyojana/Samanvaya
    pipeline, and is recorded across the Pramana (why) / Smriti (what) /
    Anubhava (pattern) / Samskara (adaptation) layers."""

    name: str
    processes: list[Karma] = field(default_factory=list)
    policies: list[Dharma] = field(default_factory=list)
    reasoner: Bhuddi = field(default_factory=Bhuddi)
    manas: Optional[Manas] = None
    smriti: Smriti = field(default_factory=Smriti)
    anubhava: Anubhava = field(default_factory=Anubhava)
    samskara: SamskaraBank = field(default_factory=SamskaraBank)

    # Per-cycle pipeline stages.
    niyamana: Niyamana = field(default_factory=Niyamana)
    arambha: Arambha = field(default_factory=Arambha)
    anveshana: Anveshana = field(default_factory=Anveshana)
    varana: Varana = field(default_factory=Varana)
    samyojana: Samyojana = field(default_factory=Samyojana)
    niyojana: Niyojana = field(default_factory=Niyojana)
    pariksha: Pariksha = field(default_factory=Pariksha)
    samanvaya: Samanvaya = field(default_factory=Samanvaya)
    smarana: Smarana = field(default_factory=Smarana)
    vyakhya: Vyakhya = field(default_factory=Vyakhya)
    parishodhana: Parishodhana = field(default_factory=Parishodhana)
    adhyayana: Adhyayana = field(default_factory=Adhyayana)
    anukulana: Anukulana = field(default_factory=Anukulana)

    # Standalone, dev-time operations -- not part of the per-cycle pipeline.
    parinama: Parinama = field(default_factory=Parinama)
    utkarsha: Utkarsha = field(default_factory=Utkarsha)

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
        violations = self.niyamana.govern(self, sankalpa)
        if violations:
            names = ", ".join(policy.name for policy in violations)
            raise PermissionError(f"Dharma violated: {names}")

        self.arambha.initialize(self)

        candidates = self.pariksha.validate_candidates(self.anveshana.discover(self, sankalpa))
        selected = self.pariksha.validate_selection(self.varana.select(self, candidates))
        plan = self.pariksha.validate_plan(self.samyojana.compose(selected))
        scheduled = self.pariksha.validate_schedule(self.niyojana.schedule(plan))
        recalled = self.smarana.recall(self.smriti, sankalpa)

        decision = self.samanvaya.orchestrate(self, sankalpa)

        pramana = self._build_pramana(sankalpa, scheduled, recalled)
        pramana.sources["explanation"] = self.vyakhya.explain(pramana, decision)
        self.parishodhana.audit(pramana)

        entry = self.smriti.record(sankalpa.text, decision, context=sankalpa.context, evidence=pramana)
        self.adhyayana.learn(self.anubhava, entry)
        self.anukulana.adapt(self.samskara, self.anubhava)
        return decision

    def _build_pramana(self, sankalpa: Sankalpa, plan: list[Varna], recalled: str) -> Pramana:
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
                "plan": [workflow.name for workflow in plan],
                "recalled_memory": recalled,
            },
        )
