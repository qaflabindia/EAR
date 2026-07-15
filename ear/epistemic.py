"""Epistemic audit -- ARC: reasoning quality on the output side.

Where AKC governs what enters `Knowledge`, ARC audits how the runtime
*reasons* (framework architecture §6). It scans the reasoning log's own
deliberation and decision records for biased premises and unsupported
assumptions, records what it finds as **advisories** -- it informs, it does
not block -- and **escalates to AGCC** when a systematic pattern emerges (a
run of flags past a threshold), so a one-off is noted but a habit is raised.

Advisory by construction: an ARC finding rides the audit spine (stage
`epistemic`) exactly like ATC's adversarial pass rides it, and the
escalation is a record AGCC's gate can act on, not a silent veto. Reason-
first: with a model bound, the model judges each excerpt
(`JudgeReasoningQuality`); offline, ARC makes no judgment and says so -- it
never manufactures a bias finding it did not actually reason to.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .reasoning_log import model_name

# Which trail stages carry the runtime's substantive reasoning -- the output
# side ARC audits (governance and accounting stages are not reasoning).
_REASONING_STAGES = {"deliberation", "decision", "explanation", "reduce"}

ESCALATION_THRESHOLD = 3


@dataclass
class EpistemicFinding:
    """One audited excerpt: which cycle/stage it came from, whether it flags
    a biased premise or an unsupported assumption, and the concern named."""

    cycle: int
    stage: str
    biased: bool
    unsupported: bool
    rationale: str

    @property
    def flagged(self) -> bool:
        return self.biased or self.unsupported


@dataclass
class EpistemicAudit:
    """The result of an ARC pass: the findings, and whether a systematic
    pattern was escalated to AGCC."""

    findings: list[EpistemicFinding] = field(default_factory=list)
    escalated: bool = False

    @property
    def flags(self) -> int:
        return sum(1 for finding in self.findings if finding.flagged)

    def summary(self) -> str:
        return f"{len(self.findings)} excerpts audited, {self.flags} flagged" + (
            " -- escalated to AGCC" if self.escalated else ""
        )


@dataclass
class EpistemicAuditor:
    """ARC -- the epistemic-quality auditor. `audit(runtime)` scans the
    runtime's reasoning trail and records advisories; a run of flags past
    `escalate_threshold` is escalated to AGCC."""

    escalate_threshold: int = ESCALATION_THRESHOLD
    max_excerpts: int = 20

    def audit(self, runtime: Any, model_binding: Any = None) -> EpistemicAudit:
        log = getattr(runtime, "reasoning_log", None)
        if log is None:
            return EpistemicAudit()
        binding = model_binding if model_binding is not None else getattr(runtime, "model_binding", None)
        lm = getattr(binding, "lm", None) if binding is not None else None
        if binding is not None:
            binding.activate()
            lm = getattr(binding, "lm", None)

        excerpts = [
            record
            for record in getattr(log, "records", [])
            if record.stage in _REASONING_STAGES and str(record.output).strip()
        ][-self.max_excerpts :]

        if lm is None:
            # No model, no judgment: record the honest non-audit rather than a
            # clean bill of health nobody actually gave.
            log.record(
                stage="epistemic",
                inputs={"excerpts": len(excerpts)},
                output="not audited -- no model bound to judge reasoning quality",
                rationale="an epistemic clearance nobody reasoned to is never written down as one",
            )
            return EpistemicAudit()

        audit = EpistemicAudit()
        for record in excerpts:
            finding = self._judge(lm, record)
            audit.findings.append(finding)
            if finding.flagged:
                log.record(
                    stage="epistemic",
                    inputs={"cycle": record.cycle, "audited_stage": record.stage},
                    output="ADVISORY -- " + ("biased premise; " if finding.biased else "")
                    + ("unsupported assumption" if finding.unsupported else ""),
                    rationale=finding.rationale,
                    model=model_name(binding),
                )

        if audit.flags >= self.escalate_threshold:
            audit.escalated = True
            log.record(
                stage="epistemic",
                inputs={"flags": audit.flags, "threshold": self.escalate_threshold},
                output=f"ESCALATE to AGCC -- {audit.flags} epistemic flags is a systematic pattern",
                rationale="ARC escalates a habit of biased or unsupported reasoning to execution governance",
            )
        return audit

    def _judge(self, lm: Any, record: Any) -> EpistemicFinding:
        from .signatures import JudgeReasoningQuality

        result = JudgeReasoningQuality.run(lm, reasoning=str(record.output))
        return EpistemicFinding(
            cycle=getattr(record, "cycle", 0),
            stage=getattr(record, "stage", ""),
            biased=bool(getattr(result, "biased", False)),
            unsupported=bool(getattr(result, "unsupported", False)),
            rationale=str(getattr(result, "rationale", "") or "epistemic judgment"),
        )
