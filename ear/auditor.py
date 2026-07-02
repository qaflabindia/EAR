"""Auditor -- inspect a cycle's Evidence for compliance before it's
committed to Memory.

Inspection is a runtime judgment: when a ModelBinding is active, the LLM
reads the decision against its evidentiary basis -- the policies checked,
the plan, the recalled memory -- the way an internal auditor would, and
states whether the evidence supports the decision, naming any gap. The
assessment is written into the Evidence itself (`audit_assessment`) and to
the ReasoningLog. With no model, the inspection falls back to marking that
the audit point was passed.

The `audited` flag is set by code either way: whether an inspection
happened is a control fact, not a judgment, so the model writes the
assessment but never decides whether the audit step ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evidence import Evidence
from .reasoning_log import calls_so_far, model_name, usage_since


@dataclass
class Auditor:
    """An Auditor audits a piece of Evidence: an LLM-written assessment of
    whether the evidence supports the decision when a model is active,
    with the audited flag always recorded by code."""

    def audit(self, evidence: Evidence, runtime: Any = None, decision: Any = None) -> Evidence:
        model_binding = getattr(runtime, "model_binding", None)
        if model_binding is not None and model_binding.lm is not None:
            start = calls_so_far(model_binding.lm)
            assessment = self._assess_with_llm(evidence, decision, model_binding.lm)
            evidence.sources["audit_assessment"] = assessment
            log = getattr(runtime, "reasoning_log", None)
            if log is not None:
                log.record(
                    stage="audit",
                    inputs={"basis": evidence.basis, "decision": str(decision)},
                    output=assessment,
                    model=model_name(model_binding),
                    usage=usage_since(model_binding.lm, start),
                )
        evidence.sources.setdefault("audited", True)
        return evidence

    @staticmethod
    def _assess_with_llm(evidence: Evidence, decision: Any, lm: Any) -> str:
        from .signatures import AuditEvidence

        summary_lines = [f"Basis: {evidence.basis}"]
        for key in ("policies_checked", "plan", "recalled_memory"):
            value = evidence.sources.get(key)
            if value:
                summary_lines.append(f"{key}: {value}")
        result = AuditEvidence.run(lm, decision=str(decision), evidence="\n".join(summary_lines))
        return str(result.assessment).strip()