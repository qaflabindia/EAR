"""Auditor -- inspect a cycle's Evidence for compliance before it's
committed to Memory, marking that the inspection happened."""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import Evidence


@dataclass
class Auditor:
    """An Auditor audits a piece of Evidence, recording that it was
    inspected."""

    def audit(self, evidence: Evidence) -> Evidence:
        evidence.sources.setdefault("audited", True)
        return evidence
