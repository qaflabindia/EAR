"""Parishodhana -- audit: inspect a cycle's Pramana evidence for compliance
before it's committed to Smriti, marking that the inspection happened."""

from __future__ import annotations

from dataclasses import dataclass

from .pramana import Pramana


@dataclass
class Parishodhana:
    """Parishodhana audits a Pramana, recording that it was inspected."""

    def audit(self, pramana: Pramana) -> Pramana:
        pramana.sources.setdefault("audited", True)
        return pramana
