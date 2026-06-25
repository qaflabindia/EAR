"""Smarana -- remember: recall the Smriti context relevant to a cycle and
snapshot it, so what was actually remembered when a decision was made is
itself part of that decision's evidence trail."""

from __future__ import annotations

from dataclasses import dataclass

from .sankalpa import Sankalpa
from .smriti import Smriti


@dataclass
class Smarana:
    """Smarana recalls a Smriti's current context window."""

    def recall(self, smriti: Smriti, sankalpa: Sankalpa) -> str:
        return smriti.context_window()
