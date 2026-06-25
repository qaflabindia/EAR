"""Anushthana -- execute: run the cycle's Kriya action. The seam between
Samanvaya's coordination and Kriya's actual performed action."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .kriya import Kriya
from .sankalpa import Sankalpa


@dataclass
class Anushthana:
    """Anushthana executes a cycle by handing it to Kriya."""

    kriya: Kriya = field(default_factory=Kriya)

    def execute(self, runtime: Any, sankalpa: Sankalpa) -> Any:
        return self.kriya.perform(runtime, sankalpa)
