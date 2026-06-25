"""Samanvaya -- orchestrate: coordinate a cycle's execution end to end.
The top of the Samanvaya -> Anushthana -> Kriya -> {Vicara, Nirnaya,
Pariksha} chain that Ksetra delegates the back half of a cycle to."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .anushthana import Anushthana
from .sankalpa import Sankalpa


@dataclass
class Samanvaya:
    """Samanvaya orchestrates a cycle by handing it to Anushthana."""

    anushthana: Anushthana = field(default_factory=Anushthana)

    def orchestrate(self, runtime: Any, sankalpa: Sankalpa) -> Any:
        return self.anushthana.execute(runtime, sankalpa)
