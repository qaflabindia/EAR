"""Vyakhya -- explain: render a human-readable explanation of why a
decision was reached, from its Pramana evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pramana import Pramana


@dataclass
class Vyakhya:
    """Vyakhya explains a decision by pairing its Pramana basis with the
    decision itself."""

    def explain(self, pramana: Pramana, decision: Any) -> str:
        return f"{pramana.basis} -> {decision}"
