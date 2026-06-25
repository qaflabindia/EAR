"""Pariksha -- validate: check a Nirnaya decision is well-formed before it's
committed to Smriti memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Pariksha:
    """Pariksha rejects a blank decision rather than letting empty output
    pass silently into memory."""

    def validate(self, decision: Any) -> Any:
        if isinstance(decision, str) and not decision.strip():
            raise ValueError("Pariksha rejected an empty decision")
        return decision
