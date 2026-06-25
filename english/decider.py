"""Decider -- commit the Deliberator's deliberation to one final decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Decider:
    """A Decider finalizes a decision from the Deliberator's deliberation,
    rejecting an empty one rather than letting it silently become "the"
    decision."""

    def decide(self, deliberation: Any) -> Any:
        if deliberation is None:
            raise ValueError("Deliberator produced no deliberation to decide from")
        return deliberation
