"""Learner -- fold a committed Memory entry into Experience. The step that
turns one remembered cycle into part of a pattern, distinct from Adapter's
job of turning that pattern into an adaptation."""

from __future__ import annotations

from dataclasses import dataclass

from .experience import Experience
from .memory import MemoryEntry


@dataclass
class Learner:
    """A Learner learns from one cycle by observing its Memory entry into
    Experience."""

    def learn(self, experience: Experience, entry: MemoryEntry) -> Experience:
        return experience.observe_entry(entry)
