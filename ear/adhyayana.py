"""Adhyayana -- learn: fold a committed Smriti entry into Anubhava
experience. The step that turns one remembered cycle into part of a
pattern, distinct from Anukulana's job of turning that pattern into an
adaptation."""

from __future__ import annotations

from dataclasses import dataclass

from .anubhava import Anubhava
from .smriti import SmritiEntry


@dataclass
class Adhyayana:
    """Adhyayana learns from one cycle by observing its Smriti entry into
    Anubhava."""

    def learn(self, anubhava: Anubhava, entry: SmritiEntry) -> Anubhava:
        return anubhava.observe_entry(entry)
