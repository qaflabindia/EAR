"""EAR -- Enterprise Agentic Runtime.

Engineering stack:   Prompt  -> Skill -> Persona -> Workflow -> Process -> Policy -> Runtime -> Reasoning
Philosophical stack: Sankalpa -> Vidya -> Guna   -> Varna    -> Karma   -> Dharma  -> Ksetra  -> Bhuddi

Manas (the LLM provider binding) is activated by Ksetra to power Bhuddi.

Ksetra also keeps four often-conflated memory concerns as four distinct
layers, each handed to the next rather than folded together:

    Pramana   -> evidence    (why a decision was made)
    Smriti    -> memory      (what happened)
    Anubhava  -> experience  (the pattern across repeated Smriti entries)
    Samskara  -> adaptation  (how future behaviour should change)
"""

from __future__ import annotations

from .anubhava import Anubhava
from .bhuddi import Bhuddi
from .dharma import Dharma
from .guna import Guna
from .karma import Karma
from .ksetra import Ksetra
from .manas import Manas
from .pramana import Pramana
from .samskara import Samskara, SamskaraBank
from .sankalpa import Sankalpa
from .smriti import Smriti, SmritiEntry
from .varna import Varna
from .vidya import Vidya

__version__ = "0.1.0"

__all__ = [
    "Sankalpa",
    "Vidya",
    "Guna",
    "Varna",
    "Karma",
    "Dharma",
    "Ksetra",
    "Manas",
    "Pramana",
    "Smriti",
    "SmritiEntry",
    "Anubhava",
    "Samskara",
    "SamskaraBank",
    "Bhuddi",
    "__version__",
]
