"""EAR -- Enterprise Agentic Runtime.

Engineering stack:   Prompt  -> Skill -> Persona -> Workflow -> Process -> Policy -> Runtime -> Reasoning
Philosophical stack: Sankalpa -> Vidya -> Guna   -> Varna    -> Karma   -> Dharma  -> Ksetra  -> Bhuddi

Manas (the LLM provider binding) is activated by Ksetra to power Bhuddi.
Smriti (persistent memory) and Samskara (learned adaptations distilled
from it) are recorded and consulted by Ksetra around every reasoning cycle.
"""

from __future__ import annotations

from .bhuddi import Bhuddi
from .dharma import Dharma
from .guna import Guna
from .karma import Karma
from .ksetra import Ksetra
from .manas import Manas
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
    "Smriti",
    "SmritiEntry",
    "Samskara",
    "SamskaraBank",
    "Bhuddi",
    "__version__",
]
