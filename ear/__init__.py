"""EAR -- Enterprise Agentic Runtime.

Engineering stack:   Prompt  -> Skill -> Persona -> Workflow -> Process -> Policy -> Runtime -> Reasoning
Philosophical stack: Sankalpa -> Vidya -> Guna   -> Varna    -> Karma   -> Dharma  -> Ksetra  -> Bhuddi
"""

from __future__ import annotations

from .bhuddi import Bhuddi
from .dharma import Dharma
from .guna import Guna
from .karma import Karma
from .ksetra import Ksetra
from .sankalpa import Sankalpa
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
    "Bhuddi",
    "__version__",
]
