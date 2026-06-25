"""EAR -- Enterprise Agentic Runtime.

Engineering stack:   Prompt  -> Skill -> Persona -> Workflow -> Process -> Policy -> Runtime -> Reasoning
Philosophical stack: Sankalpa -> Vidya -> Guna   -> Varna    -> Karma   -> Dharma  -> Ksetra  -> Bhuddi

Manas (the LLM provider binding) is activated by Ksetra to power Bhuddi.

Ksetra runs every cycle through a fully-named pipeline, each stage its own
class so operations AI runtimes often blur together stay distinct:

    Niyamana   -> govern       (enforce Dharma policy gates)
    Arambha    -> initialize   (activate Manas)
    Anveshana  -> discover     (find relevant Karma processes)
    Varana     -> select       (choose among discovered processes)
    Samyojana  -> compose      (assemble their Varna workflows into a plan)
    Niyojana   -> schedule     (order the composed plan)
    Samanvaya  -> orchestrate  (coordinate execution of the plan)
    Anushthana -> execute      (run the cycle's Kriya action)
    Kriya      -> perform      (deliberate, decide, validate)
    Vicara     -> reason       (deliberate via Bhuddi)
    Nirnaya    -> decide       (commit to one decision)
    Pariksha   -> validate     (reject a malformed decision)
    Smarana    -> remember     (recall Smriti context as evidence)
    Vyakhya    -> explain      (render why a decision was reached)
    Parishodhana -> audit      (inspect evidence for compliance)
    Smriti     -> store memory (what happened)
    Adhyayana  -> learn        (fold the cycle into Anubhava experience)
    Anukulana  -> adapt        (periodically distill a new Samskara)

Pramana (evidence) and Anubhava (experience aggregated from repeated
Smriti entries) round out the memory layers Samskara then adapts from.
Parinama (evolve) and Utkarsha (optimize) are structural, dev-time
operations on Vidya/Guna -- not part of the per-cycle pipeline.
"""

from __future__ import annotations

from .adhyayana import Adhyayana
from .anubhava import Anubhava
from .anukulana import Anukulana
from .anushthana import Anushthana
from .anveshana import Anveshana
from .arambha import Arambha
from .bhuddi import Bhuddi
from .dharma import Dharma
from .guna import Guna
from .karma import Karma
from .kriya import Kriya
from .ksetra import Ksetra
from .manas import Manas
from .nirnaya import Nirnaya
from .niyamana import Niyamana
from .niyojana import Niyojana
from .parinama import Parinama
from .parishodhana import Parishodhana
from .pariksha import Pariksha
from .pramana import Pramana
from .samanvaya import Samanvaya
from .samskara import Samskara, SamskaraBank
from .samyojana import Samyojana
from .sankalpa import Sankalpa
from .smarana import Smarana
from .smriti import Smriti, SmritiEntry
from .utkarsha import Utkarsha
from .varana import Varana
from .varna import Varna
from .vicara import Vicara
from .vidya import Vidya
from .vyakhya import Vyakhya

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
    "Niyamana",
    "Arambha",
    "Anveshana",
    "Varana",
    "Samyojana",
    "Niyojana",
    "Samanvaya",
    "Anushthana",
    "Kriya",
    "Vicara",
    "Nirnaya",
    "Pariksha",
    "Smarana",
    "Vyakhya",
    "Parishodhana",
    "Adhyayana",
    "Anukulana",
    "Parinama",
    "Utkarsha",
    "__version__",
]
