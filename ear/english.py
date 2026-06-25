"""English-terminology aliases for every EAR class.

EAR's classes are named after the Sanskrit term for each role (see this
package's `__init__.py` docstring for the full stack and pipeline). This
module re-exports the identical classes under their English equivalents
so the package can be used under either vocabulary -- these are aliases,
not copies: `Runtime is Ksetra` holds, and an instance built via one name
is exactly an instance of the other.
"""

from __future__ import annotations

from .adhyayana import Adhyayana as Learner
from .anubhava import Anubhava as Experience
from .anukulana import Anukulana as Adapter
from .anushthana import Anushthana as Executor
from .anveshana import Anveshana as Discoverer
from .arambha import Arambha as Initializer
from .bhuddi import Bhuddi as Reasoner
from .dharma import Dharma as Policy
from .guna import Guna as Persona
from .karma import Karma as Process
from .kriya import Kriya as Performer
from .ksetra import Ksetra as Runtime
from .manas import Manas as ModelBinding
from .nirnaya import Nirnaya as Decider
from .niyamana import Niyamana as Governor
from .niyojana import Niyojana as Scheduler
from .parinama import Parinama as Evolver
from .parishodhana import Parishodhana as Auditor
from .pariksha import Pariksha as Validator
from .pramana import Pramana as Evidence
from .samanvaya import Samanvaya as Orchestrator
from .samskara import Samskara as Adaptation
from .samskara import SamskaraBank as AdaptationBank
from .samyojana import Samyojana as Composer
from .sankalpa import Sankalpa as Intent
from .smarana import Smarana as Recaller
from .smriti import Smriti as Memory
from .smriti import SmritiEntry as MemoryEntry
from .utkarsha import Utkarsha as Optimizer
from .varana import Varana as Selector
from .varna import Varna as Workflow
from .vicara import Vicara as Deliberator
from .vidya import Vidya as Skill
from .vyakhya import Vyakhya as Explainer

__all__ = [
    "Intent",
    "Skill",
    "Persona",
    "Workflow",
    "Process",
    "Policy",
    "Runtime",
    "ModelBinding",
    "Evidence",
    "Memory",
    "MemoryEntry",
    "Experience",
    "Adaptation",
    "AdaptationBank",
    "Reasoner",
    "Governor",
    "Initializer",
    "Discoverer",
    "Selector",
    "Composer",
    "Scheduler",
    "Orchestrator",
    "Executor",
    "Performer",
    "Deliberator",
    "Decider",
    "Validator",
    "Recaller",
    "Explainer",
    "Auditor",
    "Learner",
    "Adapter",
    "Evolver",
    "Optimizer",
]
