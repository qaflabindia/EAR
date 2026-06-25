"""EAR -- Enterprise Agentic Runtime.

Every class, field, method parameter and message here is named in plain
English. The package's judgment-laden pipeline stages (discovery, policy
enforcement, deliberation, explanation) reason in natural language against
a live LLM (via DSPy, see `ear/signatures.py`), each with a deterministic,
dependency-free fallback, so the package stays usable and testable with no
LLM configured at all.

Stack:   Intent -> Skill -> Persona -> Workflow -> Process -> Policy -> Runtime -> Reasoner

Runtime runs every cycle through a fully-named pipeline, each stage its
own class so operations AI runtimes often blur together stay distinct:

    Governor     -> govern       (enforce Policy gates; LLM-judged, with a safe-eval fallback)
    Initializer  -> initialize   (activate the ModelBinding)
    Discoverer   -> discover     (find relevant Processes; LLM-ranked, with a keyword fallback)
    Selector     -> select       (choose among discovered processes)
    Composer     -> compose      (assemble their Workflows into a plan)
    Scheduler    -> schedule     (order the composed plan)
    Orchestrator -> orchestrate  (coordinate execution of the plan)
    Executor     -> execute      (run the cycle's Performer action)
    Performer    -> perform      (deliberate, decide, validate)
    Deliberator  -> deliberate   (deliberate via the Reasoner)
    Decider      -> decide       (commit to one decision)
    Validator    -> validate     (reject a malformed decision)
    Recaller     -> remember     (recall Memory context as evidence)
    Explainer    -> explain      (render why a decision was reached; LLM-written, with an f-string fallback)
    Auditor      -> audit        (inspect evidence for compliance)
    Memory       -> store memory (what happened)
    Learner      -> learn        (fold the cycle into Experience)
    Adapter      -> adapt        (periodically distill a new Adaptation)

Evidence (why) and Experience (pattern aggregated from repeated Memory
entries) round out the memory layers Adaptation then adapts from.
Evolver (evolve) and Optimizer (optimize) are structural, dev-time
operations on Skill/Persona -- not part of the per-cycle pipeline.

DSPy and GEPA are used deliberately, not on every class: see
`ear/reasoner.py` and `ear/signatures.py` for where and why.
"""

from __future__ import annotations

from .adaptation import Adaptation, AdaptationBank
from .adapter import Adapter
from .auditor import Auditor
from .composer import Composer
from .decider import Decider
from .deliberator import Deliberator
from .discoverer import Discoverer
from .evidence import Evidence
from .evolver import Evolver
from .executor import Executor
from .experience import Experience
from .explainer import Explainer
from .governor import Governor
from .initializer import Initializer
from .intent import Intent
from .learner import Learner
from .memory import Memory, MemoryEntry
from .model_binding import ModelBinding
from .optimizer import Optimizer
from .orchestrator import Orchestrator
from .performer import Performer
from .persona import Persona
from .policy import Policy
from .process import Process
from .reasoner import Reasoner
from .recaller import Recaller
from .runtime import Runtime
from .scheduler import Scheduler
from .selector import Selector
from .skill import Skill
from .step import Step
from .validator import Validator
from .workflow import Workflow

__version__ = "0.1.0"

__all__ = [
    "Intent",
    "Skill",
    "Persona",
    "Step",
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
    "__version__",
]
