"""Optional EAR backends: dspy (core), openevolve and skillopt (extras).

Submodules import their third-party dependency lazily, so `import
ear.integrations` never fails even when the optional extras aren't
installed -- only calling a function that needs them does.
"""

from __future__ import annotations

__all__ = ["dspy_backend", "evolve_backend", "skillopt_backend"]
