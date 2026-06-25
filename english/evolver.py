"""Evolver -- transform a Skill's source code with openevolve. A
structural, dev-time operation -- evolving a skill's code isn't part of
running it, so this sits outside the per-cycle Runtime pipeline and is
invoked directly when you want to evolve a skill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Union

from .skill import Skill


@dataclass
class Evolver:
    """Evolver evolves a Skill against an evaluator, wrapping
    `integrations.evolve_backend.evolve_skill`."""

    def evolve(
        self,
        skill: Skill,
        evaluator: Union[Callable[[str], dict], str],
        iterations: "int | None" = None,
        **run_kwargs: Any,
    ) -> Skill:
        from .integrations.evolve_backend import evolve_skill

        return evolve_skill(skill, evaluator, iterations=iterations, **run_kwargs)
