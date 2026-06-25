"""Parinama -- evolve: transform a Vidya skill's source code with
openevolve. A structural, dev-time operation -- evolving a skill's code
isn't part of running it, so this sits outside the per-cycle Ksetra
pipeline and is invoked directly when you want to evolve a skill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Union

from .vidya import Vidya


@dataclass
class Parinama:
    """Parinama evolves a Vidya skill against an evaluator, wrapping
    `integrations.evolve_backend.evolve_skill`."""

    def evolve(
        self,
        skill: Vidya,
        evaluator: Union[Callable[[str], dict], str],
        iterations: "int | None" = None,
        **run_kwargs: Any,
    ) -> Vidya:
        from .integrations.evolve_backend import evolve_skill

        return evolve_skill(skill, evaluator, iterations=iterations, **run_kwargs)
