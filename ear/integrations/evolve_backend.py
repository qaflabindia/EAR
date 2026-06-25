"""Evolve backend -- evolves a Vidya's source code with openevolve.

Requires the optional 'evolve' extra: `pip install ear[evolve]`.
"""

from __future__ import annotations

from typing import Any, Callable, Union

from ..vidya import Vidya


def evolve_skill(
    skill: Vidya,
    evaluator: Union[Callable[[str], dict], str],
    iterations: "int | None" = None,
    **run_kwargs: Any,
) -> Vidya:
    """Evolve `skill.source` (an OpenEvolve `EVOLVE-BLOCK-START`/`-END`
    program) against `evaluator`, then write the best candidate back onto
    the skill.

    `evaluator` is either a path to an OpenEvolve evaluator script or a
    callable that takes a program path and returns a metrics dict -- see
    `openevolve.run_evolution` for the full contract.
    """
    if not skill.source:
        raise ValueError(f"Vidya '{skill.name}' has no source to evolve")

    try:
        from openevolve import run_evolution
    except ImportError as exc:
        raise ImportError(
            "evolve_skill requires the optional 'evolve' dependency: pip install ear[evolve]"
        ) from exc

    result = run_evolution(
        initial_program=skill.source,
        evaluator=evaluator,
        iterations=iterations,
        **run_kwargs,
    )
    skill.source = result.best_code
    return skill
