"""SkillOpt backend -- trains a Persona's skill document with Microsoft
SkillOpt's ReflACT loop (rollout -> reflect -> aggregate -> select ->
update -> evaluate).

Requires the optional 'skillopt' extra: `pip install ear[skillopt]`.

SkillOpt optimizes a skill document against an environment adapter you
supply (it ships adapters for benchmarks like ALFWorld and DocVQA, and
documents how to write a custom `skillopt.envs.base.EnvAdapter` for your
own enterprise environment). This package does not invent a generic
one-call API on top of this -- it just gets a configured trainer into
your hands.
"""

from __future__ import annotations

from typing import Any, Union

from ..persona import Persona


def build_trainer(config: Union[str, dict], adapter: Any) -> Any:
    """Build a `skillopt.engine.trainer.ReflACTTrainer` for the given config
    (a path to a YAML config, or an already-loaded flat config dict) and
    environment adapter. Call `.train()` on the result to run the loop."""
    try:
        from skillopt.config import flatten_config, load_config
        from skillopt.engine import ReflACTTrainer
    except ImportError as exc:
        raise ImportError(
            "build_trainer requires the optional 'skillopt' dependency: "
            "pip install ear[skillopt]"
        ) from exc

    cfg = flatten_config(load_config(config)) if isinstance(config, str) else config
    return ReflACTTrainer(cfg, adapter)


def apply_trained_skill(persona: Persona, skill_name: str, trained_document: str) -> Persona:
    """Write a SkillOpt-trained skill document back onto the matching
    Skill's description, so the persona picks up the optimized
    instructions."""
    skill = persona.get_skill(skill_name)
    if skill is None:
        raise KeyError(f"Persona '{persona.name}' has no skill named '{skill_name}'")
    skill.description = trained_document
    return persona
