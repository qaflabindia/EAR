"""Utkarsha -- optimize: refine a Guna persona's skill document with
Microsoft SkillOpt's ReflACT loop. A structural, dev-time operation, kept
outside the per-cycle Ksetra pipeline for the same reason Parinama is."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

from .guna import Guna


@dataclass
class Utkarsha:
    """Utkarsha builds a SkillOpt trainer and applies its trained document
    back onto a Guna persona's skill. SkillOpt has no generic one-call
    train-and-apply API, so this mirrors that rather than inventing one:
    call `optimize` to get a trainer, run `.train()` yourself, then `apply`
    the result."""

    def optimize(self, config: Union[str, dict], adapter: Any) -> Any:
        from .integrations.skillopt_backend import build_trainer

        return build_trainer(config, adapter)

    def apply(self, persona: Guna, skill_name: str, trained_document: str) -> Guna:
        from .integrations.skillopt_backend import apply_trained_skill

        return apply_trained_skill(persona, skill_name, trained_document)
