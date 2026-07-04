"""SkillSelector -- stack only the skills relevant to the intent, instead
of every skill a persona carries.

This is the Discoverer pattern applied one level down: Discoverer ranks
whole Processes for relevance to an Intent; SkillSelector ranks a Persona's
Skills the same way, so a persona can carry a large library of skills
without every prompt being stacked into every reasoning call. Relevance is
judged natively (see `ear/judgment.py`, `RankRelevantSkills` in
`ear/signatures.py`) when a model is active and falls back to keyword
overlap offline, exactly like Discoverer -- and it short-circuits (returns
every skill, in order) whenever a persona has no more than `top_k` skills,
so the common case costs nothing and behaviour is unchanged for small
personas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .intent import Intent
from .section import normalize
from .skill import Skill


@dataclass
class SkillSelector:
    """A SkillSelector selects the `top_k` skills of a persona most relevant
    to an intent. With no LLM it ranks by keyword overlap; with one it ranks
    by native judgment. Returns every skill unchanged when the persona has
    `top_k` or fewer."""

    top_k: int = 8

    def select(self, persona: Any, intent: Intent, lm: Optional[Any] = None) -> list[Skill]:
        skills = list(getattr(persona, "skills", []))
        if len(skills) <= self.top_k:
            return skills
        if lm is not None:
            return self._rank_with_llm(skills, intent, lm)
        return self._rank_by_keyword(skills, intent)

    def _rank_by_keyword(self, skills: list[Skill], intent: Intent) -> list[Skill]:
        words = {word.lower() for word in intent.text.split() if len(word) > 3}

        def overlap(skill: Skill) -> int:
            haystack = f"{skill.name} {skill.instruction()}".lower()
            return sum(1 for word in words if word in haystack)

        # Stable sort keeps original order among equally-scoring skills, so a
        # persona with no keyword overlap still yields its first `top_k`.
        ranked = sorted(skills, key=overlap, reverse=True)
        return ranked[: self.top_k]

    def _rank_with_llm(self, skills: list[Skill], intent: Intent, lm: Any) -> list[Skill]:
        from .signatures import RankRelevantSkills

        catalogue = "\n".join(f"{skill.name}: {skill.instruction()}" for skill in skills)
        result = RankRelevantSkills.run(lm, intent_text=intent.text, available_skills=catalogue)
        by_name = {normalize(skill.name): skill for skill in skills}
        found = [by_name[normalize(str(name))] for name in result.relevant_skill_names if normalize(str(name)) in by_name]
        return (found or skills)[: self.top_k]
