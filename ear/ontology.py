"""Ontology -- the working vocabulary the runtime reasons with.

Ontological settings are stacked in `memory.md`: one bullet per term,
`term: what it means in this enterprise`. The runtime folds the rendered
ontology into the Reasoner's prompt, so every judgment-laden stage reads the
enterprise's own definitions -- what a "risk grade" is, what counts as a
"decision" -- instead of guessing from general usage. Definitions are plain
English, never code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Ontology:
    """A small dictionary of term -> meaning, plus any free-form notes,
    rendered into the reasoning prompt as the runtime's vocabulary."""

    terms: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def define(self, term: str, meaning: str) -> "Ontology":
        self.terms[term.strip()] = meaning.strip()
        return self

    def meaning_of(self, term: str) -> str:
        return self.terms.get(term.strip(), "")

    def render(self) -> str:
        if not self.terms and not self.notes:
            return ""
        lines = ["Ontology -- the working vocabulary this runtime reasons with:"]
        lines += [f"- {term}: {meaning}" for term, meaning in self.terms.items()]
        if self.notes:
            lines.append(self.notes)
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.terms)
