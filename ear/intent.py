"""Intent -- the prompt: a resolved request that starts a reasoning cycle.

An Intent is markdown-native: `Intent.from_markdown` reads an intent
document -- the `#` title is the request, any prose elaborates it, and a
`## Context` section's bullets carry the facts (`- loan_amount: 18500`,
coerced back to numbers/booleans) -- and `to_markdown` renders one back.
So the runtime's input, like everything else in EAR, can be a plain
natural-language markdown file rather than constructed in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .section import coerce, normalize, parse_document


@dataclass
class Intent:
    """An Intent is a prompt: the request handed to the runtime."""

    text: str
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    @classmethod
    def from_markdown(cls, markdown: str) -> "Intent":
        """Read an Intent from an intent document: title and prose become
        the request text; `## Context` bullets become the context facts;
        any other section's prose elaborates the request."""
        document = parse_document(markdown)
        parts = [part for part in (document.title, document.preamble) if part]
        context: dict[str, Any] = {}
        for section in document.sections:
            body = section.body()
            if "context" in normalize(section.name):
                for bullet in body.bullets:
                    key, separator, value = bullet.partition(": ")
                    if not separator:
                        key, separator, value = bullet.partition(":")
                    if separator:
                        context[key.strip()] = coerce(value)
            elif body.prose:
                parts.append(body.prose)
        return cls(text="\n\n".join(parts).strip(), context=context)

    def to_markdown(self) -> str:
        """Render this Intent as an intent document."""
        first, _, rest = self.text.partition("\n")
        lines = [f"# {first.strip()}"]
        if rest.strip():
            lines += ["", rest.strip()]
        if self.context:
            lines += ["", "## Context", ""]
            lines += [f"- {key}: {value}" for key, value in self.context.items()]
        return "\n".join(lines) + "\n"