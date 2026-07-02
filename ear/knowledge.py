"""Knowledge -- the enterprise's reference material, declared in natural
language and consulted at reasoning time.

Sources are stacked in `memory.md` under a Knowledge section: one bullet
per source, `name: path-or-glob`, resolved relative to the stack directory
at load. Markdown sources are chunked into Passages by the same Section
parser the whole stack is authored with (one passage per section, plus the
preamble); plain-text sources are chunked by paragraph. A source that
matches no file fails loudly at load -- knowledge the author declared and
the runtime silently doesn't have is a governance hole, not a default.

Retrieval over this knowledge is the Librarian's judgment; Knowledge
itself only holds the passages and offers the structural candidate scoring
(word overlap, the same mechanics as the Discoverer's keyword fallback)
that narrows what the model is asked to judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .section import parse_document


@dataclass
class Passage:
    """One retrievable chunk of a knowledge source: where it came from,
    and its text verbatim."""

    source: str
    text: str

    def render(self) -> str:
        return f"[{self.source}]\n{self.text}"


@dataclass
class Knowledge:
    """The runtime's reference corpus: passages chunked from the declared
    sources, with structural candidate scoring for the Librarian."""

    passages: list[Passage] = field(default_factory=list)

    def add_document(self, source_name: str, filename: str, text: str) -> "Knowledge":
        """Chunk one document into passages. Markdown chunks by section
        through the shared parser; anything else chunks by paragraph."""
        if filename.endswith(".md"):
            document = parse_document(text)
            if document.preamble:
                self.passages.append(Passage(source=f"{source_name} -- {filename}", text=document.preamble))
            for section in document.sections:
                body = section.body()
                content = "\n".join(
                    filter(None, [body.prose] + [f"- {bullet}" for bullet in body.bullets] + body.numbered)
                )
                if content:
                    self.passages.append(
                        Passage(source=f"{source_name} -- {filename} § {section.name}", text=content)
                    )
        else:
            for paragraph in text.split("\n\n"):
                cleaned = " ".join(paragraph.split())
                if cleaned:
                    self.passages.append(Passage(source=f"{source_name} -- {filename}", text=cleaned))
        return self

    def candidates(self, query: str, limit: int) -> list[Passage]:
        """The structurally best-matching passages for a query, by word
        overlap -- the narrowing step before the model judges relevance,
        and the whole of retrieval when no model is bound."""
        words = {word.lower() for word in query.split() if len(word) > 3}
        if not words or not self.passages:
            return self.passages[:limit]
        scored = sorted(
            self.passages,
            key=lambda passage: sum(1 for word in words if word in passage.text.lower()),
            reverse=True,
        )
        return scored[:limit]

    def __len__(self) -> int:
        return len(self.passages)