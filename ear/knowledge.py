"""Knowledge -- the enterprise's reference material, declared in natural
language and consulted at reasoning time.

Sources are stacked in `memory.md` under a Knowledge section: one bullet
per source, `name: path-or-glob` for files in the stack, or `name: URL`
for documents fetched over EAR's own HTTPS client and cached under
`.ear/knowledge/` (the refresh cadence -- "refetch weekly" -- is declared
in the same bullet's prose). Markdown sources are chunked into Passages by
the same Section parser the whole stack is authored with (one passage per
section, plus the preamble); plain-text sources are chunked by paragraph.
A source that matches no file fails loudly at load -- knowledge the author
declared and the runtime silently doesn't have is a governance hole, not a
default.

Retrieval over this knowledge is the Librarian's judgment; Knowledge
itself holds the passages and offers the structural narrowing that decides
what the model is asked to judge. Narrowing is BM25 -- inverse document
frequency, term-frequency saturation and length normalization, in pure
Python -- scored over each passage's text *and* its gist: a one-line,
model-written summary in everyday words, built once per corpus and
persisted to `.ear/index.md` keyed by content hash, so a question phrased
in synonyms still finds the passage whose jargon never uses them. With no
model bound the gists are simply absent and BM25 over the raw text alone
stands -- and `narrowing()` says which of the two is in force, so the
retrieval record never overstates what happened.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from .section import normalize, parse_document

# BM25's two standard constants: k1 sets how quickly repeated terms stop
# adding score (saturation), b how strongly long passages are penalized
# (length normalization). Retrieval mechanics, not judgment.
BM25_K1 = 1.5
BM25_B = 0.75

# How much of the SHA-256 hex digest keys an index entry. Twelve hex
# characters (48 bits) is far beyond collision reach for any real corpus
# while keeping the index readable.
FINGERPRINT_LENGTH = 12


def content_hash(text: str) -> str:
    """A short, stable fingerprint of a passage's text (stdlib SHA-256).
    The gist index is keyed by it, so editing a source silently retires
    the stale gist: the edited passage hashes differently, misses the
    index, and is re-gisted on the next indexed load."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


def words_of(text: str) -> list[str]:
    """Lowercased alphanumeric words: every non-alphanumeric character is
    a boundary. Plain string mechanics -- no patterns, no stemming; the
    vocabulary bridging that stemming approximates is the gist's job."""
    return "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()


@dataclass
class KnowledgeSource:
    """One declared source from memory.md's Knowledge section: a file
    pattern resolved against the stack directory, or a URL fetched over
    the native client with an optional refresh cadence in days."""

    name: str
    pattern: str = ""
    url: str = ""
    refresh_days: Optional[float] = None


@dataclass
class Passage:
    """One retrievable chunk of a knowledge source: where it came from,
    its text verbatim, and -- once the corpus is indexed -- the one-line
    model-written gist that narrowing also scores against."""

    source: str
    text: str
    gist: str = ""

    def render(self) -> str:
        return f"[{self.source}]\n{self.text}"

    @property
    def fingerprint(self) -> str:
        return content_hash(self.text)

    def searchable(self) -> str:
        """What narrowing scores: the passage's own words, plus the gist's
        everyday synonyms when the index has one."""
        return f"{self.text}\n{self.gist}" if self.gist else self.text


@dataclass
class Knowledge:
    """The runtime's reference corpus: passages chunked from the declared
    sources, BM25 narrowing for the Librarian, and the persisted gist
    index that lets differently-phrased questions find them."""

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

    # -- narrowing (BM25) ------------------------------------------------------

    def candidates(self, query: str, limit: int) -> list[Passage]:
        """The structurally best-matching passages for a query, by BM25
        over text and gist -- the narrowing step before the model judges
        relevance, and the whole of retrieval when no model is bound. Only
        passages that actually match are returned; when nothing matches at
        all, the first passages stand in so the model still sees the
        corpus rather than an empty room."""
        terms = set(words_of(query))
        if not terms or not self.passages:
            return self.passages[:limit]
        documents = [words_of(passage.searchable()) for passage in self.passages]
        counts = [Counter(document) for document in documents]
        total = len(documents)
        average_length = sum(len(document) for document in documents) / total
        frequency = {term: sum(1 for count in counts if term in count) for term in terms}
        scored: list[tuple[float, Passage]] = []
        for passage, document, count in zip(self.passages, documents, counts):
            saturation_floor = BM25_K1 * (1 - BM25_B + BM25_B * len(document) / average_length)
            score = sum(
                math.log(1 + (total - frequency[term] + 0.5) / (frequency[term] + 0.5))
                * (count[term] * (BM25_K1 + 1))
                / (count[term] + saturation_floor)
                for term in terms
                if term in count
            )
            scored.append((score, passage))
        matching = [passage for score, passage in sorted(scored, key=lambda pair: -pair[0]) if score > 0]
        return matching[:limit] if matching else self.passages[:limit]

    def narrowing(self) -> str:
        """What narrowing is actually scoring right now, for the retrieval
        record -- so the trail never claims an index that isn't there."""
        if any(passage.gist for passage in self.passages):
            return "BM25 over passage text and index gists"
        return "BM25 over passage text alone (no gist index)"

    # -- the persisted gist index ----------------------------------------------

    def missing_gists(self) -> list[Passage]:
        return [passage for passage in self.passages if not passage.gist]

    def load_index(self, path: Union[str, Path]) -> int:
        """Attach gists from a persisted `.ear/index.md` to the passages
        whose content hash still matches, and report how many attached.
        Entries for edited or removed passages simply find no match."""
        path = Path(path)
        if not path.exists():
            return 0
        gists: dict[str, str] = {}
        for section in parse_document(path.read_text(encoding="utf-8")).sections:
            words = section.name.split()
            if len(words) == 2 and normalize(words[0]) == "passage":
                gist = section.body(field_keys=("gist", "source")).field("gist")
                if gist:
                    gists[words[1]] = gist
        attached = 0
        for passage in self.passages:
            gist = gists.get(passage.fingerprint)
            if gist:
                passage.gist = gist
                attached += 1
        return attached

    def build_gists(self, lm: Any) -> int:
        """Ask the model for a one-line gist of every passage that lacks
        one. Each gist lands on its Passage as it is written, so a failure
        partway through loses nothing already built."""
        from .signatures import GistPassage

        built = 0
        for passage in self.missing_gists():
            reply = GistPassage.run(lm, passage=passage.render())
            lines = str(reply.gist).strip().splitlines()
            passage.gist = lines[0].strip() if lines else ""
            if passage.gist:
                built += 1
        return built

    def write_index(self, path: Union[str, Path], model_label: str = "") -> None:
        """Persist every gist to the index document, keyed by content
        hash, through the same Section codec the whole stack is authored
        with -- reviewable, diffable, and safe to delete (it rebuilds)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = f" by {model_label}" if model_label else ""
        lines = [
            "# Knowledge Index",
            "",
            f"One-line gists of the knowledge passages, written{writer} so",
            "differently-phrased questions still find them. Keyed by content",
            "hash: edit a source and its entry is rewritten on the next",
            "indexed load. Safe to delete -- it rebuilds.",
        ]
        for passage in self.passages:
            if not passage.gist:
                continue
            lines += [
                "",
                f"## Passage {passage.fingerprint}",
                "",
                f"Source: {passage.source}",
                f"Gist: {passage.gist}",
            ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def __len__(self) -> int:
        return len(self.passages)
