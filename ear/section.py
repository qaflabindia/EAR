"""Section -- one named block of a stacked markdown document.

The whole authoring model of EAR is markdown written in plain English:
prompts are stacked in `skills.md`, skills in `persona.md`, steps in
`workflow.md`, workflows in `process.md`, governance in `policy.md` and the
operating strategy in `memory.md`. This module is the one structural parser
they all share: it splits a document into a title, a preamble and named
Sections, and lets each loader pull out only the field lines it knows about
-- every other line stays natural-language prose, never swallowed by
accident.

Parsing here is structural, not judgmental (headings, bullets, numbered
items, `Key: value` lines), so -- like the Selector, Composer and Scheduler
-- it stays plain Python. The judgment over what the prose *means* happens
later, in the runtime's LLM-backed reasoning stages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")


def normalize(text: str) -> str:
    """Fold case, whitespace, hyphens and underscores so authors can refer
    to 'Credit Risk Guru', 'credit-risk-guru' or 'credit_risk_guru'
    interchangeably."""
    return re.sub(r"[\s_-]+", " ", text.strip().lower())


@dataclass
class Body:
    """A Section's content, structured: recognized `Key: value` fields,
    bullets, numbered items, and everything else kept verbatim as prose."""

    fields: dict[str, str] = field(default_factory=dict)
    bullets: list[str] = field(default_factory=list)
    numbered: list[str] = field(default_factory=list)
    prose: str = ""

    def field(self, *names: str) -> str:
        """The first recognized field value among the given names."""
        for name in names:
            value = self.fields.get(normalize(name))
            if value:
                return value
        return ""


@dataclass
class Section:
    """One heading and the lines beneath it, up to the next heading."""

    name: str
    lines: list[str] = field(default_factory=list)

    def body(self, field_keys: Iterable[str] = ()) -> Body:
        """Structure this section's lines. Only lines whose key appears in
        `field_keys` become fields -- a colon inside ordinary prose (or an
        unknown key) stays part of the prose, so nothing an author writes is
        silently dropped."""
        keys = {normalize(key) for key in field_keys}
        body = Body()
        prose_lines: list[str] = []
        for line in self.lines:
            if not line.strip():
                prose_lines.append("")
                continue
            bullet = _BULLET.match(line)
            if bullet:
                body.bullets.append(bullet.group(1).strip())
                continue
            numbered = _NUMBERED.match(line)
            if numbered:
                body.numbered.append(numbered.group(1).strip())
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                if normalize(key) in keys:
                    body.fields[normalize(key)] = value.strip()
                    continue
            prose_lines.append(line.strip())
        body.prose = _paragraphs(prose_lines)
        return body


@dataclass
class Document:
    """A parsed markdown file: its `# Title`, any prose before the first
    section heading, and the named Sections that follow."""

    title: str = ""
    preamble: str = ""
    sections: list[Section] = field(default_factory=list)

    def section_named(self, name: str) -> Optional[Section]:
        key = normalize(name)
        return next((s for s in self.sections if normalize(s.name) == key), None)


def parse_document(text: str) -> Document:
    """Split markdown text into a Document of named Sections. The first
    level-1 heading is the document title; every later heading (any level)
    starts a new section."""
    document = Document()
    current: Optional[Section] = None
    preamble_lines: list[str] = []
    for raw in text.lstrip("\ufeff").replace("\r\n", "\n").split("\n"):
        heading = _HEADING.match(raw)
        if heading:
            level, name = len(heading.group(1)), heading.group(2).strip()
            if level == 1 and not document.title and not document.sections:
                document.title = name
                continue
            current = Section(name=name)
            document.sections.append(current)
            continue
        if current is None:
            preamble_lines.append(raw)
        else:
            current.lines.append(raw)
    document.preamble = _paragraphs([line.strip() for line in preamble_lines])
    return document


def _paragraphs(lines: list[str]) -> str:
    """Join wrapped lines back into paragraphs, preserving blank-line
    paragraph breaks."""
    paragraphs: list[list[str]] = [[]]
    for line in lines:
        if line == "":
            if paragraphs[-1]:
                paragraphs.append([])
        else:
            paragraphs[-1].append(line)
    return "\n\n".join(" ".join(paragraph) for paragraph in paragraphs if paragraph)
