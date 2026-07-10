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


def coerce(text: str):
    """Read a markdown value back as the plain type it looks like --
    numbers as numbers, yes/no as booleans, everything else verbatim -- so
    facts written as `- loan_amount: 18500` in an intent document reach
    policies and reasoning as values, not strings."""
    value = text.strip()
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def quote(text: str) -> str:
    """Render free text as a markdown blockquote, so multi-line values (a
    decision, a rationale, a stacked capabilities block) can never be
    mistaken for document structure."""
    return "\n".join("> " + line if line else ">" for line in str(text).splitlines())


def unquote(lines: list[str]) -> str:
    """Reassemble a blockquote back into its original text."""
    recovered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("> "):
            recovered.append(stripped[2:])
        elif stripped == ">":
            recovered.append("")
    return "\n".join(recovered)


def labelled_blocks(lines: list[str]) -> dict[str, str]:
    """Collect `Label:` lines followed by blockquotes into label -> text --
    the reading half of the `Label:\\n> ...` idiom every markdown artifact
    in this package (session entries, trail records, decision documents)
    writes with `quote`. Labels are normalized; a label with no quote
    beneath it yields nothing."""
    blocks: dict[str, str] = {}
    label: str = ""
    pending: list[str] = []

    def commit() -> None:
        nonlocal label, pending
        if label and pending:
            blocks[normalize(label)] = unquote(pending)
        label, pending = "", []

    for line in lines + [""]:
        stripped = line.strip()
        if stripped.startswith(">"):
            if label:
                pending.append(line)
        elif _is_label(stripped):
            commit()
            label = stripped[:-1]
        elif stripped:
            commit()
        elif label and pending:
            commit()
    commit()
    return blocks


def argument_blocks(lines: list[str]) -> dict[str, str]:
    """Parse a tool call's arguments in either of two forms, freely mixed: a
    short scalar as a one-line bullet (`- name: value`, the original
    convention -- still the natural choice for a path, a flag, a number),
    or a value that needs more than one line as a label followed by a
    `>`-quoted block (`name:` then `> ...` lines) -- the only layout in this
    codec that reliably carries a multi-line value like a whole file's
    source, the way `quote`/`labelled_blocks` already carry a decision's or
    an explanation's multi-paragraph prose.

    Unlike `labelled_blocks`, a blank or unquoted line never ends an open
    label's block on its own -- only the next bullet or label does. That
    matters here specifically: `labelled_blocks` is read against markdown
    *EAR itself* writes with `quote`, which always quotes a blank line as a
    bare `>`; here the *model* is producing the text, and forgetting to
    quote one blank line inside a script must not silently truncate it.

    Unlike `labelled_blocks`, names are kept verbatim (only stripped), never
    case/underscore-folded by `normalize` -- a tool argument becomes a
    Python keyword argument, so `applicant_id` must stay `applicant_id`,
    not fold into `applicant id`."""
    blocks: dict[str, str] = {}
    label: str = ""
    pending: list[str] = []

    def commit() -> None:
        nonlocal label, pending
        if label:
            blocks[label] = "\n".join(pending).strip("\n")
        label, pending = "", []

    for line in lines:
        bullet = _BULLET.match(line)
        if bullet:
            commit()
            name, separator, value = bullet.group(1).partition(":")
            if separator and name.strip():
                blocks[name.strip()] = value.strip()
            continue
        stripped = line.strip()
        if not stripped.startswith(">") and _is_label(stripped):
            commit()
            label = stripped[:-1]
            continue
        if not label:
            continue
        if stripped.startswith("> "):
            pending.append(stripped[2:])
        elif stripped == ">":
            pending.append("")
        else:
            pending.append(line)
    commit()
    return blocks


def _is_label(stripped: str) -> bool:
    """A label line is a short word-or-phrase ending in a bare colon,
    e.g. 'Decision:' or 'Evidence basis:' -- never a sentence."""
    if not stripped.endswith(":") or len(stripped) > 40:
        return False
    head = stripped[:-1]
    return bool(head) and head[0].isalpha() and all(ch.isalnum() or ch in " _-" for ch in head)


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
        # The list a wrapped line continues: authors wrap long bullets and
        # numbered steps at the column limit, indenting the continuation.
        # Treating each line independently silently truncated every wrapped
        # item -- a workflow step lost the persona delegation at its end, a
        # deliverable bullet lost most of its meaning. An indented line that
        # is not itself a new item folds into the item above; a blank line
        # or a flush-left line ends the item, exactly as markdown reads.
        open_item: Optional[list] = None
        for line in self.lines:
            if not line.strip():
                prose_lines.append("")
                open_item = None
                continue
            bullet = _BULLET.match(line)
            if bullet:
                body.bullets.append(bullet.group(1).strip())
                open_item = body.bullets
                continue
            numbered = _NUMBERED.match(line)
            if numbered:
                body.numbered.append(numbered.group(1).strip())
                open_item = body.numbered
                continue
            if open_item is not None and line[:1].isspace():
                open_item[-1] = f"{open_item[-1]} {line.strip()}"
                continue
            open_item = None
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
