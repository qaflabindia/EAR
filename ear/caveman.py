"""caveman -- a deterministic, zero-dependency prose compressor for text that
re-enters the native tool loop's `gathered` context (see `Reasoner._compress_tool_result`).

Ported from the MIT-licensed `caveman-shrink` MCP middleware
(github.com/JuliusBrussee/caveman, `src/mcp-servers/caveman-shrink/compress.js`),
which exists for exactly this shape of problem: compressing text a model
did not itself generate (a tool's raw output, an MCP server's tool
description), where an exact fact -- a row count, an exit code, a file
path -- must never be paraphrased away.

This is a pure `re.sub` pipeline, never a model call: it can only delete
matched filler words from the input, never generate replacement text, so
it is structurally incapable of the failure an LLM-based summarizer is
prone to -- inventing or garbling a number it was supposed to preserve
(observed in the wild: a Haiku summary reported "1000 rows" for a file
that actually held 907, which then caused a downstream cycle to distrust
a perfectly good dataset and block on a fact that was never true).

Boundaries (never touched, via sentinel substitution before compression
and restoration after):
    - fenced code blocks (``` ... ```)
    - inline code (`...`)
    - URLs (https?://...)
    - filesystem paths (anything with / or \\)
    - CONST_CASE identifiers
    - dotted.method() / pkg.fn() calls
    - bare function calls (name(args))
    - version numbers (1.2.3)
Bare numbers need no explicit protection: none of the compression rules
below match digits at all.

Compression applied to everything else: drop articles, filler words,
pleasantries, hedging, and leading "I'll"/"I will"/"let me"-style phrases;
collapse the whitespace that removal leaves behind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FILLERS = re.compile(
    r"\b(?:just|really|basically|actually|simply|quite|very|essentially|literally)\b",
    re.IGNORECASE,
)

_PLEASANTRIES = re.compile(
    r"\b(?:please|kindly|thank you|thanks|sure|certainly|of course|happy to|i'?d be happy)\b[,.]?\s*",
    re.IGNORECASE,
)

_HEDGES = re.compile(
    r"\b(?:perhaps|maybe|might|could potentially|would like to|i think|in my opinion|it seems|it appears)\b\s*",
    re.IGNORECASE,
)

_LEADERS = re.compile(
    r"^(?:i'?ll|i will|i can|i'?d|you can|we will|we can|let me|let'?s)\s+",
    re.IGNORECASE | re.MULTILINE,
)

_ARTICLES = re.compile(r"\b(?:a|an|the)\s+(?=[a-z])", re.IGNORECASE)

# Spans we never rewrite, even sitting inside ordinary prose. Order matters:
# fenced/inline code and URLs are claimed first so a path- or identifier-like
# regex further down can't fire on a substring already inside one of them.
_PROTECTED_PATTERNS = [
    re.compile(r"```[\s\S]*?```"),                              # fenced code
    re.compile(r"`[^`\n]+`"),                                   # inline code
    re.compile(r"\bhttps?://\S+", re.IGNORECASE),                # URLs
    re.compile(r"\b[\w.-]*[/\\][\w./\\-]+"),                     # paths with / or \
    re.compile(r"\b[A-Z][A-Za-z0-9]*(?:_[A-Z][A-Za-z0-9]*)+\b"),  # CONST_CASE
    re.compile(r"\b\w+\.\w+(?:\.\w+)*\(\)?"),                    # dotted.method or pkg.fn()
    re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)"),           # function calls
    re.compile(r"\b\d+\.\d+\.\d+\b"),                            # version numbers
]

_SENTINEL = re.compile(r"\x00(\d+)\x00")


def _with_protected_segments(text: str, transform) -> str:
    """Replace every protected match with a numbered sentinel, run
    `transform` on what's left, then splice the originals back in -- so
    compression never sees, and can never touch, a protected span."""
    segments: list[str] = []

    def claim(match: re.Match) -> str:
        index = len(segments)
        segments.append(match.group(0))
        return f"\x00{index}\x00"

    working = text
    for pattern in _PROTECTED_PATTERNS:
        working = pattern.sub(claim, working)

    out = transform(working)
    return _SENTINEL.sub(lambda m: segments[int(m.group(1))], out)


def _compress_prose(text: str) -> str:
    s = text
    s = _LEADERS.sub("", s)
    s = _PLEASANTRIES.sub("", s)
    s = _HEDGES.sub("", s)
    s = _FILLERS.sub("", s)
    s = _ARTICLES.sub("", s)
    # Collapse repeated whitespace introduced by removals.
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    # Capitalize the first letter of each sentence we may have left lowercase.
    s = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    return s.strip()


@dataclass
class Compressed:
    """A compression's result: the text, and how much it shrank."""

    text: str
    before: int
    after: int

    @property
    def saved_pct(self) -> float:
        return 0.0 if self.before == 0 else round((self.before - self.after) / self.before * 100, 1)


def compress(text: str) -> Compressed:
    """Compress `text` deterministically -- drop filler prose, never touch a
    protected span or a bare number. Safe to call on anything, including
    text with no natural language in it at all (a bare number, a path): it
    degrades to a no-op rather than raising."""
    if not isinstance(text, str) or not text:
        return Compressed(text=text if isinstance(text, str) else str(text), before=0, after=0)
    before = len(text)
    compressed_text = _with_protected_segments(text, _compress_prose)
    return Compressed(text=compressed_text, before=before, after=len(compressed_text))
