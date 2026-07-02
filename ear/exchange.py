"""Exchange -- the runtime's markdown-only boundary: intents arrive as
`.md` documents and decisions leave as `.md` documents, so nothing but
natural language crosses in either direction.

Two shapes:

- text in, text out: `exchange.respond(runtime, intent_markdown)` reads an
  intent document and returns a decision document.
- file drop: `exchange.run(runtime)` reads `intent.md` (answered as
  `decision.md`) and every `intents/<name>.md` (answered as
  `decisions/<name>.md`, same stem, so requests and responses pair by
  name). Intents whose decision document already exists are skipped, so
  re-running is idempotent -- an inbox, not a replay.

A Policy block is not an error at this boundary: the decision document is
written with `Status: BLOCKED` and carries the violated judgments with
their rationale, because a refusal is an outcome the requester -- and the
auditor -- must see. Every free-text value in the document (the decision,
the explanation, each rationale) is blockquoted so it can never be
mistaken for document structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .intent import Intent
from .reasoning_log import model_name
from .section import coerce, normalize, parse_document, quote


@dataclass
class Exchange:
    """A markdown in/out boundary rooted at one directory."""

    directory: Union[str, Path]

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)

    def respond(self, runtime: Any, intent_markdown: str) -> str:
        """Reason one intent document through the runtime and return the
        decision document."""
        intent = Intent.from_markdown(intent_markdown)
        blocked: Optional[PermissionError] = None
        decision: Any = None
        try:
            decision = runtime.reason(intent)
        except PermissionError as refusal:
            blocked = refusal
        return self._render(runtime, intent, decision, blocked)

    def run(self, runtime: Any) -> list[Path]:
        """Answer every unanswered intent document in the directory and
        return the decision documents written."""
        written: list[Path] = []
        for intent_path, decision_path in self._pairs():
            if decision_path.exists():
                continue
            decision_path.parent.mkdir(parents=True, exist_ok=True)
            decision_path.write_text(
                self.respond(runtime, intent_path.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
            written.append(decision_path)
        return written

    def _pairs(self) -> list[tuple[Path, Path]]:
        pairs: list[tuple[Path, Path]] = []
        single = self.directory / "intent.md"
        if single.exists():
            pairs.append((single, self.directory / "decision.md"))
        inbox = self.directory / "intents"
        if inbox.is_dir():
            for intent_path in sorted(inbox.glob("*.md")):
                pairs.append((intent_path, self.directory / "decisions" / intent_path.name))
        return pairs

    @staticmethod
    def _render(runtime: Any, intent: Intent, decision: Any, blocked: Optional[PermissionError]) -> str:
        title = intent.text.partition("\n")[0].strip()
        cycle = getattr(runtime.reasoning_log, "cycle", 0)
        lines = [
            f"# Decision -- {title}",
            "",
            f"Runtime: {runtime.name}",
            f"Cycle: {cycle}",
            f"Model: {model_name(getattr(runtime, 'model_binding', None))}",
            f"Status: {'BLOCKED' if blocked is not None else 'decided'}",
            "",
            "## Intent",
            "",
            quote(intent.text),
        ]
        if intent.context:
            lines += ["", "Context:", ""]
            lines += [f"- {key}: {value}" for key, value in intent.context.items()]

        lines += ["", "## Decision", ""]
        if blocked is not None:
            lines += [quote(str(blocked))]
        else:
            lines += [quote(str(decision))]

        if blocked is None and runtime.memory.working:
            evidence = runtime.memory.working[-1].evidence
            if evidence is not None:
                data = evidence.sources.get("data")
                if data:
                    lines += ["", "## Data", ""]
                    lines += [f"- {name}: {value}" for name, value in data.items()]
                explanation = evidence.sources.get("explanation", "")
                if explanation:
                    lines += ["", "## Explanation", "", quote(str(explanation))]
                lines += ["", "## Evidence", "", f"Basis: {evidence.basis}"]
                plan = evidence.sources.get("plan")
                if plan:
                    lines += [f"Plan: {', '.join(plan)}"]
                citations = evidence.sources.get("citations")
                if citations:
                    lines += ["", "## Sources", ""]
                    lines += [f"- {citation}" for citation in citations]

        judgments = [record for record in runtime.reasoning_log.for_cycle(cycle) if record.stage == "policy"]
        if judgments:
            lines += ["", "## Policy judgments", ""]
            for record in judgments:
                lines += [f"- {record.inputs.get('policy', 'policy')}: {record.output}"]
                if record.rationale:
                    lines += [f"  {record.rationale}"]
        return "\n".join(lines) + "\n"


def data_from_decision_document(markdown: str) -> dict:
    """Read the `## Data` section of a decision document back into typed
    values -- the parse half of the Contract's markdown round-trip, through
    the same Section parser and `coerce` codec the whole stack uses."""
    data: dict = {}
    for section in parse_document(markdown).sections:
        if normalize(section.name) != "data":
            continue
        for bullet in section.body().bullets:
            name, separator, value = bullet.partition(": ")
            if not separator:
                name, separator, value = bullet.partition(":")
            if separator and name.strip():
                data[name.strip()] = coerce(value)
    return data