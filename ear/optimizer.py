"""Optimizer -- dev-time improvement of what the runtime reasons with,
native to EAR and fed by the artifacts the runtime already produces: the
reasoning trail and reviewer-labelled decision documents. No third-party
optimizer, no dataset format to maintain.

Three parts:

1. **Read the trail as a corpus.** `trainset_from_trail` turns the
   ReasoningLog's deliberation records (markdown or JSONL) into `Example`s
   -- the exact intent, context and stacked capabilities the model
   reasoned with, and the decision it reached.
2. **Read reviewer judgments as labels.** `verdicts_from_documents` reads
   decision documents a reviewer marked with a `## Review` section and a
   `Verdict:` line.
3. **Refine, natively.** `refine` reflectively improves a `Judgment`'s
   instruction against those examples -- the model reads what the current
   instruction produced (and any wrong verdicts) and rewrites the
   instruction, graded by the same `JudgeDecisionQuality`-backed `metric`
   the Examiner uses, so evaluation and optimization share one notion of
   quality. `refine_reasoner` applies this to the Reasoner's core prompt.

All dev-time, kept outside the per-cycle pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .section import labelled_blocks, normalize, parse_document, unquote

# The verdict vocabulary a reviewer's `Verdict:` line is read with --
# symmetrical with `coerce`'s yes/no reading. An unrecognized verdict is
# excluded rather than guessed at.
_POSITIVE_VERDICTS = {"correct", "pass", "passed", "yes", "true", "good", "approved"}
_NEGATIVE_VERDICTS = {"incorrect", "fail", "failed", "no", "false", "wrong", "bad"}


@dataclass
class Example:
    """One worked example drawn from the runtime's own record: what the
    model reasoned with and what it decided, plus an optional reviewer
    verdict. Dependency-free -- a plain record, not a framework object."""

    intent: str = ""
    context: dict = field(default_factory=dict)
    capabilities: str = ""
    decision: str = ""
    verdict: Optional[bool] = None
    note: str = ""


@dataclass
class Optimizer:
    """Curates trainsets and metrics from the runtime's markdown artifacts
    and refines reasoning instructions natively against them."""

    # -- read the trail --------------------------------------------------------

    def trainset_from_trail(self, path: Union[str, Path]) -> list[Example]:
        """Turn a reasoning trail's deliberation records into Examples.
        Reads both trail codecs by extension: `.md` through the Section
        parser, anything else as JSONL. A markdown record whose output
        survives only clipped (no `Output:` block, a shortened heading) is
        omitted -- a truncated label is not a training label."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        records = self._records_from_markdown(text) if path.suffix == ".md" else self._records_from_jsonl(text)
        examples: list[Example] = []
        for record in records:
            if record["stage"] != "deliberation" or not record["decision"]:
                continue
            examples.append(
                Example(
                    intent=str(record["inputs"].get("intent", "")),
                    context=record["inputs"].get("context", {}) if isinstance(record["inputs"].get("context"), dict) else {},
                    capabilities=str(record["inputs"].get("capabilities", "")),
                    decision=record["decision"],
                )
            )
        return examples

    def verdicts_from_documents(self, directory: Union[str, Path]) -> list[Example]:
        """Read reviewer-labelled decision documents: any `*.md` carrying a
        `## Review` section with a `Verdict:` line (and an optional
        blockquoted note). Documents with no review, or a verdict outside
        the vocabulary, are excluded -- never guessed."""
        labelled: list[Example] = []
        for path in sorted(Path(directory).glob("*.md")):
            document = parse_document(path.read_text(encoding="utf-8"))
            intent_text = decision_text = note = ""
            verdict: Optional[bool] = None
            for section in document.sections:
                key = normalize(section.name)
                blocks = labelled_blocks(section.lines)
                if key == "intent":
                    intent_text = unquote(section.lines)
                elif key == "decision":
                    decision_text = unquote(section.lines)
                elif "review" in key:
                    verdict = self._read_verdict(section.body(field_keys=("verdict",)).field("verdict"))
                    note = blocks.get("note", "") or unquote(section.lines)
            if verdict is None or not decision_text:
                continue
            labelled.append(Example(intent=intent_text, decision=decision_text, verdict=verdict, note=note))
        return labelled

    # -- the shared quality metric ---------------------------------------------

    def metric(self, model_binding: Optional[Any] = None) -> Callable[[str, str], float]:
        """A metric shared with the Examiner: with a model, a candidate
        decision is graded against the reference by `JudgeDecisionQuality`;
        without one, by normalized-containment equivalence -- structural,
        and only fit for smoke runs, which is exactly what no-model means."""

        def grade(expected: str, actual: str) -> float:
            if model_binding is not None and getattr(model_binding, "lm", None) is not None:
                from .signatures import JudgeDecisionQuality

                result = JudgeDecisionQuality.run(model_binding.lm, expected=str(expected), actual=str(actual))
                return 1.0 if result.passed else 0.0
            expected_key, actual_key = normalize(str(expected)), normalize(str(actual))
            return 1.0 if expected_key and (expected_key in actual_key or actual_key in expected_key) else 0.0

        return grade

    # -- native refinement ------------------------------------------------------

    def refine(self, judgment: Any, examples: list[Example], model_binding: Any) -> str:
        """Reflectively improve a Judgment's instruction against worked
        examples, in place. Requires a live model -- reflection is itself a
        judgment; with none bound this is a no-op returning the unchanged
        instruction. Returns the (possibly new) instruction."""
        if model_binding is None or getattr(model_binding, "lm", None) is None:
            return judgment.instruction
        if not examples:
            raise ValueError("refine needs at least one example to reflect on")
        from .signatures import RefineInstruction

        rendered = "\n\n".join(self._render_example(example) for example in examples)
        result = RefineInstruction.run(
            model_binding.lm, current_instruction=judgment.instruction, examples=rendered
        )
        improved = str(result.improved_instruction).strip()
        if improved:
            judgment.instruction = improved
        return judgment.instruction

    def refine_reasoner(
        self, runtime: Any, trail_path: Union[str, Path], reviews_directory: Optional[Union[str, Path]] = None
    ) -> str:
        """Refine the Reasoner's core instruction (`ReasonAboutIntent`)
        against the trail -- and reviewer verdicts, when a reviews
        directory is given -- in one call. Returns the refined instruction."""
        from .signatures import ReasonAboutIntent

        examples = self.trainset_from_trail(trail_path)
        if reviews_directory is not None:
            examples += self.verdicts_from_documents(reviews_directory)
        if not examples:
            raise ValueError(f"No usable examples found in trail '{trail_path}'")
        return self.refine(ReasonAboutIntent, examples, getattr(runtime, "model_binding", None))

    @staticmethod
    def _render_example(example: Example) -> str:
        lines = [f"Intent: {example.intent}", f"Decision: {example.decision}"]
        if example.verdict is not None:
            lines.append(f"Reviewer verdict: {'correct' if example.verdict else 'incorrect'}")
        if example.note:
            lines.append(f"Reviewer note: {example.note}")
        return "\n".join(lines)

    # -- trail codecs -----------------------------------------------------------

    @staticmethod
    def _records_from_jsonl(text: str) -> list[dict[str, Any]]:
        records = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            records.append(
                {
                    "stage": str(entry.get("stage", "")),
                    "inputs": dict(entry.get("inputs", {}) or {}),
                    "decision": str(entry.get("output", "")),
                }
            )
        return records

    def _records_from_markdown(self, text: str) -> list[dict[str, Any]]:
        records = []
        for section in parse_document(text).sections:
            stage, heading_output = self._read_record_heading(section.name)
            if not stage:
                continue
            body = section.body()
            inputs: dict[str, Any] = {}
            for bullet in body.bullets:
                key, separator, value = bullet.partition(": ")
                if separator:
                    inputs[key.strip()] = value.strip()
            blocks = labelled_blocks(section.lines)
            for label, block in blocks.items():
                if label not in ("output", "why"):
                    inputs.setdefault(label, block)
            decision = blocks.get("output", "")
            if not decision:
                # No Output block: the heading carries the full output only
                # when it was short enough to escape clipping.
                decision = "" if heading_output.endswith("...") else heading_output
            records.append({"stage": stage, "inputs": inputs, "decision": decision})
        return records

    @staticmethod
    def _read_record_heading(name: str) -> tuple[str, str]:
        """Split a trail record heading '### stage -- output (model)' back
        into (stage, output). Cycle headings and anything else return an
        empty stage."""
        stage, separator, rest = name.partition(" -- ")
        stage = stage.strip().lower()
        if not separator or not stage.isidentifier():
            return "", ""
        head, opener, tail = rest.rpartition("  (")
        output = head if opener and tail.endswith(")") else rest
        return stage, output.strip()

    @staticmethod
    def _read_verdict(value: str) -> Optional[bool]:
        """Read a reviewer's verdict word against the vocabulary; anything
        outside it is None -- excluded, never guessed."""
        words = normalize(value).split()
        if not words:
            return None
        if words[0] in _POSITIVE_VERDICTS:
            return True
        if words[0] in _NEGATIVE_VERDICTS:
            return False
        return None
