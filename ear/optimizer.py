"""Optimizer -- dev-time refinement of what the runtime reasons with.

Two loops live here, both fed by artifacts the runtime already produces --
no hand-built datasets:

1. **Trail -> GEPA.** `trainset_from_trail` turns the ReasoningLog's
   deliberation records (markdown or JSONL) into `dspy.Example`s -- the
   exact intent, context and stacked capabilities the model reasoned with,
   and the decision it reached. `optimize_from_trail` feeds them to the
   Reasoner's GEPA hook, with `metric` grading candidates the same way the
   Examiner grades evaluations (`JudgeDecisionQuality`), so evaluation and
   optimization share one notion of quality. Reviewer judgments enter the
   loop as markdown too: `verdicts_from_documents` reads decision documents
   to which a reviewer added a `## Review` section with a `Verdict:` line.

2. **SkillOpt.** The pre-existing ReflACT loop for refining a Persona's
   skill document (`optimize`/`apply`), unchanged.

Both are structural, dev-time operations, kept outside the per-cycle
pipeline for the same reason Evolver is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .persona import Persona
from .section import labelled_blocks, normalize, parse_document, unquote

# The verdict vocabulary a reviewer's `Verdict:` line is read with --
# symmetrical with `coerce`'s yes/no reading. An unrecognized verdict is
# excluded from the trainset rather than guessed at.
_POSITIVE_VERDICTS = {"correct", "pass", "passed", "yes", "true", "good", "approved"}
_NEGATIVE_VERDICTS = {"incorrect", "fail", "failed", "no", "false", "wrong", "bad"}


@dataclass
class Optimizer:
    """Optimizer curates trainsets and metrics from the runtime's own
    markdown artifacts and drives the sanctioned optimization hooks; plus
    the SkillOpt trainer/apply pair for Persona skill documents."""

    # -- trail -> GEPA --------------------------------------------------------

    def trainset_from_trail(self, path: Union[str, Path]) -> list[Any]:
        """Turn a reasoning trail's deliberation records into
        `dspy.Example`s (intent, context, capabilities -> decision).

        Reads both trail codecs by extension: `.md` through the Section
        parser, anything else as JSONL. A markdown record whose output
        survives only in clipped form (no `Output:` block and a shortened
        heading) is omitted -- a truncated label is not a training label.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        records = self._records_from_markdown(text) if path.suffix == ".md" else self._records_from_jsonl(text)
        import dspy

        examples = []
        for record in records:
            if record["stage"] != "deliberation" or not record["decision"]:
                continue
            examples.append(
                dspy.Example(
                    intent=record["inputs"].get("intent", ""),
                    context=record["inputs"].get("context", {}),
                    capabilities=record["inputs"].get("capabilities", ""),
                    decision=record["decision"],
                ).with_inputs("intent", "context", "capabilities")
            )
        return examples

    def verdicts_from_documents(self, directory: Union[str, Path]) -> list[Any]:
        """Read reviewer-labelled decision documents: any `decisions/*.md`
        carrying a `## Review` section with a `Verdict:` line (and an
        optional blockquoted note). Returns `dspy.Example`s with intent,
        decision, verdict (bool) and note. Documents with no review, or a
        verdict outside the vocabulary, are excluded, never guessed."""
        import dspy

        labelled = []
        for path in sorted(Path(directory).glob("*.md")):
            document = parse_document(path.read_text(encoding="utf-8"))
            intent_text = decision_text = note = ""
            verdict: Optional[bool] = None
            for section in document.sections:
                key = normalize(section.name)
                blocks = labelled_blocks(section.lines)
                if key == "intent":
                    intent_text = self._quoted_text(section.lines)
                elif key == "decision":
                    decision_text = self._quoted_text(section.lines)
                elif "review" in key:
                    verdict = self._read_verdict(section.body(field_keys=("verdict",)).field("verdict"))
                    note = blocks.get("note", "") or self._quoted_text(section.lines)
            if verdict is None or not decision_text:
                continue
            labelled.append(
                dspy.Example(intent=intent_text, decision=decision_text, verdict=verdict, note=note).with_inputs(
                    "intent"
                )
            )
        return labelled

    def metric(self, model_binding: Optional[Any] = None) -> Callable[..., float]:
        """A GEPA/Examiner-shared metric: with a model, the candidate
        decision is graded against the reference by `JudgeDecisionQuality`;
        without one, by normalized-containment equivalence -- structural,
        and only fit for smoke runs, which is exactly what no-model means."""

        def grade(gold: Any, prediction: Any, trace: Any = None, *args: Any, **kwargs: Any) -> float:
            expected = str(getattr(gold, "decision", gold))
            actual = str(getattr(prediction, "decision", prediction))
            if model_binding is not None and getattr(model_binding, "lm", None) is not None:
                import dspy

                from .signatures import JudgeDecisionQuality

                judge = dspy.Predict(JudgeDecisionQuality)
                with dspy.context(lm=model_binding.lm):
                    result = judge(expected=expected, actual=actual)
                return 1.0 if result.passed else 0.0
            expected_key, actual_key = normalize(expected), normalize(actual)
            return 1.0 if expected_key and (expected_key in actual_key or actual_key in expected_key) else 0.0

        return grade

    def optimize_from_trail(
        self,
        runtime: Any,
        trail_path: Union[str, Path],
        metric: Optional[Callable[..., float]] = None,
        **gepa_kwargs: Any,
    ) -> Any:
        """The whole loop in one call: trail -> trainset -> GEPA on the
        runtime's Reasoner, graded by the shared metric against the
        runtime's own model binding."""
        trainset = self.trainset_from_trail(trail_path)
        if not trainset:
            raise ValueError(f"No usable deliberation records found in trail '{trail_path}'")
        chosen_metric = metric or self.metric(getattr(runtime, "model_binding", None))
        return runtime.reasoner.optimize_with_gepa(trainset, chosen_metric, **gepa_kwargs)

    # -- trail codecs ----------------------------------------------------------

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
    def _quoted_text(lines: list[str]) -> str:
        return unquote(lines)

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

    # -- SkillOpt (unchanged) ---------------------------------------------------

    def optimize(self, config: Union[str, dict], adapter: Any) -> Any:
        """Build a SkillOpt trainer: call `.train()` yourself, then `apply`
        the trained document -- SkillOpt has no one-call API, and this
        mirrors that rather than inventing one."""
        from .integrations.skillopt_backend import build_trainer

        return build_trainer(config, adapter)

    def apply(self, persona: Persona, skill_name: str, trained_document: str) -> Persona:
        from .integrations.skillopt_backend import apply_trained_skill

        return apply_trained_skill(persona, skill_name, trained_document)