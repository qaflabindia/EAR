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

from .section import labelled_blocks, normalize, parse_document, quote, unquote

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
class SearchOutcome:
    """What an instruction search found: the winning instruction, how it
    scored against the baseline on the held-out examples, and what the
    search spent finding it."""

    instruction: str
    baseline_score: float
    best_score: float
    generations: int
    candidates_tried: int
    evaluations: int


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

    # -- refining Skills (the same primitive, aimed at a prompt) ------------------

    def refine_skill(self, skill: Any, examples: list[Example], model_binding: Any) -> str:
        """Reflectively improve a Skill's prompt against worked examples --
        the same `RefineInstruction` reflection N1.4 uses for a Judgment's
        instruction, aimed at `skill.prompt` instead. Requires a live
        model -- reflection is itself a judgment; with none bound this is a
        no-op returning the prompt unchanged."""
        current = skill.instruction()
        if model_binding is None or getattr(model_binding, "lm", None) is None:
            return current
        if not examples:
            raise ValueError("refine_skill needs at least one example to reflect on")
        from .signatures import RefineInstruction

        rendered = "\n\n".join(self._render_example(example) for example in examples)
        result = RefineInstruction.run(model_binding.lm, current_instruction=current, examples=rendered)
        improved = str(result.improved_instruction).strip()
        if improved:
            skill.prompt = improved
        return skill.instruction()

    def persist_skill(self, path: Union[str, Path], skill: Any) -> str:
        """Write a refined Skill back into its stacked skills.md, replacing
        just that skill's section -- the round-trip `Skill.to_markdown()`
        was built for, so a refined skill is reviewable and diffable
        exactly like any human-authored one, and survives past this
        session. Every other section is carried forward untouched."""
        path = Path(path)
        key = normalize(skill.name)
        rendered = skill.to_markdown().rstrip()
        parts: list[str] = []
        replaced = False
        if path.exists():
            document = parse_document(path.read_text(encoding="utf-8"))
            if document.title:
                parts.append(f"# {document.title}")
            for section in document.sections:
                if normalize(section.name) == key:
                    parts.append(rendered)
                    replaced = True
                else:
                    parts.append(f"## {section.name}\n" + "\n".join(section.lines).strip())
        if not replaced:
            parts.append(rendered)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n\n".join(part.strip() for part in parts if part.strip()) + "\n", encoding="utf-8")
        return str(path)

    # -- iterative instruction search --------------------------------------------

    def search(
        self,
        judgment: Any,
        examples: list[Example],
        model_binding: Any,
        metric: Optional[Callable[[str, str], float]] = None,
        generations: int = 2,
        candidates: int = 2,
        holdout: float = 0.5,
    ) -> SearchOutcome:
        """Iteratively improve a Judgment's instruction: each generation the
        model proposes `candidates` rewrites (reflecting on the current best
        and the failures), each candidate is graded on held-out reference
        examples by the shared quality metric, and the best survives. The
        loop, the split and the selection are code; the proposals and the
        grading are the model. Requires a live model -- optimization is
        judgment, so offline it refuses rather than pretending."""
        if model_binding is None or getattr(model_binding, "lm", None) is None:
            raise ValueError("search needs a live model -- optimization is judgment, and nobody made one offline")
        references = [example for example in examples if example.decision and example.verdict is not False]
        failures = [example for example in examples if example.verdict is False]
        if not references:
            raise ValueError("search needs at least one reference example (a decision not judged incorrect)")
        held_out = references[: max(1, int(len(references) * holdout))]
        reflect_on = references[len(held_out):] + failures or references

        grade = metric or self.metric(model_binding)
        primary = judgment.outputs[0]
        evaluations = 0

        def score(instruction: str) -> float:
            nonlocal evaluations
            original = judgment.instruction
            judgment.instruction = instruction
            try:
                total = 0.0
                for example in held_out:
                    values = {
                        spec.name: getattr(example, spec.name, "")
                        for spec in judgment.inputs
                        if getattr(example, spec.name, "")
                    }
                    result = judgment.run(model_binding.lm, **values)
                    reference = getattr(example, primary.name, "") or example.decision
                    total += grade(str(reference), str(getattr(result, primary.name, "")))
                    evaluations += 1
                return total / len(held_out)
            finally:
                judgment.instruction = original

        baseline = judgment.instruction
        best, best_score = baseline, score(baseline)
        baseline_score = best_score
        tried = 0
        from .signatures import RefineInstruction

        for _ in range(generations):
            proposed_this_generation: list[str] = []
            for _ in range(candidates):
                rendered = "\n\n".join(self._render_example(example) for example in reflect_on)
                if proposed_this_generation:
                    rendered += "\n\nAlready proposed this round (propose something meaningfully different):\n" + "\n---\n".join(
                        proposal[:300] for proposal in proposed_this_generation
                    )
                result = RefineInstruction.run(
                    model_binding.lm, current_instruction=best, examples=rendered
                )
                proposal = str(result.improved_instruction).strip()
                if not proposal or proposal == best:
                    continue
                proposed_this_generation.append(proposal)
                tried += 1
                candidate_score = score(proposal)
                if candidate_score > best_score:
                    best, best_score = proposal, candidate_score
        # The winner is applied only if it did not lose to the baseline --
        # a search that found nothing better changes nothing.
        judgment.instruction = best if best_score >= baseline_score else baseline
        return SearchOutcome(
            instruction=judgment.instruction,
            baseline_score=baseline_score,
            best_score=max(best_score, baseline_score),
            generations=generations,
            candidates_tried=tried,
            evaluations=evaluations,
        )

    # -- worked-example demos -----------------------------------------------------

    def select_demos(self, judgment: Any, examples: list[Example], budget_chars: int = 4000) -> list[dict]:
        """Choose which worked examples earn a place in the judgment's
        prompt: reviewer-approved first, then unreviewed, never ones judged
        incorrect, within a character budget enforced in code. Sets
        `judgment.demos` and returns them."""
        ranked = [e for e in examples if e.verdict is True] + [e for e in examples if e.verdict is None]
        field_names = [spec.name for spec in judgment.inputs + judgment.outputs]
        demos: list[dict] = []
        spent = 0
        for example in ranked:
            demo = {
                name: getattr(example, name)
                for name in field_names
                if str(getattr(example, name, "") or "").strip()
            }
            if not any(spec.name in demo for spec in judgment.outputs):
                continue  # a worked example without its answer teaches nothing
            size = sum(len(str(value)) for value in demo.values())
            if spent + size > budget_chars:
                break
            demos.append(demo)
            spent += size
        judgment.demos = demos
        return demos

    # -- persisted instructions ------------------------------------------------------

    def save_instructions(self, path: Union[str, Path], judgments: Optional[dict[str, Any]] = None) -> str:
        """Write every judgment's instruction (and demos) to a reviewable
        markdown file -- a persisted, diffable override of the shipped
        defaults in signatures.py."""
        if judgments is None:
            from .signatures import REGISTRY as judgments  # noqa: N811
        path = Path(path)
        lines = ["# Instructions -- refined", ""]
        for name, judgment in judgments.items():
            lines += [f"## {name}", "", "Instruction:", quote(judgment.instruction), ""]
            for number, demo in enumerate(judgment.demos, start=1):
                lines += [f"### {name} demo {number}", ""]
                for spec in judgment.inputs + judgment.outputs:
                    if spec.name in demo:
                        lines += [f"{spec.heading.capitalize()}:", quote(str(demo[spec.name])), ""]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    def load_instructions(self, path: Union[str, Path], judgments: Optional[dict[str, Any]] = None) -> list[str]:
        """Apply a persisted instructions file back onto the judgments.
        Refined instructions apply to the running process's judgments (the
        same scope as editing signatures.py, which is what this file is: a
        reviewable override of it). Returns the names applied."""
        if judgments is None:
            from .signatures import REGISTRY as judgments  # noqa: N811
        by_key = {normalize(name): (name, judgment) for name, judgment in judgments.items()}
        applied: list[str] = []
        current: Optional[Any] = None
        for section in parse_document(Path(path).read_text(encoding="utf-8")).sections:
            key = normalize(section.name)
            if key in by_key:
                name, current = by_key[key]
                blocks = labelled_blocks(section.lines)
                instruction = blocks.get("instruction", "")
                if instruction:
                    current.instruction = instruction
                    current.demos = []
                    applied.append(name)
            elif "demo" in key and current is not None:
                blocks = labelled_blocks(section.lines)
                demo = {}
                for spec in current.inputs + current.outputs:
                    value = blocks.get(normalize(spec.heading))
                    if value:
                        demo[spec.name] = value
                if demo:
                    current.demos.append(demo)
        return applied

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
