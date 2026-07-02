"""Examiner -- examine: run a markdown-native evaluation suite against the
runtime and grade every outcome, so quality is measured before prompts are
optimized and regressions are caught before they ship.

An evaluation is one markdown document in an `evaluations/` directory: an
ordinary intent document plus an `## Expected` section -- prose criteria
for what should happen (a blocked refusal can itself be the expectation),
and optionally bullets of `field: value` naming deliverable Data the
decision must carry. No dataset format, no JSON: the same authoring
surface as everything else.

Grading is a runtime judgment: with a model bound, `JudgeDecisionQuality`
reads the expectation against the actual outcome and returns a verdict
with a rationale. Offline, only the structural expectations (the bullets)
are checked -- by typed equality against the decision's Data or normalized
containment in the decision text -- and prose-only criteria are reported
as **ungraded** rather than faked: a judgment nobody made is never written
down as one.

Every verdict lands on the runtime's ReasoningLog (stage `evaluation`) and
in `evaluations/report.md`, written and read through the same Section
codec as every other artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from .intent import Intent
from .reasoning_log import calls_so_far, model_name, usage_since
from .section import coerce, normalize, parse_document, quote

REPORT_FILENAME = "report.md"


@dataclass
class EvaluationResult:
    """One graded evaluation: the verdict (True/False, or None when it
    could not be honestly graded), and why."""

    name: str
    passed: Optional[bool]
    rationale: str
    expected: str
    outcome: str

    @property
    def verdict(self) -> str:
        if self.passed is None:
            return "ungraded"
        return "passed" if self.passed else "FAILED"


@dataclass
class Examination:
    """A whole suite's results, with the roll-up the CI gate reads."""

    runtime_name: str
    results: list[EvaluationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when nothing graded failed. Ungraded results do not fail
        the suite -- they are visible in the report instead."""
        return all(result.passed is not False for result in self.results)

    def counts(self) -> dict[str, int]:
        return {
            "examined": len(self.results),
            "passed": sum(1 for r in self.results if r.passed is True),
            "failed": sum(1 for r in self.results if r.passed is False),
            "ungraded": sum(1 for r in self.results if r.passed is None),
        }

    def render(self) -> str:
        counts = self.counts()
        lines = [
            f"# Evaluation Report -- {self.runtime_name}",
            "",
            f"Examined: {counts['examined']} | Passed: {counts['passed']} | "
            f"Failed: {counts['failed']} | Ungraded: {counts['ungraded']}",
        ]
        for result in self.results:
            lines += ["", f"## {result.name} -- {result.verdict}", ""]
            lines += ["Expected:", quote(result.expected), ""]
            lines += ["Outcome:", quote(result.outcome), ""]
            if result.rationale:
                lines += ["Why:", quote(result.rationale), ""]
        return "\n".join(lines) + "\n"


@dataclass
class Examiner:
    """An Examiner examines a runtime against an evaluations directory."""

    def examine(self, runtime: Any, directory: Union[str, Path]) -> Examination:
        directory = Path(directory)
        examination = Examination(runtime_name=runtime.name)
        for path in sorted(directory.glob("*.md")):
            if path.name == REPORT_FILENAME:
                continue
            examination.results.append(self._examine_one(runtime, path))
        (directory / REPORT_FILENAME).write_text(examination.render(), encoding="utf-8")
        return examination

    def _examine_one(self, runtime: Any, path: Path) -> EvaluationResult:
        text = path.read_text(encoding="utf-8")
        intent = Intent.from_markdown(text, skip_sections=("expected",))
        expected_prose, expected_fields = self._expectation(text)

        try:
            outcome = str(runtime.reason(intent))
            data = self._delivered_data(runtime)
        except PermissionError as blocked:
            # A refusal is an outcome an evaluation may well expect -- and
            # a blocked cycle delivered no data, whatever an earlier cycle
            # left in memory.
            outcome = f"BLOCKED: {blocked}"
            data = {}
        graded_outcome = outcome if not data else outcome + "\n\nData:\n" + "\n".join(
            f"- {name}: {value}" for name, value in data.items()
        )

        model_binding = getattr(runtime, "model_binding", None)
        grade_start = calls_so_far(getattr(model_binding, "lm", None))
        passed, rationale = self._grade(runtime, expected_prose, expected_fields, graded_outcome, data)
        result = EvaluationResult(
            name=path.stem,
            passed=passed,
            rationale=rationale,
            expected=self._render_expectation(expected_prose, expected_fields),
            outcome=graded_outcome,
        )
        runtime.reasoning_log.record(
            stage="evaluation",
            inputs={"evaluation": result.name, "expected": result.expected},
            output=result.verdict,
            rationale=rationale,
            model="" if passed is None else model_name(model_binding),
            usage=usage_since(getattr(model_binding, "lm", None), grade_start),
        )
        runtime.reasoning_log.flush()
        return result

    # -- reading the expectation ---------------------------------------------

    @staticmethod
    def _expectation(text: str) -> tuple[str, dict[str, Any]]:
        prose = ""
        fields: dict[str, Any] = {}
        for section in parse_document(text).sections:
            if "expected" not in normalize(section.name):
                continue
            body = section.body()
            prose = body.prose
            for bullet in body.bullets:
                name, separator, value = bullet.partition(": ")
                if not separator:
                    name, separator, value = bullet.partition(":")
                if separator and name.strip():
                    fields[name.strip()] = coerce(value)
        return prose, fields

    @staticmethod
    def _render_expectation(prose: str, fields: dict[str, Any]) -> str:
        parts = [prose] if prose else []
        parts += [f"- {name}: {value}" for name, value in fields.items()]
        return "\n".join(parts) or "no expectation declared"

    # -- grading ---------------------------------------------------------------

    def _grade(
        self,
        runtime: Any,
        expected_prose: str,
        expected_fields: dict[str, Any],
        outcome: str,
        data: dict[str, Any],
    ) -> tuple[Optional[bool], str]:
        model_binding = getattr(runtime, "model_binding", None)
        if model_binding is not None and getattr(model_binding, "lm", None) is not None:
            expectation = self._render_expectation(expected_prose, expected_fields)
            return self._grade_with_llm(expectation, outcome, model_binding.lm)
        if expected_fields:
            return self._grade_structurally(expected_fields, outcome, data)
        if expected_prose:
            return None, "ungraded offline -- prose criteria need a model to judge; only field expectations are checked without one"
        return None, "ungraded -- the evaluation declares no expectation"

    @staticmethod
    def _grade_with_llm(expectation: str, outcome: str, lm: Any) -> tuple[bool, str]:
        from .signatures import JudgeDecisionQuality

        result = JudgeDecisionQuality.run(lm, expected=expectation, actual=outcome)
        return bool(result.passed), str(result.rationale)

    @staticmethod
    def _grade_structurally(
        expected_fields: dict[str, Any], outcome: str, data: dict[str, Any]
    ) -> tuple[bool, str]:
        """The offline check: an expected field passes on typed equality
        against the decision's delivered Data, or -- when no Data carries
        it -- on normalized containment in the outcome text. Structural,
        and reported as such."""
        delivered = {normalize(str(name)): value for name, value in data.items()}
        failures: list[str] = []
        for name, expected_value in expected_fields.items():
            key = normalize(str(name))
            if key in delivered:
                if delivered[key] != expected_value:
                    failures.append(f"{name}: delivered '{delivered[key]}', expected '{expected_value}'")
            elif normalize(str(expected_value)) not in normalize(outcome):
                failures.append(f"{name}: '{expected_value}' not found in the outcome")
        if failures:
            return False, "structural check only (no model bound): " + "; ".join(failures)
        return True, "structural check only (no model bound): every expected field was honored"

    @staticmethod
    def _delivered_data(runtime: Any) -> dict[str, Any]:
        memory = getattr(runtime, "memory", None)
        if memory is None or not memory.working:
            return {}
        evidence = memory.working[-1].evidence
        if evidence is None:
            return {}
        return dict(evidence.sources.get("data", {}) or {})