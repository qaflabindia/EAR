"""Examiner -- examine: run a markdown-native evaluation suite against the
runtime and grade every outcome, so quality is measured before prompts are
optimized and regressions are caught before they ship.

An evaluation is one markdown document in an `evaluations/` directory: an
ordinary intent document plus an `## Expected` section -- prose criteria
for what should happen (a blocked refusal can itself be the expectation),
`- field: value` bullets naming deliverable Data the decision must carry,
and colon-less bullets as a graded rubric: each such criterion is judged
separately, with its own verdict and rationale, so an evaluation scores on
several axes rather than one pass/fail. No dataset format, no JSON: the
same authoring surface as everything else.

Grading is a runtime judgment: with a model bound, `JudgeDecisionQuality`
reads the expectation against the actual outcome and returns a verdict
with a rationale -- once for the prose-and-fields expectation, once per
rubric criterion. Offline, only the structural expectations (the field
bullets) are checked -- by typed equality against the decision's Data or
normalized containment in the decision text -- and prose criteria are
reported as **ungraded** rather than faked: a judgment nobody made is
never written down as one.

Every run is regression history: reports append to
`evaluations/reports/<timestamp>.md` with `report.md` always the latest,
and each report diffs itself against the previous one -- newly failing,
newly passing, still failing -- so a prompt edit shows its consequences as
a markdown document, not a dashboard.

`compare(runtime_a, runtime_b, directory)` answers the other evaluation
question -- not "is this stack good?" but "which of these two stacks is
better?": both answer every evaluation, a pairwise `JudgePreference`
judgment picks A, B or tie per expectation, and the preference report
lands in `evaluations/comparison.md`. Comparison refuses to run without a
model: a preference judgment nobody made is never written down.

Every verdict lands on the runtime's ReasoningLog (stage `evaluation`, and
`comparison` for pairwise runs), written and read through the same Section
codec as every other artifact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from .intent import Intent
from .reasoning_log import calls_so_far, model_name, usage_since
from .section import coerce, normalize, parse_document, quote

REPORT_FILENAME = "report.md"
REPORTS_DIRNAME = "reports"
COMPARISON_FILENAME = "comparison.md"

# How a report section heading carries its verdict: "## name -- verdict".
_VERDICT_SEPARATOR = " -- "
_KNOWN_VERDICTS = {"passed", "failed", "ungraded"}


@dataclass
class CriterionResult:
    """One rubric criterion's verdict: the authored criterion, whether the
    outcome satisfied it (None when nobody could honestly judge), and why."""

    text: str
    passed: Optional[bool]
    rationale: str

    @property
    def verdict(self) -> str:
        if self.passed is None:
            return "ungraded"
        return "passed" if self.passed else "FAILED"


@dataclass
class EvaluationResult:
    """One graded evaluation: the verdict (True/False, or None when it
    could not be honestly graded), why, and the rubric's per-criterion
    verdicts."""

    name: str
    passed: Optional[bool]
    rationale: str
    expected: str
    outcome: str
    criteria: list[CriterionResult] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.passed is None:
            return "ungraded"
        return "passed" if self.passed else "FAILED"


@dataclass
class Examination:
    """A whole suite's results, with the roll-up the CI gate reads and the
    regression diff against the previous report."""

    runtime_name: str
    results: list[EvaluationResult] = field(default_factory=list)
    # Verdicts parsed from the previous report.md, or None when this is
    # the first report -- {} means a previous report existed but graded
    # nothing, which is a different story than "no history".
    prior_verdicts: Optional[dict[str, str]] = None

    @property
    def passed(self) -> bool:
        """True when nothing graded failed. Ungraded results do not fail
        the suite -- they are visible in the report instead."""
        return all(result.passed is not False for result in self.results)

    def counts(self) -> dict[str, int]:
        criteria = [criterion for result in self.results for criterion in result.criteria]
        return {
            "examined": len(self.results),
            "passed": sum(1 for r in self.results if r.passed is True),
            "failed": sum(1 for r in self.results if r.passed is False),
            "ungraded": sum(1 for r in self.results if r.passed is None),
            "criteria": len(criteria),
            "criteria_passed": sum(1 for c in criteria if c.passed is True),
            "criteria_failed": sum(1 for c in criteria if c.passed is False),
        }

    def changes(self) -> dict[str, list[str]]:
        """The regression diff against the previous report: what fails now
        that didn't before, what recovered, and what is still broken."""
        prior = self.prior_verdicts or {}
        current = {result.name: result.verdict for result in self.results}
        return {
            "newly failing": sorted(
                name for name, verdict in current.items() if verdict == "FAILED" and prior.get(name) != "FAILED"
            ),
            "newly passing": sorted(
                name for name, verdict in current.items() if verdict == "passed" and prior.get(name) == "FAILED"
            ),
            "still failing": sorted(
                name for name, verdict in current.items() if verdict == "FAILED" and prior.get(name) == "FAILED"
            ),
        }

    def render(self) -> str:
        counts = self.counts()
        headline = (
            f"Examined: {counts['examined']} | Passed: {counts['passed']} | "
            f"Failed: {counts['failed']} | Ungraded: {counts['ungraded']}"
        )
        if counts["criteria"]:
            headline += f" | Rubric criteria: {counts['criteria_passed']}/{counts['criteria']} passed"
        lines = [f"# Evaluation Report -- {self.runtime_name}", "", headline]
        if self.prior_verdicts is not None:
            lines += ["", "## Changes Since Last Report", ""]
            changed = False
            for label, names in self.changes().items():
                if names:
                    lines.append(f"- {label}: {', '.join(names)}")
                    changed = True
            if not changed:
                lines.append("- no verdict changed since the last report")
        for result in self.results:
            lines += ["", f"## {result.name}{_VERDICT_SEPARATOR}{result.verdict}", ""]
            lines += ["Expected:", quote(result.expected), ""]
            lines += ["Outcome:", quote(result.outcome), ""]
            if result.rationale:
                lines += ["Why:", quote(result.rationale), ""]
            if result.criteria:
                lines += ["Rubric:"] + [
                    f"- {criterion.verdict}: {criterion.text}"
                    + (f" ({criterion.rationale})" if criterion.rationale else "")
                    for criterion in result.criteria
                ] + [""]
        return "\n".join(lines).rstrip("\n") + "\n"


@dataclass
class PreferenceResult:
    """One pairwise verdict: which runtime's outcome better satisfied one
    evaluation's expectation, and why."""

    name: str
    preference: str  # "A", "B" or "tie"
    rationale: str
    expected: str
    outcome_a: str
    outcome_b: str


@dataclass
class Comparison:
    """A whole A/B run: every evaluation answered by both runtimes, with
    the pairwise preferences and the roll-up."""

    name_a: str
    name_b: str
    results: list[PreferenceResult] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "A": sum(1 for r in self.results if r.preference == "A"),
            "B": sum(1 for r in self.results if r.preference == "B"),
            "tie": sum(1 for r in self.results if r.preference == "tie"),
        }

    def render(self) -> str:
        counts = self.counts()
        lines = [
            f"# Comparison -- A: {self.name_a} vs B: {self.name_b}",
            "",
            f"Preferred A: {counts['A']} | Preferred B: {counts['B']} | Tie: {counts['tie']}",
        ]
        for result in self.results:
            lines += ["", f"## {result.name}{_VERDICT_SEPARATOR}{result.preference}", ""]
            lines += ["Expected:", quote(result.expected), ""]
            lines += [f"Outcome A ({self.name_a}):", quote(result.outcome_a), ""]
            lines += [f"Outcome B ({self.name_b}):", quote(result.outcome_b), ""]
            if result.rationale:
                lines += ["Why:", quote(result.rationale), ""]
        return "\n".join(lines).rstrip("\n") + "\n"


@dataclass
class Examiner:
    """An Examiner examines a runtime against an evaluations directory,
    and compares two runtimes over the same directory."""

    def examine(self, runtime: Any, directory: Union[str, Path]) -> Examination:
        directory = Path(directory)
        examination = Examination(
            runtime_name=runtime.name,
            prior_verdicts=self._prior_verdicts(directory / REPORT_FILENAME),
        )
        for path in self._evaluation_paths(directory):
            examination.results.append(self._examine_one(runtime, path))
        report = examination.render()
        (directory / REPORT_FILENAME).write_text(report, encoding="utf-8")
        self._archive(directory, report)
        return examination

    def compare(
        self, runtime_a: Any, runtime_b: Any, directory: Union[str, Path], judge: Any = None
    ) -> Comparison:
        """Both runtimes answer every evaluation; a pairwise judgment
        prefers A, B or tie per expectation. `judge` names the ModelBinding
        that judges -- pass a separate one to keep the referee independent
        of the contestants; by default A's (then B's) binding judges.
        Refuses without a model either way -- a preference judgment nobody
        made is never written down."""
        judge = self._judging_binding(runtime_a, runtime_b, judge)
        directory = Path(directory)
        comparison = Comparison(name_a=runtime_a.name, name_b=runtime_b.name)
        for path in self._evaluation_paths(directory):
            text = path.read_text(encoding="utf-8")
            expected_prose, expected_fields, criteria = self._expectation(text)
            expected = self._render_expectation(expected_prose, expected_fields, criteria)
            outcome_a = self._outcome_of(runtime_a, text)
            outcome_b = self._outcome_of(runtime_b, text)
            start = calls_so_far(judge.lm)
            preference, rationale = self._judge_preference(judge.lm, expected, outcome_a, outcome_b)
            comparison.results.append(
                PreferenceResult(
                    name=path.stem,
                    preference=preference,
                    rationale=rationale,
                    expected=expected,
                    outcome_a=outcome_a,
                    outcome_b=outcome_b,
                )
            )
            runtime_a.reasoning_log.record(
                stage="comparison",
                inputs={"evaluation": path.stem, "expected": expected, "a": runtime_a.name, "b": runtime_b.name},
                output=preference,
                rationale=rationale,
                model=judge.model_id,
                usage=usage_since(judge.lm, start),
            )
            runtime_a.reasoning_log.flush()
        (directory / COMPARISON_FILENAME).write_text(comparison.render(), encoding="utf-8")
        return comparison

    # -- the suite's files -----------------------------------------------------

    @staticmethod
    def _evaluation_paths(directory: Path) -> list[Path]:
        return [
            path
            for path in sorted(directory.glob("*.md"))
            if path.name not in (REPORT_FILENAME, COMPARISON_FILENAME)
        ]

    @staticmethod
    def _prior_verdicts(report_path: Path) -> Optional[dict[str, str]]:
        """The previous report's verdicts by evaluation name, read back
        through the Section codec -- or None when there is no history."""
        if not report_path.exists():
            return None
        verdicts: dict[str, str] = {}
        for section in parse_document(report_path.read_text(encoding="utf-8")).sections:
            name, separator, verdict = section.name.rpartition(_VERDICT_SEPARATOR)
            if separator and verdict.lower() in _KNOWN_VERDICTS:
                verdicts[name] = verdict
        return verdicts

    @staticmethod
    def _archive(directory: Path, report: str) -> None:
        """Append this run to `reports/` under a UTC timestamp, so the
        report history reads as regression history."""
        reports = directory / REPORTS_DIRNAME
        reports.mkdir(exist_ok=True)
        stamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
        path = reports / f"{stamp}.md"
        counter = 2
        while path.exists():
            path = reports / f"{stamp}-{counter}.md"
            counter += 1
        path.write_text(report, encoding="utf-8")

    # -- running one evaluation -------------------------------------------------

    def _examine_one(self, runtime: Any, path: Path) -> EvaluationResult:
        text = path.read_text(encoding="utf-8")
        expected_prose, expected_fields, criteria = self._expectation(text)
        graded_outcome = self._outcome_of(runtime, text)

        model_binding = getattr(runtime, "model_binding", None)
        grade_start = calls_so_far(getattr(model_binding, "lm", None))
        passed, rationale, criterion_results = self._grade(
            runtime, expected_prose, expected_fields, criteria, graded_outcome
        )
        result = EvaluationResult(
            name=path.stem,
            passed=passed,
            rationale=rationale,
            expected=self._render_expectation(expected_prose, expected_fields, criteria),
            outcome=graded_outcome,
            criteria=criterion_results,
        )
        runtime.reasoning_log.record(
            stage="evaluation",
            inputs={"evaluation": result.name, "expected": result.expected},
            output=result.verdict
            + (
                "; rubric: " + ", ".join(f"{c.verdict}: {c.text}" for c in criterion_results)
                if criterion_results
                else ""
            ),
            rationale=rationale,
            model="" if passed is None and not any(c.passed is not None for c in criterion_results)
            else model_name(model_binding),
            usage=usage_since(getattr(model_binding, "lm", None), grade_start),
        )
        runtime.reasoning_log.flush()
        return result

    def _outcome_of(self, runtime: Any, text: str) -> str:
        """One runtime's answer to one evaluation, with its delivered
        Data appended -- the outcome every grading judgment reads."""
        intent = Intent.from_markdown(text, skip_sections=("expected",))
        try:
            outcome = str(runtime.reason(intent))
            data = self._delivered_data(runtime)
        except PermissionError as blocked:
            # A refusal is an outcome an evaluation may well expect -- and
            # a blocked cycle delivered no data, whatever an earlier cycle
            # left in memory.
            outcome = f"BLOCKED: {blocked}"
            data = {}
        if not data:
            return outcome
        return outcome + "\n\nData:\n" + "\n".join(f"- {name}: {value}" for name, value in data.items())

    # -- reading the expectation ---------------------------------------------

    @staticmethod
    def _expectation(text: str) -> tuple[str, dict[str, Any], list[str]]:
        """The `## Expected` section, structured: prose, `field: value`
        bullets as deliverable expectations, and colon-less bullets as
        rubric criteria -- each of those graded separately."""
        prose = ""
        fields: dict[str, Any] = {}
        criteria: list[str] = []
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
                elif bullet.strip():
                    criteria.append(bullet.strip())
        return prose, fields, criteria

    @staticmethod
    def _render_expectation(prose: str, fields: dict[str, Any], criteria: list[str]) -> str:
        parts = [prose] if prose else []
        parts += [f"- {name}: {value}" for name, value in fields.items()]
        parts += [f"- {criterion}" for criterion in criteria]
        return "\n".join(parts) or "no expectation declared"

    # -- grading ---------------------------------------------------------------

    def _grade(
        self,
        runtime: Any,
        expected_prose: str,
        expected_fields: dict[str, Any],
        criteria: list[str],
        outcome: str,
    ) -> tuple[Optional[bool], str, list[CriterionResult]]:
        model_binding = getattr(runtime, "model_binding", None)
        lm = getattr(model_binding, "lm", None) if model_binding is not None else None
        if lm is not None:
            passed, rationale = self._grade_main_with_llm(expected_prose, expected_fields, outcome, lm)
            criterion_results = []
            for criterion in criteria:
                criterion_passed, criterion_rationale = self._grade_with_llm(criterion, outcome, lm)
                criterion_results.append(
                    CriterionResult(text=criterion, passed=criterion_passed, rationale=criterion_rationale)
                )
        else:
            passed, rationale = self._grade_main_offline(expected_prose, expected_fields, outcome, runtime)
            criterion_results = [
                CriterionResult(
                    text=criterion,
                    passed=None,
                    rationale="ungraded offline -- a rubric criterion needs a model to judge",
                )
                for criterion in criteria
            ]
        # A failed criterion fails the evaluation; all-passing criteria
        # carry an evaluation that had nothing else to grade.
        if any(c.passed is False for c in criterion_results):
            passed = False
        elif (
            passed is None
            and criterion_results
            and all(c.passed is True for c in criterion_results)
            and not expected_prose
            and not expected_fields
        ):
            passed = True
        return passed, rationale, criterion_results

    def _grade_main_with_llm(
        self, expected_prose: str, expected_fields: dict[str, Any], outcome: str, lm: Any
    ) -> tuple[Optional[bool], str]:
        if not expected_prose and not expected_fields:
            return None, "no prose or field expectation declared; graded on the rubric alone"
        expectation = self._render_expectation(expected_prose, expected_fields, [])
        return self._grade_with_llm(expectation, outcome, lm)

    def _grade_main_offline(
        self, expected_prose: str, expected_fields: dict[str, Any], outcome: str, runtime: Any
    ) -> tuple[Optional[bool], str]:
        if expected_fields:
            return self._grade_structurally(expected_fields, outcome, self._delivered_data(runtime))
        if expected_prose:
            return None, (
                "ungraded offline -- prose criteria need a model to judge; "
                "only field expectations are checked without one"
            )
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

    # -- pairwise preference ----------------------------------------------------

    @staticmethod
    def _judging_binding(runtime_a: Any, runtime_b: Any, judge: Any = None) -> Any:
        """The ModelBinding that judges the pairwise preference -- the
        dedicated `judge` when one is passed, else A's when bound, else
        B's. Refusing without one is the whole point: offline there is no
        honest way to prefer one outcome."""
        candidates = [judge] if judge is not None else [
            getattr(runtime, "model_binding", None) for runtime in (runtime_a, runtime_b)
        ]
        for binding in candidates:
            if binding is not None and getattr(binding, "lm", None) is None:
                binding.activate()
            if binding is not None and getattr(binding, "lm", None) is not None:
                return binding
        raise ValueError(
            "compare needs a bound model on one of the runtimes -- a pairwise "
            "preference is a judgment, and a judgment nobody made is never written down"
        )

    @staticmethod
    def _judge_preference(lm: Any, expected: str, outcome_a: str, outcome_b: str) -> tuple[str, str]:
        from .signatures import JudgePreference

        result = JudgePreference.run(lm, expected=expected, outcome_a=outcome_a, outcome_b=outcome_b)
        raw = normalize(str(result.preference))
        first = raw.split()[0] if raw.split() else ""
        if first in ("a", "outcome_a") or raw.startswith("outcome a"):
            return "A", str(result.rationale)
        if first in ("b", "outcome_b") or raw.startswith("outcome b"):
            return "B", str(result.rationale)
        if first in ("tie", "neither", "both", "equal", "even"):
            return "tie", str(result.rationale)
        # An unreadable preference is recorded as a tie with the raw reply
        # on the record -- never silently coerced to a winner.
        return "tie", f"unreadable preference '{result.preference}' treated as a tie -- {result.rationale}"
