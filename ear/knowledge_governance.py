"""Knowledge governance -- AKC: nothing enters Knowledge ungoverned.

The cognitive plane's ingestion gate (framework architecture §6). A claim
does not reach the runtime's `Knowledge` corpus by being written to a file;
it is *admitted* -- validated for form and source, scored epistemically, and
checked for contradiction against what the base already holds -- and only
then chunked in via `Knowledge.add_document`. Contradiction and retirement
are lifecycle events that remove or supersede passages. Every admission,
rejection and retirement lands on the one audit spine (stage `ingest`).

Reason-first, above a deterministic floor -- the same division EAR draws
everywhere. With a model bound, the model judges the claim's epistemic
quality (`JudgeKnowledgeAdmission`) and any contradiction
(`JudgeContradiction`); offline, a deterministic structural floor stands in
and says so: a claim with no source or no real content is refused, and a
short claim scores below a long, specific one. A judgment nobody made is
never written down as one -- an offline admission is labelled a fallback,
and an epistemic score nobody computed is the floor's heuristic, not an
invented certainty.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .reasoning_log import model_name

ADMISSION_THRESHOLD = 0.5


@dataclass
class Admission:
    """The verdict on one claim: whether it was admitted, its epistemic
    score, why, and the passage it contradicted (if any)."""

    admitted: bool
    score: float
    reason: str
    contradiction: str = ""
    basis: str = ""


@dataclass
class KnowledgeGate:
    """AKC -- the governed door into `Knowledge`. `admit` is the only
    sanctioned way a claim enters the corpus; `retire` and `supersede` are
    the lifecycle events that remove or replace what is already there."""

    threshold: float = ADMISSION_THRESHOLD

    def admit(
        self,
        runtime: Any,
        source_name: str,
        filename: str,
        text: str,
        model_binding: Any = None,
        allow_contradiction: bool = False,
    ) -> Admission:
        """Admit a claim into the runtime's knowledge, if it passes. Judges
        epistemic quality and contradiction, adds the document only on a
        pass, and records the outcome on the trail. A contradiction refuses
        admission unless `allow_contradiction` (then the contradicted passage
        is superseded)."""
        knowledge = self._knowledge(runtime)
        binding = model_binding if model_binding is not None else getattr(runtime, "model_binding", None)
        lm = getattr(binding, "lm", None) if binding is not None else None
        if binding is not None:
            binding.activate()
            lm = getattr(binding, "lm", None)

        if lm is not None:
            admission = self._judge(lm, knowledge, source_name, text)
        else:
            admission = self._floor(source_name, text)

        contradicted = admission.contradiction
        if contradicted and not allow_contradiction:
            admission.admitted = False
            admission.reason = f"contradicts existing passage '{contradicted}' -- refused (supersede explicitly to replace it)"
        elif contradicted and allow_contradiction:
            self.retire(runtime, contradicted)

        if admission.admitted:
            knowledge.add_document(source_name, filename, text)

        log = getattr(runtime, "reasoning_log", None)
        if log is not None:
            verdict = "ADMITTED" if admission.admitted else "REFUSED"
            log.record(
                stage="ingest",
                inputs={
                    "source": source_name,
                    "filename": filename,
                    "score": admission.score,
                    "contradiction": contradicted,
                    "basis": admission.basis,
                },
                output=f"{verdict} (score {admission.score:.2f}) -- {source_name}/{filename}",
                rationale=admission.reason,
                model=model_name(binding),
            )
        return admission

    def retire(self, runtime: Any, source_substring: str) -> int:
        """Remove every passage whose source names `source_substring` -- the
        retirement lifecycle event. Returns how many were retired, on the
        record."""
        knowledge = self._knowledge(runtime)
        wanted = source_substring.lower()
        before = len(knowledge.passages)
        knowledge.passages = [p for p in knowledge.passages if wanted not in p.source.lower()]
        retired = before - len(knowledge.passages)
        log = getattr(runtime, "reasoning_log", None)
        if log is not None and retired:
            log.record(
                stage="ingest",
                inputs={"retired_source": source_substring, "retired": retired},
                output=f"RETIRED {retired} passage(s) matching '{source_substring}'",
                rationale="knowledge lifecycle: a superseded or withdrawn claim is retired, not left to mislead",
            )
        return retired

    def supersede(
        self,
        runtime: Any,
        old_source_substring: str,
        source_name: str,
        filename: str,
        text: str,
        model_binding: Any = None,
    ) -> Admission:
        """Retire the passages matching `old_source_substring` and admit the
        new claim in their place -- one lifecycle event, on the record."""
        self.retire(runtime, old_source_substring)
        return self.admit(runtime, source_name, filename, text, model_binding=model_binding, allow_contradiction=True)

    # -- the two paths ------------------------------------------------------

    def _judge(self, lm: Any, knowledge: Any, source_name: str, text: str) -> Admission:
        from .signatures import JudgeContradiction, JudgeKnowledgeAdmission

        existing = "; ".join(p.source for p in knowledge.passages[:20]) or "(empty)"
        result = JudgeKnowledgeAdmission.run(lm, claim=text, source=source_name or "(unstated)", existing=existing)
        score = self._read_score(getattr(result, "score", ""))
        # The model's admission decision is authoritative -- the score is
        # recorded, not used to overrule the judgment. (The threshold governs
        # only the deterministic offline floor, where no model judged.) A
        # hardcoded number second-guessing the model would be the very
        # breach reason-first governance forbids.
        admit = bool(getattr(result, "admit", False))
        rationale = str(getattr(result, "rationale", "") or "judged by the model")

        contradiction = ""
        if admit and knowledge.passages:
            passages = "\n".join(f"[{p.source}] {p.text[:200]}" for p in knowledge.passages[:20])
            conflict = JudgeContradiction.run(lm, new_claim=text, existing_passages=passages)
            if bool(getattr(conflict, "contradicts", False)):
                contradiction = str(getattr(conflict, "passage", "") or "an existing passage")
        return Admission(
            admitted=admit,
            score=score,
            reason=rationale,
            contradiction=contradiction,
            basis="judged by the model",
        )

    def _floor(self, source_name: str, text: str) -> Admission:
        """The deterministic structural floor, offline: a claim needs a
        source and real content; the score is a transparent heuristic of
        length and sourcing, never an invented certainty."""
        content = text.strip()
        if not content:
            return Admission(False, 0.0, "no content to admit", basis="deterministic floor (no model)")
        if not source_name.strip():
            return Admission(
                False, 0.2, "no source declared -- an unsourced claim is not admitted offline",
                basis="deterministic floor (no model)",
            )
        # A longer, sourced claim scores higher than a terse one -- a coarse
        # stand-in for epistemic quality, and labelled as exactly that: a
        # substantive sourced claim (~20+ words) clears the floor, a terse
        # one does not.
        words = len(content.split())
        score = min(0.9, 0.4 + words / 200.0)
        admitted = score >= self.threshold
        reason = (
            "sourced and substantive enough for the offline floor"
            if admitted
            else "too terse to admit without a model to judge it"
        )
        return Admission(admitted, score, reason, basis="deterministic floor (no model)")

    @staticmethod
    def _read_score(value: Any) -> float:
        try:
            score = float(str(value).strip().split()[0])
        except (ValueError, IndexError):
            return 0.5
        return max(0.0, min(1.0, score))

    @staticmethod
    def _knowledge(runtime: Any) -> Any:
        """The runtime's Knowledge corpus, created and attached if absent so
        a runtime that declared no knowledge sources can still be governed
        into holding some."""
        from .knowledge import Knowledge

        librarian = getattr(runtime, "librarian", None)
        if librarian is None:
            raise ValueError("runtime has no Librarian to hold Knowledge")
        if getattr(librarian, "knowledge", None) is None:
            librarian.knowledge = Knowledge()
        return librarian.knowledge
