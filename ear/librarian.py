"""Librarian -- research: retrieve the knowledge passages relevant to an
intent, with citations, before deliberation.

Retrieval is a runtime judgment: Knowledge's BM25 narrowing (over passage
text and, when the corpus is indexed, the model-written gists) brings the
corpus down to a handful of candidates -- retrieval mechanics, like the
Discoverer's keyword fallback -- and the model then judges which of those a careful
analyst would actually consult -- choosing none is a valid judgment, and
the model can only choose among the narrowed candidates, never invent a
passage. With no model bound, retrieval falls back to the structural
candidates alone and says so in the record.

What was consulted is first-class evidence: every research lands on the
ReasoningLog (stage `retrieval`) with its citations, the citations travel
into the decision's Evidence and its decision document (`## Sources`), and
the retrieved text reaches the Reasoner marked as reference material --
knowledge informs a decision, it never instructs the runtime.

A custom retriever -- anything with `retrieve(query) -> list[Passage]` --
can replace the structural narrowing; the model's relevance judgment and
the audit record stay the same either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .intent import Intent
from .knowledge import Knowledge, Passage
from .reasoning_log import calls_so_far, model_name, usage_since


@dataclass
class Research:
    """One cycle's retrieved knowledge: the passages judged relevant, the
    sources they cite, and the rendered block deliberation reads."""

    passages: list[Passage] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    rendered: str = ""


@dataclass
class Librarian:
    """A Librarian researches the runtime's Knowledge for one intent:
    structural narrowing, then the model's relevance judgment, always on
    the record."""

    knowledge: Optional[Knowledge] = None
    retriever: Optional[Any] = None
    candidate_limit: int = 6

    def research(self, runtime: Any, intent: Intent) -> Optional[Research]:
        candidates = self._candidates(intent)
        if not candidates:
            return None
        model_binding = getattr(runtime, "model_binding", None)
        start = calls_so_far(getattr(model_binding, "lm", None))
        if model_binding is not None and getattr(model_binding, "lm", None) is not None:
            chosen, rationale = self._judge_with_llm(intent, candidates, model_binding.lm)
        else:
            # Structural retrieval only -- and the record says so.
            chosen = candidates[:3]
            rationale = "structural retrieval only (no model bound): best BM25 candidates included"
        research = Research(
            passages=chosen,
            citations=[passage.source for passage in chosen],
            rendered="\n\n".join(passage.render() for passage in chosen),
        )
        log = getattr(runtime, "reasoning_log", None)
        if log is not None:
            log.record(
                stage="retrieval",
                inputs={
                    "intent": intent.text,
                    "candidates": [passage.source for passage in candidates],
                    "narrowing": self._narrowing_basis(),
                },
                output="; ".join(research.citations) or "nothing judged relevant",
                rationale=rationale,
                model=model_name(model_binding),
                usage=usage_since(getattr(model_binding, "lm", None), start),
            )
        return research

    def _narrowing_basis(self) -> str:
        """How the candidates were narrowed, for the retrieval record --
        a custom retriever's own judgment, or Knowledge's BM25 with or
        without the gist index."""
        if self.retriever is not None:
            return f"custom retriever ({type(self.retriever).__name__})"
        if self.knowledge is not None and len(self.knowledge):
            return self.knowledge.narrowing()
        return "no corpus"

    def _candidates(self, intent: Intent) -> list[Passage]:
        if self.retriever is not None:
            retrieved = self.retriever.retrieve(intent.text)
            return [passage for passage in retrieved if isinstance(passage, Passage)][: self.candidate_limit]
        if self.knowledge is not None and len(self.knowledge):
            return self.knowledge.candidates(intent.text, self.candidate_limit)
        return []

    @staticmethod
    def _judge_with_llm(intent: Intent, candidates: list[Passage], lm: Any) -> tuple[list[Passage], str]:
        from .signatures import SelectRelevantPassages

        numbered = "\n\n".join(f"{number}. {passage.render()}" for number, passage in enumerate(candidates, start=1))
        result = SelectRelevantPassages.run(lm, intent_text=intent.text, passages=numbered)
        chosen: list[Passage] = []
        for number in result.relevant_numbers:
            try:
                index = int(number) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(candidates) and candidates[index] not in chosen:
                chosen.append(candidates[index])
        return chosen, str(result.rationale)