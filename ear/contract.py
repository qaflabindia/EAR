"""Contract -- a workflow's deliverable, declared in natural language and
honored as structured data.

Authored in `workflow.md` as a `### Deliverable` section directly beneath
the workflow it belongs to: the prose describes the deliverable, and each
bullet declares one field as `name: what it means`. No schema language, no
type annotations -- the meaning *is* the specification, and the same
`coerce` codec that types intent context values types the delivered
values.

At runtime the extraction is a model judgment: a DSPy signature is built
dynamically from the contract's fields (one output per field, the authored
meaning as its description), the model reads the prose decision and fills
the fields, and the model then *judges its own filling against the
meanings* via `JudgeContractConformance` -- with a deterministic fallback
that checks every field is present and non-empty, so conformance never
silently passes as a judgment it wasn't. With no model bound there is
nothing honest to extract, so the runtime records that the deliverable was
skipped rather than fabricating values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .section import coerce


@dataclass
class ContractField:
    """One declared deliverable field: its authored name and its meaning
    in plain English."""

    name: str
    meaning: str = ""

    @property
    def identifier(self) -> str:
        """The field name as a signature-safe identifier ('risk grade' ->
        'risk_grade'), derived by character mapping, never lossy for
        rendering -- the authored `name` is what documents show."""
        mapped = "".join(ch if ch.isalnum() else "_" for ch in self.name.strip().lower())
        return mapped or "field"


@dataclass
class Contract:
    """A Contract is the structured promise a workflow's decision must
    honor: named fields with plain-English meanings."""

    name: str
    description: str = ""
    fields: list[ContractField] = field(default_factory=list)

    def add_field(self, name: str, meaning: str = "") -> "Contract":
        self.fields.append(ContractField(name=name, meaning=meaning))
        return self

    def render_fields(self) -> str:
        return "\n".join(f"- {f.name}: {f.meaning or 'no meaning declared'}" for f in self.fields)

    # -- extraction (model judgment; no offline path exists honestly) -------

    def extract(self, decision: Any, intent: Any, model_binding: Any, hint: str = "") -> dict[str, Any]:
        """Fill the contract's fields from the prose decision with the
        bound model. Returns authored-name -> typed value. Raises nothing
        on a poor answer -- conformance is judged separately -- but
        requires a live model: with none bound the caller records a skip."""
        import dspy

        instructions = (
            "Fill the deliverable's fields from the decision, exactly as the "
            "decision states them -- never invent a value the decision does "
            "not support. Each field's description is its authored meaning."
        )
        if hint:
            instructions += f"\nA prior attempt was judged nonconforming: {hint}"
        signature_fields: dict[str, Any] = {
            "intent": dspy.InputField(desc="The intent the decision resolves"),
            "decision": dspy.InputField(desc="The prose decision to fill the fields from"),
        }
        for contract_field in self.fields:
            signature_fields[contract_field.identifier] = dspy.OutputField(
                desc=contract_field.meaning or contract_field.name
            )
        signature = dspy.Signature(signature_fields, instructions)
        extractor = dspy.Predict(signature)
        with dspy.context(lm=model_binding.lm):
            result = extractor(intent=str(intent), decision=str(decision))
        data: dict[str, Any] = {}
        for contract_field in self.fields:
            raw = getattr(result, contract_field.identifier, "")
            # One line per value: deliverable fields are facts, and the
            # bullet they are rendered as must round-trip through the
            # Section parser.
            data[contract_field.name] = coerce(" ".join(str(raw).split()))
        return data

    # -- conformance (model judgment with a structural fallback) ------------

    def judge(self, data: dict[str, Any], model_binding: Optional[Any] = None) -> tuple[bool, str]:
        """Judge whether the filled data honors the fields' meanings.
        Returns (conforms, rationale). The LLM judges meaning; the
        deterministic fallback checks structure only -- every declared
        field present with a non-empty value -- and says so."""
        if model_binding is not None and getattr(model_binding, "lm", None) is not None:
            return self._judge_with_llm(data, model_binding.lm)
        missing = [f.name for f in self.fields if str(data.get(f.name, "")).strip() == ""]
        if missing:
            return False, f"structural check only (no model bound): missing or empty fields: {', '.join(missing)}"
        return True, "structural check only (no model bound): every declared field is present and non-empty"

    def _judge_with_llm(self, data: dict[str, Any], lm: Any) -> tuple[bool, str]:
        import dspy

        from .signatures import JudgeContractConformance

        rendered = "\n".join(f"- {name}: {value}" for name, value in data.items())
        judge = dspy.Predict(JudgeContractConformance)
        with dspy.context(lm=lm):
            result = judge(contract=self.render_fields(), data=rendered)
        return bool(result.conforms), str(result.rationale)