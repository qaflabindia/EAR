"""judgment -- native structured prompting, EAR's replacement for DSPy.

A `Judgment` is a declared reasoning task: an instruction, the inputs the
model is given, and the outputs it must return. It renders those into a
prompt and parses the reply back into typed values -- with no framework
underneath, only the same markdown Section codec the whole package is
built on. That is the trick that keeps EAR dependency-free and consistent:
the model answers in markdown sections, exactly the format EAR authors,
persists and audits in, and `parse_document` reads them back.

Each output field declares a kind:

    "text"  -> the section's prose, verbatim (a decision, a rationale)
    "bool"  -> yes/no, read by the same `coerce` codec as everything else
    "list"  -> the section's bullets (or numbered items)
    "str"   -> a short one-line value

A missing or unparseable field degrades to a safe empty value rather than
raising, so a stage always gets a well-formed result object; the caller's
own fallbacks (which every judgment stage has) handle a genuinely unusable
answer. The result is a plain namespace whose attributes are the output
field names -- so a call site reads `result.decision`, `result.complies`,
just as it did before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Optional

from .section import argument_blocks, coerce, normalize, parse_document

_KIND_GUIDANCE = {
    "text": "the full text of the answer, as prose",
    "str": "a short one-line value",
    "bool": "answer with a single word: yes or no",
    "list": "one item per line, each line beginning with '- '",
    "map": (
        "one '- name: value' bullet per short argument, exactly as 'list' "
        "above; for a value that needs more than one line (source code, a "
        "whole file's content), write 'name:' alone on its line followed by "
        "the value as a blockquote -- every line, including a blank one, "
        "starting with '> ' (a lone '>' for a blank line). Never put a "
        "multi-line value on a single bullet line."
    ),
}


@dataclass
class Field:
    """One declared input or output of a Judgment."""

    name: str
    desc: str = ""
    kind: str = "text"

    @property
    def heading(self) -> str:
        return self.name.replace("_", " ")


@dataclass
class Judgment:
    """A declared reasoning task, rendered to a prompt and parsed from a
    markdown-section reply. Nothing here hardcodes an answer: the
    instruction and field descriptions frame the task, the model decides.

    `demos` are worked examples -- dicts of field name -> value covering
    the judgment's inputs and outputs -- rendered into the prompt in the
    same section shape the model must answer in. The Optimizer selects
    them from the runtime's own record (`select_demos`), and refined
    instructions/demos persist as markdown (`save_instructions`)."""

    instruction: str
    inputs: list[Field] = field(default_factory=list)
    outputs: list[Field] = field(default_factory=list)
    demos: list[dict[str, Any]] = field(default_factory=list)
    # The name of the one input whose value is volatile across otherwise-
    # identical calls (a growing tool-result log, say). When set, that input
    # renders last so everything before it is a stable, cacheable prefix, and
    # `run` passes that prefix to the LM as a provider-neutral cache hint.
    # None (the default) leaves rendering and calls exactly as before.
    cache_boundary: Optional[str] = None

    def run(self, lm: Any, **values: Any) -> SimpleNamespace:
        prompt, cache_prefix = self._render(values)
        # Pass the cache hint only when a boundary actually produced one, so an
        # LM whose `complete` predates the parameter is called exactly as before.
        if cache_prefix:
            reply = lm.complete(prompt, system=self.instruction, cache_prefix=cache_prefix)
        else:
            reply = lm.complete(prompt, system=self.instruction)
        return self.parse_reply(reply)

    # -- prompt ---------------------------------------------------------------

    def render_prompt(self, values: dict[str, Any]) -> str:
        return self._render(values)[0]

    def _render(self, values: dict[str, Any]) -> tuple[str, str]:
        """Render the prompt and, when a `cache_boundary` input is declared,
        the stable leading span that precedes its value (a byte-prefix of the
        prompt). With no boundary the span is empty and rendering is unchanged."""
        lines: list[str] = [self.instruction, ""]
        for number, demo in enumerate(self.demos, start=1):
            lines += [f"Worked example {number}:", ""]
            for spec in self.inputs + self.outputs:
                if spec.name in demo:
                    lines += [f"## {spec.heading}", "", str(demo[spec.name]).strip(), ""]
        if self.demos:
            lines += ["Now the task at hand:", ""]
        ordered = self.inputs
        if self.cache_boundary:
            ordered = [s for s in self.inputs if s.name != self.cache_boundary]
            ordered += [s for s in self.inputs if s.name == self.cache_boundary]
        cache_prefix = ""
        for spec in ordered:
            lines += [f"## {spec.heading}", ""]
            if self.cache_boundary and spec.name == self.cache_boundary and not cache_prefix:
                # Everything rendered up to (not including) this value repeats
                # across calls; the value itself and the footer are the tail.
                cache_prefix = "\n".join(lines)
            lines += [str(values.get(spec.name, "")).strip(), ""]
        lines += [
            "Respond using exactly the following markdown sections, each a"
            " level-2 heading (`## Name`) followed by its value. Add nothing"
            " outside these sections:",
            "",
        ]
        for spec in self.outputs:
            guidance = _KIND_GUIDANCE.get(spec.kind, _KIND_GUIDANCE["text"])
            detail = f"{spec.desc} -- {guidance}" if spec.desc else guidance
            lines += [f"## {spec.heading}", f"({detail})", ""]
        return "\n".join(lines), cache_prefix

    # -- parsing --------------------------------------------------------------

    def parse_reply(self, reply: str) -> SimpleNamespace:
        sections = {normalize(section.name): section for section in parse_document(reply).sections}
        result: dict[str, Any] = {}
        for spec in self.outputs:
            section = sections.get(normalize(spec.heading)) or sections.get(normalize(spec.name))
            result[spec.name] = self._read(spec, section)
        return SimpleNamespace(**result)

    @staticmethod
    def _read(spec: Field, section: Any) -> Any:
        if section is None:
            return {"list": [], "map": {}, "bool": False, "str": "", "text": ""}[spec.kind]
        if spec.kind == "map":
            return argument_blocks(section.lines)
        body = section.body()
        if spec.kind == "list":
            return list(body.bullets or body.numbered)
        text = "\n".join(filter(None, [body.prose] + body.bullets)).strip()
        if spec.kind == "bool":
            value = coerce(text.split()[0]) if text.split() else False
            return value is True
        if spec.kind == "str":
            return text.splitlines()[0].strip() if text else ""
        return text
