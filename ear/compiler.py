"""StackCompiler -- compile one acc-skills command centre into a complete
EAR markdown stack that `load_runtime` reads unchanged.

Phase 1 (`ear/enterprise.py`) bound a centre's *constitution* onto a
runtime as policies. Phase 2 compiles the *whole centre* into a stack: an
operational command centre becomes a directory of the same six
natural-language files an author would otherwise write by hand, so a centre
runs as a first-class EAR runtime -- personas, skills, workflows, knowledge,
governance and tenant, all authored in English, no code.

The mapping (framework architecture section 3.5):

    acc-skills artifact                  EAR stack file
    -----------------------------------  ------------------------------------
    SKILL.md mission prose               persona.md   (the persona's instructions)
    SKILL.md `## Capabilities`           skills.md    (one skill per capability)
    SKILL.md `## Procedures`             workflow.md  (steps delegating to the persona)
    (a process wrapping the workflows)   process.md   (the runtime's title)
    references/constitutional_rules.md   policy.md    (via Constitution.to_policy_markdown)
    references/*.md (the rest)           knowledge/   (documents the Librarian cites)
    SKILL.md frontmatter org context     tenant.md    (org id, fiscal year)
    operating strategy                   memory.md    (knowledge sources, audit trail)

Two invariants from the architecture hold here. **Nothing an author wrote
is silently dropped**: a `##` section the compiler does not recognize as
capabilities or procedures folds into the persona's instructions rather
than vanishing, and `compile()` can verify the result by loading it (every
cross-reference resolves or the loader fails loudly). **English stays the
source of truth**: the output is markdown an author can read, diff and edit
-- the compiler is a starting point, not a lock-in.

Standard library only, like the rest of the package.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from .enterprise import CommandCentre, Constitution
from .section import normalize

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")

# Words dropped when abbreviating a centre's name into a short label used
# for the persona and process ("Agentic Finance Command Centre" -> "Finance").
_LABEL_NOISE = {"agentic", "command", "centre", "center", "the", "of", "and"}

# The `##` sections the compiler consumes structurally. Anything else is
# domain prose folded into the persona so it is never dropped.
_CAPABILITY_HEADINGS = {"capabilities", "skills", "abilities"}
_PROCEDURE_HEADINGS = {"procedures", "procedure", "workflows", "workflow", "playbooks"}


# --------------------------------------------------------------------------
# A tiny level-aware heading tree -- the flat Section codec loses the
# `##` / `###` nesting the compiler needs, so it is rebuilt here.
# --------------------------------------------------------------------------


@dataclass
class _Node:
    level: int
    title: str
    body_lines: list[str] = field(default_factory=list)
    children: list["_Node"] = field(default_factory=list)

    def prose(self) -> str:
        return "\n".join(self.body_lines).strip()


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Peel a leading `---`-fenced YAML-ish frontmatter block off the top of
    a file. Only simple `key: value` lines are read -- enough for a centre's
    name, slug, plane and org context -- and the remaining body is returned
    verbatim. No frontmatter yields an empty mapping and the whole text."""
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    fields: dict[str, str] = {}
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            body = "\n".join(lines[index + 1 :])
            return fields, body
        if ":" in lines[index]:
            key, _, value = lines[index].partition(":")
            fields[normalize(key)] = value.strip().strip("\"'")
    # An unterminated fence is not frontmatter -- treat the whole file as body.
    return {}, text


def _tree(text: str) -> tuple[str, list[_Node]]:
    """Parse markdown into (preamble, roots) where each root is a heading
    node carrying its own body prose and nested child headings. Preamble is
    the prose before the first heading."""
    preamble: list[str] = []
    roots: list[_Node] = []
    stack: list[_Node] = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        match = _HEADING.match(raw)
        if not match:
            (stack[-1].body_lines if stack else preamble).append(raw)
            continue
        node = _Node(level=len(match.group(1)), title=match.group(2).strip())
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)
    return "\n".join(preamble).strip(), roots


def _mission_and_sections(text: str) -> tuple[str, list[_Node]]:
    """The centre's mission prose and its top-level sections. A single
    level-1 `#` title carries the mission in its own body and the `##`
    sections as its children; without one, the preamble is the mission and
    the level-2 headings are the sections."""
    preamble, roots = _tree(text)
    level_ones = [node for node in roots if node.level == 1]
    if len(level_ones) == 1:
        title = level_ones[0]
        return title.prose() or preamble, title.children
    return preamble, [node for node in roots if node.level >= 2]


def _short_label(name: str) -> str:
    words = [word for word in re.split(r"[\s_-]+", name) if word and normalize(word) not in _LABEL_NOISE]
    # Drop a trailing acronym in parentheses, e.g. "Finance (AFCC)" -> "Finance".
    words = [word for word in words if not (word.startswith("(") and word.endswith(")"))]
    return " ".join(words).strip() or name.strip()


def _strip_marker(line: str) -> str:
    """A body line with any leading bullet/number marker removed and
    trimmed. A blank or marker-only line yields the empty string."""
    stripped = line.strip()
    if not stripped:
        return ""
    bullet = _BULLET.match(line)
    if bullet:
        return bullet.group(1).strip()
    numbered = _NUMBERED.match(line)
    if numbered:
        return numbered.group(1).strip()
    return stripped


def _numbered_or_bulleted(prose_lines: list[str]) -> list[str]:
    """Steps read from a section body: numbered items first, else bullets.
    Multi-line wrapped items are not expected in a compiled procedure, so
    each matching line is one step."""
    numbered = [m.group(1).strip() for line in prose_lines if (m := _NUMBERED.match(line))]
    if numbered:
        return numbered
    return [m.group(1).strip() for line in prose_lines if (m := _BULLET.match(line))]


# --------------------------------------------------------------------------
# The compiled stack.
# --------------------------------------------------------------------------


@dataclass
class CompiledStack:
    """The result of compiling a command centre: where the stack was
    written, the files produced, and a mapping report naming which centre
    artifact produced each file -- so a reviewer sees nothing was dropped."""

    directory: Path
    files: list[str] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)
    skills: list[str] = field(default_factory=list)
    workflows: list[str] = field(default_factory=list)
    knowledge: list[str] = field(default_factory=list)

    def load(self, name: Optional[str] = None):
        """Load the compiled stack as a Runtime, exactly as any authored
        stack loads. Local import: the loader imports widely, so a
        module-level import here would risk a cycle."""
        from .loader import load_runtime

        return load_runtime(self.directory, name=name)

    def summary(self) -> str:
        return (
            f"compiled to {self.directory}: {len(self.files)} files, "
            f"{len(self.skills)} skills, {len(self.workflows)} workflows, "
            f"{len(self.knowledge)} knowledge docs"
        )


@dataclass
class StackCompiler:
    """Compile a single command centre directory into an EAR stack.

    Reuses `CommandCentre.load` for the centre's name, plane and
    constitution (Phase 1), then compiles the SKILL.md structure and
    references into the remaining stack files."""

    centre: CommandCentre

    @classmethod
    def from_directory(cls, directory: Union[str, Path]) -> "StackCompiler":
        return cls(centre=CommandCentre.load(directory))

    def compile(self, output_dir: Union[str, Path], verify: bool = True) -> CompiledStack:
        """Compile the centre into `output_dir` and return a CompiledStack.
        With `verify` (the default), the written stack is loaded once so any
        unresolved cross-reference fails loudly here rather than at first
        use -- the loader's own contract, applied to compilation."""
        source = self.centre.directory
        if source is None:
            raise ValueError("StackCompiler needs a centre loaded from a directory")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        skill_text = (source / "SKILL.md").read_text(encoding="utf-8") if (source / "SKILL.md").exists() else ""
        frontmatter, body = _split_frontmatter(skill_text)
        mission, sections = _mission_and_sections(body)

        capabilities = self._capabilities(sections)
        procedures = self._procedures(sections)
        folded = self._folded_prose(sections)

        short = _short_label(self.centre.name)
        persona_name = frontmatter.get("persona") or f"{short} Operator"

        stack = CompiledStack(directory=out)
        self._write_skills(out, capabilities, stack)
        self._write_persona(out, persona_name, mission, folded, capabilities, stack)
        self._write_policy(out, stack)
        self._write_workflows(out, procedures, persona_name, capabilities, stack)
        self._write_process(out, short, procedures, persona_name, stack)
        self._write_knowledge(out, source, stack)
        self._write_tenant(out, frontmatter, stack)
        self._write_memory(out, short, stack)

        if verify:
            stack.load()
        return stack

    # -- reading the centre -------------------------------------------------

    @staticmethod
    def _capabilities(sections: list[_Node]) -> list[tuple[str, str]]:
        """(skill name, prompt) pairs from a `## Capabilities` section's
        `###` children, or from its bullets when it has no children."""
        for section in sections:
            if normalize(section.title) in _CAPABILITY_HEADINGS:
                if section.children:
                    return [(child.title, child.prose()) for child in section.children]
                pairs = []
                for bullet in _numbered_or_bulleted(section.body_lines):
                    name, _, prompt = bullet.partition(":")
                    pairs.append((name.strip(), prompt.strip() or name.strip()))
                return pairs
        return []

    @staticmethod
    def _procedures(sections: list[_Node]) -> list[tuple[str, list[str]]]:
        """(workflow name, steps) pairs from a `## Procedures` section's
        `###` children."""
        for section in sections:
            if normalize(section.title) in _PROCEDURE_HEADINGS:
                procedures = []
                for child in section.children:
                    steps = _numbered_or_bulleted(child.body_lines)
                    if steps:
                        procedures.append((child.title, steps))
                # A procedures section written as one flat numbered list, no
                # sub-workflows, becomes a single workflow.
                if not procedures:
                    steps = _numbered_or_bulleted(section.body_lines)
                    if steps:
                        procedures.append((f"{section.title} Workflow", steps))
                return procedures
        return []

    @staticmethod
    def _folded_prose(sections: list[_Node]) -> str:
        """Domain prose from sections the compiler does not consume
        structurally (triggers, scope, notes) -- folded into the persona so
        nothing an author wrote is dropped.

        Rendered as plain paragraphs, never as markdown bullets: a bullet in
        a persona's instructions is read by the loader as a *skill
        reference*, so folded content is flattened to sentences to keep it
        instructions, not accidental cross-references."""
        parts = []
        for section in sections:
            key = normalize(section.title)
            if key in _CAPABILITY_HEADINGS or key in _PROCEDURE_HEADINGS:
                continue
            items = [_strip_marker(line) for line in section.body_lines if _strip_marker(line)]
            items += [
                f"{child.title.rstrip('.')}: {child.prose()}".strip().rstrip(":")
                for child in section.children
                if child.title
            ]
            body = " ".join(item.rstrip(".") + "." for item in items if item)
            if body:
                parts.append(f"{section.title.rstrip(':')}: {body}")
        return "\n\n".join(parts)

    # -- writing the stack --------------------------------------------------

    def _write_skills(self, out: Path, capabilities: list[tuple[str, str]], stack: CompiledStack) -> None:
        lines = [
            "# Skills",
            "",
            "Prompts stacked into skills, compiled from the command centre's",
            "capabilities. Each heading names a skill; the prose is the prompt.",
            "",
        ]
        for name, prompt in capabilities:
            lines += [f"## {name}", "", prompt or f"Carry out the {name} capability.", ""]
            stack.skills.append(name)
        self._emit(out, "skills.md", lines, stack, mapping="SKILL.md ## Capabilities")

    def _write_persona(
        self,
        out: Path,
        persona_name: str,
        mission: str,
        folded: str,
        capabilities: list[tuple[str, str]],
        stack: CompiledStack,
    ) -> None:
        instructions = "\n\n".join(filter(None, [mission, folded])) or (
            f"Operate the {self.centre.name} conservatively and name the decisive "
            "factor behind every decision."
        )
        lines = ["# Personas", "", f"## {persona_name}", "", instructions, ""]
        if capabilities:
            skill_refs = ", ".join(name for name, _ in capabilities)
            lines.append(f"Skills: {skill_refs}")
            lines.append("")
        self._emit(out, "persona.md", lines, stack, mapping="SKILL.md mission + folded prose")

    def _write_policy(self, out: Path, stack: CompiledStack) -> None:
        constitution: Constitution = self.centre.constitution
        text = constitution.to_policy_markdown()
        (out / "policy.md").write_text(text, encoding="utf-8")
        stack.files.append("policy.md")
        stack.mapping["policy.md"] = "references/constitutional_rules.md"

    def _write_workflows(
        self,
        out: Path,
        procedures: list[tuple[str, list[str]]],
        persona_name: str,
        capabilities: list[tuple[str, str]],
        stack: CompiledStack,
    ) -> None:
        lines = [
            "# Workflows",
            "",
            "Steps stacked into workflows, compiled from the centre's procedures.",
            "Each step is delegated to the centre's persona; the runtime-scoped",
            "constitution in policy.md governs every step.",
            "",
        ]
        if not procedures:
            # A centre with no procedures still runs: one workflow that
            # delegates the whole request to the persona, every capability
            # available to it.
            steps = [
                f"Apply the {name} capability where the request calls for it." for name, _ in capabilities
            ] or [f"Handle the request as the {persona_name}."]
            procedures = [(f"{_short_label(self.centre.name)} Workflow", steps)]
        for name, steps in procedures:
            lines.append(f"## {name}")
            lines.append("")
            for index, step in enumerate(steps, start=1):
                lines.append(f"{index}. {step} ({persona_name})")
            lines.append("")
            stack.workflows.append(name)
        self._emit(out, "workflow.md", lines, stack, mapping="SKILL.md ## Procedures")

    def _write_process(
        self,
        out: Path,
        short: str,
        procedures: list[tuple[str, list[str]]],
        persona_name: str,
        stack: CompiledStack,
    ) -> None:
        workflow_names = stack.workflows or [f"{short} Workflow"]
        lines = [
            f"# {short} Enterprise Runtime",
            "",
            "Workflows stacked into a process, and the process stacked into the",
            "runtime this file's title names.",
            "",
            f"## {short} Operations",
            "",
            f"Runs the {self.centre.name}'s procedures end to end.",
            "",
            f"Workflows: {', '.join(workflow_names)}",
            "",
        ]
        self._emit(out, "process.md", lines, stack, mapping="a process wrapping the workflows")

    def _write_knowledge(self, out: Path, source: Path, stack: CompiledStack) -> None:
        references = source / "references"
        if not references.exists():
            return
        knowledge_dir = out / "knowledge"
        for path in sorted(references.glob("*.md")):
            if path.name == "constitutional_rules.md":
                continue  # compiled to policy.md, not knowledge
            knowledge_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, knowledge_dir / path.name)
            stack.knowledge.append(path.name)
        if stack.knowledge:
            stack.files.extend(f"knowledge/{name}" for name in stack.knowledge)
            stack.mapping["knowledge/"] = "references/*.md (non-constitutional)"

    def _write_tenant(self, out: Path, frontmatter: dict[str, str], stack: CompiledStack) -> None:
        org = frontmatter.get("org") or frontmatter.get("org id")
        if not org:
            return  # no org context -> the default tenant, no tenant.md
        name = frontmatter.get("name") or self.centre.name
        lines = ["# Tenant", "", f"## {name}", "", f"Org id: {org}"]
        if frontmatter.get("fiscal year start"):
            lines.append(f"Fiscal year start: {frontmatter['fiscal year start']}")
        if frontmatter.get("fiscal year end"):
            lines.append(f"Fiscal year end: {frontmatter['fiscal year end']}")
        if frontmatter.get("timezone"):
            lines.append(f"Timezone: {frontmatter['timezone']}")
        lines.append("")
        self._emit(out, "tenant.md", lines, stack, mapping="SKILL.md frontmatter org context")

    def _write_memory(self, out: Path, short: str, stack: CompiledStack) -> None:
        lines = [
            "# Memory & Strategy",
            "",
            "The compiled centre's operating strategy, in plain English.",
            "",
            "## Reasoning Audit Trail",
            "",
            "Log every reasoning step -- each policy judgment with its rationale,",
            "process discovery, deliberation and explanation -- to `.ear/reasoning.md`,",
            "append-only across sessions, so the whole centre writes to one spine.",
            "",
            "## Skills Discovery",
            "",
            "Rank processes by reading their descriptions against the intent, most",
            "relevant first, and prefer a single best-fit process over a broad sweep.",
            "",
        ]
        if stack.knowledge:
            lines += [
                "## Knowledge",
                "",
                f"The reference material the Librarian may consult and cite while the",
                f"{short} runtime reasons; sources resolve relative to this stack directory.",
                "",
            ]
            for name in stack.knowledge:
                label = normalize(Path(name).stem).replace("-", " ")
                lines.append(f"- {label}: `knowledge/{name}`")
            lines.append("")
        self._emit(out, "memory.md", lines, stack, mapping="operating strategy")

    @staticmethod
    def _emit(out: Path, filename: str, lines: list[str], stack: CompiledStack, mapping: str) -> None:
        text = "\n".join(lines).rstrip() + "\n"
        (out / filename).write_text(text, encoding="utf-8")
        stack.files.append(filename)
        stack.mapping[filename] = mapping


def compile_command_centre(
    directory: Union[str, Path], output_dir: Union[str, Path], verify: bool = True
) -> CompiledStack:
    """Compile the command centre at `directory` into an EAR stack at
    `output_dir`. The one-call entry point over `StackCompiler`."""
    return StackCompiler.from_directory(directory).compile(output_dir, verify=verify)
