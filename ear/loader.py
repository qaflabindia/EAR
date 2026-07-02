"""Loader -- build a whole Runtime from a directory of stacked markdown
files, written entirely in natural language. The user writes no code:

    skills.md    prompts stacked into Skills      (heading = skill, prose = prompt)
    persona.md   skills stacked into Personas     (prose = instructions, `Skills:` = stack)
    workflow.md  steps stacked into Workflows     (numbered steps, `(Persona)` delegates)
    process.md   workflows stacked into Processes (prose = description, `Workflows:` = stack)
    policy.md    governance, risk and controls    (prose = statement, `Applies to:` = scope)
    memory.md    the operating Strategy           (context history, cross-session data,
                                                   subagent spawning, model selection, MCP,
                                                   tools, skills discovery, ontology)

`load_runtime(directory)` reads whichever of those files exist and stacks
them into one Runtime: skills into personas, steps into workflows (each
delegated to a persona), workflows into processes, processes into the
runtime, policies onto the runtime or onto the workflows they name, and the
memory.md strategy onto the runtime's memory, session store, spawner and
model binding. Every cross-reference is by name, resolved case- and
punctuation-insensitively, and an unresolved reference fails loudly with
the list of known names -- nothing an author writes is silently dropped.

Nothing here decides anything: the loader is structural (like the Selector
and Composer), and every judgment in the loaded runtime still happens in
natural language against the active ModelBinding at reasoning time.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .contract import Contract
from .knowledge import Knowledge
from .persona import Persona
from .policy import Policy
from .process import Process
from .runtime import Runtime
from .section import Document, Section, normalize, parse_document
from .session_store import SessionStore
from .skill import Skill
from .strategy import Strategy
from .workflow import Workflow

_FILE_CANDIDATES = {
    "skills": ("skills.md", "skill.md"),
    "personas": ("persona.md", "personas.md"),
    "workflows": ("workflow.md", "workflows.md"),
    "processes": ("process.md", "processes.md"),
    "policies": ("policy.md", "policies.md"),
    "memory": ("memory.md",),
}

_RUNTIME_SCOPES = {"runtime", "the runtime", "all", "everything", "global", "the whole runtime"}

_DELEGATION_PATTERNS = (
    re.compile(r"\((?P<who>[^()]+)\)\s*$"),
    re.compile(r"\[(?P<who>[^\[\]]+)\]\s*$"),
    re.compile(r"\s(?:--|—|–)\s*(?P<who>[^—–]+?)\s*$"),
)
_DELEGATION_PREFIX = re.compile(r"^(?:delegated?\s+to|persona|by)\s*:?\s*", re.IGNORECASE)


@dataclass
class Loader:
    """Loads a stacked-markdown directory into a Runtime."""

    directory: Union[str, Path]

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)

    def load(self, name: Optional[str] = None) -> Runtime:
        skills_doc = self._parse("skills")
        personas_doc = self._parse("personas")
        policies_doc = self._parse("policies")
        workflows_doc = self._parse("workflows")
        processes_doc = self._parse("processes")
        memory_text = self._read("memory")

        skills = self._load_skills(skills_doc)
        personas = self._load_personas(personas_doc, skills)
        policies, policy_scopes = self._load_policies(policies_doc)
        workflows = self._load_workflows(workflows_doc, personas, policies)
        processes, referenced = self._load_processes(processes_doc, workflows)

        runtime = Runtime(name=name or processes_doc.title or self.directory.name)
        for process in processes:
            runtime.add_process(process)
        # A workflow no process references is still the author's work: wrap
        # it in a process of its own rather than dropping it.
        for key, workflow in workflows.items():
            if key not in referenced:
                orphan = Process(name=workflow.name, description=f"Runs the {workflow.name} workflow.")
                orphan.add_workflow(workflow)
                runtime.add_process(orphan)

        self._apply_policy_scopes(runtime, policies, policy_scopes, workflows)
        self._apply_strategy(runtime, Strategy.from_markdown(memory_text))
        return runtime

    # -- reading files ------------------------------------------------------

    def _read(self, kind: str) -> str:
        for filename in _FILE_CANDIDATES[kind]:
            path = self.directory / filename
            if path.exists():
                return path.read_text(encoding="utf-8")
        return ""

    def _parse(self, kind: str) -> Document:
        return parse_document(self._read(kind))

    # -- stacking each layer ------------------------------------------------

    def _load_skills(self, document: Document) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        for section in document.sections:
            body = section.body(field_keys=("description",))
            prompt = "\n".join(filter(None, [body.prose] + [f"- {bullet}" for bullet in body.bullets]))
            skills[normalize(section.name)] = Skill(
                name=section.name,
                prompt=prompt,
                description=body.field("description"),
            )
        return skills

    def _load_personas(self, document: Document, skills: dict[str, Skill]) -> dict[str, Persona]:
        personas: dict[str, Persona] = {}
        for section in document.sections:
            body = section.body(field_keys=("skills", "skill"))
            persona = Persona(name=section.name, instructions=body.prose)
            for reference in _split_references(body.field("skills", "skill")):
                persona.add_skill(_resolve(skills, reference, "skill"))
            for bullet in body.bullets:
                key = normalize(bullet)
                if key in skills:
                    persona.add_skill(skills[key])
                elif ":" in bullet:
                    # An inline skill, defined right where it is stacked.
                    inline_name, _, inline_prompt = bullet.partition(":")
                    persona.add_skill(Skill(name=inline_name.strip(), prompt=inline_prompt.strip()))
                else:
                    _resolve(skills, bullet, "skill")  # raises with the known names
            personas[normalize(section.name)] = persona
        return personas

    def _load_policies(self, document: Document) -> tuple[dict[str, Policy], dict[str, str]]:
        policies: dict[str, Policy] = {}
        scopes: dict[str, str] = {}
        for section in document.sections:
            body = section.body(
                field_keys=("fallback", "fallback expression", "applies to", "applies", "scope", "approval")
            )
            statement = "\n".join(filter(None, [body.prose] + [f"- {bullet}" for bullet in body.bullets]))
            key = normalize(section.name)
            policies[key] = Policy(
                name=section.name,
                statement=statement,
                fallback_expression=body.field("fallback", "fallback expression"),
                approval_required=self._read_approval_field(section.name, body.field("approval")),
            )
            scopes[key] = body.field("applies to", "applies", "scope")
        return policies, scopes

    @staticmethod
    def _read_approval_field(policy_name: str, value: str) -> bool:
        """Read a policy's `Approval:` field. Absent means no gate; a
        negated value means no gate; an affirming value means violations
        park for a human. Anything unreadable fails loudly -- a gate the
        author declared and the runtime silently ignores is a governance
        hole."""
        if not value:
            return False
        words = set(normalize(value).split())
        if words & {"no", "not", "none", "never", "false"}:
            return False
        if words & {"required", "needed", "mandatory", "human", "yes", "true"}:
            return True
        raise ValueError(
            f"Policy '{policy_name}' has an unreadable Approval field '{value}' -- "
            "write 'Approval: required' or 'Approval: not required'"
        )

    def _load_workflows(
        self,
        document: Document,
        personas: dict[str, Persona],
        policies: dict[str, Policy],
    ) -> dict[str, Workflow]:
        workflows: dict[str, Workflow] = {}
        last_workflow: Optional[Workflow] = None
        for section in document.sections:
            if "deliverable" in normalize(section.name):
                # A Deliverable section declares the Contract of the
                # workflow authored directly above it.
                if last_workflow is None:
                    raise ValueError(
                        f"Deliverable section '{section.name}' has no workflow above it to attach to"
                    )
                last_workflow.contract = self._load_contract(section, last_workflow)
                continue
            body = section.body(field_keys=("persona", "delegate to", "delegate", "policies", "policy", "pattern"))
            workflow = Workflow(name=section.name, pattern=body.field("pattern"))
            default_persona: Optional[Persona] = None
            default_reference = body.field("persona", "delegate to", "delegate")
            if default_reference:
                default_persona = _resolve(personas, default_reference, "persona")
            for item in body.numbered or body.bullets:
                instruction, persona = _split_delegation(item, personas)
                workflow.add_step(instruction, persona=persona or default_persona)
            for reference in _split_references(body.field("policies", "policy")):
                workflow.add_policy(_resolve(policies, reference, "policy"))
            workflows[normalize(section.name)] = workflow
            last_workflow = workflow
        return workflows

    @staticmethod
    def _load_contract(section: Section, workflow: Workflow) -> Contract:
        body = section.body()
        contract = Contract(name=f"{workflow.name} Deliverable", description=body.prose)
        for bullet in body.bullets:
            name, separator, meaning = bullet.partition(": ")
            if not separator:
                name, separator, meaning = bullet.partition(":")
            if separator and name.strip():
                contract.add_field(name.strip(), meaning.strip())
            else:
                raise ValueError(
                    f"Deliverable field '{bullet}' in '{workflow.name}' must be written as 'name: meaning'"
                )
        if not contract.fields:
            raise ValueError(f"Deliverable of '{workflow.name}' declares no fields -- add '- name: meaning' bullets")
        return contract

    def _load_processes(
        self, document: Document, workflows: dict[str, Workflow]
    ) -> tuple[list[Process], set[str]]:
        processes: list[Process] = []
        referenced: set[str] = set()
        for section in document.sections:
            body = section.body(field_keys=("workflows", "workflow"))
            process = Process(name=section.name, description=body.prose)
            for reference in _split_references(body.field("workflows", "workflow")):
                workflow = _resolve(workflows, reference, "workflow")
                process.add_workflow(workflow)
                referenced.add(normalize(reference))
            for bullet in body.bullets:
                key = normalize(bullet)
                if key in workflows:
                    process.add_workflow(workflows[key])
                    referenced.add(key)
                elif bullet:
                    # A bullet that names no workflow is descriptive prose.
                    process.description = "\n".join(filter(None, [process.description, f"- {bullet}"]))
            processes.append(process)
        return processes, referenced

    # -- mapping governance and strategy onto the runtime --------------------

    def _apply_policy_scopes(
        self,
        runtime: Runtime,
        policies: dict[str, Policy],
        scopes: dict[str, str],
        workflows: dict[str, Workflow],
    ) -> None:
        for key, policy in policies.items():
            scope = scopes.get(key, "")
            targets = _split_references(scope) or ["runtime"]
            for target in targets:
                lowered = target.lower().strip()
                if lowered in _RUNTIME_SCOPES or "runtime" in lowered:
                    runtime.add_policy(policy)
                else:
                    workflow = _resolve(workflows, target, "workflow")
                    if policy not in workflow.policies:
                        workflow.add_policy(policy)

    def _apply_strategy(self, runtime: Runtime, strategy: Strategy) -> None:
        runtime.strategy = strategy

        if strategy.history_capacity:
            runtime.memory.capacity = strategy.history_capacity

        if strategy.subagents_configured:
            runtime.spawner.enabled = strategy.subagents_enabled
            runtime.spawner.limit = strategy.max_subagents

        binding = strategy.model_binding()
        if binding is not None and (binding.resolve_api_key() is not None or binding.api_base):
            # Attach the declared binding only when its credential (or a
            # local endpoint) is actually reachable, so a stack loaded on a
            # machine without keys degrades to the deterministic fallback
            # instead of crashing mid-reasoning.
            runtime.model_binding = binding

        if strategy.cross_session and strategy.session_enabled:
            raw_path = Path(strategy.session_path or ".ear/session.md")
            path = raw_path if raw_path.is_absolute() else self.directory / raw_path
            store = SessionStore(str(path))
            store.restore(runtime)
            runtime.session_store = store

        if strategy.audit_trail and strategy.audit_enabled:
            raw_path = Path(strategy.audit_path or ".ear/reasoning.md")
            path = raw_path if raw_path.is_absolute() else self.directory / raw_path
            runtime.reasoning_log.path = str(path)
            runtime.reasoning_log.resume()

        if strategy.knowledge_sources:
            knowledge = Knowledge()
            for name, pattern in strategy.knowledge_sources:
                for path in self._knowledge_paths(name, pattern):
                    knowledge.add_document(name, path.name, path.read_text(encoding="utf-8"))
            runtime.librarian.knowledge = knowledge


    def _knowledge_paths(self, name: str, pattern: str) -> list[Path]:
        """Resolve one declared knowledge source to real files, loudly:
        knowledge the author declared and the runtime silently doesn't
        have is a governance hole, not a default."""
        candidate = Path(pattern)
        if candidate.is_absolute():
            paths = [candidate] if candidate.exists() else []
        else:
            try:
                paths = sorted(self.directory.glob(pattern))
            except ValueError:
                paths = []
        paths = [path for path in paths if path.is_file()]
        if not paths:
            raise ValueError(
                f"Knowledge source '{name}' matched no files for '{pattern}' under {self.directory}"
            )
        return paths


def load_runtime(directory: Union[str, Path], name: Optional[str] = None) -> Runtime:
    """Stack a directory of natural-language markdown files into a Runtime."""
    return Loader(directory).load(name=name)


# -- reference resolution ----------------------------------------------------


def _split_references(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in re.split(r"[,;]", value) if part.strip()]


def _resolve(mapping: dict, reference: str, kind: str):
    key = normalize(reference)
    if key not in mapping:
        known = ", ".join(sorted(item.name for item in mapping.values())) or "none"
        close = difflib.get_close_matches(key, list(mapping), n=1)
        hint = f" -- did you mean '{mapping[close[0]].name}'?" if close else ""
        raise ValueError(f"Unknown {kind} '{reference}' referenced in the stack{hint} -- known {kind}s: {known}")
    return mapping[key]


def _split_delegation(item: str, personas: dict[str, Persona]) -> tuple[str, Optional[Persona]]:
    """Split a narrated step like 'Band the profile. (Credit Risk Guru)'
    into its instruction and the Persona it delegates to. The trailing name
    is only treated as a delegation when it resolves to a known persona --
    otherwise the whole line stays the instruction, untouched."""
    for pattern in _DELEGATION_PATTERNS:
        match = pattern.search(item)
        if not match:
            continue
        who = _DELEGATION_PREFIX.sub("", match.group("who").strip())
        key = normalize(who)
        if key in personas:
            return item[: match.start()].rstrip(), personas[key]
    return item, None
