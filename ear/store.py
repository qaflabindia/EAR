"""Store -- named catalogues of Skills, Personas, Tasks, Workflows,
Processes and Policies: one markdown file per object, on disk.

`Loader` stacks *one directory's* skills.md/persona.md/workflow.md/
process.md/policy.md into *one* Runtime -- the authoring model for a
single stack. A Store is the complementary shape: a reusable *library*,
one named object per file (`skills/band-credit-profile.md`,
`personas/credit-risk-guru.md`, ...), addressable by name and composable
into any number of stacks, the same way a shared code library is imported
by many programs instead of copy-pasted into each.

Each kind-specific store (`SkillStore`, `PersonaStore`, `TaskStore`,
`WorkflowStore`, `ProcessStore`, `PolicyStore`) reads and writes through
the exact same `Section`/`Document` codec, and for every kind but `Task`
(a new object with no place in the existing stacked files) parses through
`Loader`'s own per-kind parsing -- a store file is simply a one-section
stacked-markdown document, so nothing here duplicates or drifts from what
`skills.md` etc. already mean. Cross-references between kinds (a Persona's
`Skills:` line, a Workflow's delegated Personas and Policies, a Process's
`Workflows:` line) are resolved the same way Loader resolves them within
one file: load the referenced kind first, pass its catalogue in.

Nothing here decides anything -- like Loader, a Store is structural. It
does not judge, retrieve by relevance, or version; it is a filing cabinet,
not a search engine. (Swap in `Knowledge`/`Librarian`'s BM25 narrowing on
top of a Store's directory if relevance-ranked discovery is needed later --
the file-per-object shape here is exactly what that narrowing chunks over.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .loader import Loader
from .persona import Persona
from .policy import Policy
from .process import Process
from .section import normalize, parse_document
from .skill import Skill
from .task import TaskDefinition
from .workflow import Workflow

_SLUG_UNSAFE = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    """A filesystem-safe file stem for a catalogued name, folded the same
    way every cross-reference in the stack already is -- 'Credit Risk
    Guru', 'credit-risk-guru' and 'credit_risk_guru' all address the same
    file."""
    slug = _SLUG_UNSAFE.sub("-", normalize(name)).strip("-")
    return slug or "unnamed"


@dataclass
class Store:
    """Raw file operations for one directory of `<slug>.md` objects. The
    kind-specific stores below build named domain objects on top of these
    primitives; nothing else in this module talks to the filesystem
    directly."""

    directory: Union[str, Path]

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[str]:
        """The catalogued names, read back from each file's own heading
        rather than the filename -- the slug is an address, not the name."""
        names = []
        for path in sorted(self.directory.glob("*.md")):
            document = parse_document(path.read_text(encoding="utf-8"))
            if document.sections:
                names.append(document.sections[0].name)
        return names

    def path_for(self, name: str) -> Path:
        return self.directory / f"{_slug(name)}.md"

    def exists(self, name: str) -> bool:
        return self.path_for(name).exists()

    def read(self, name: str) -> str:
        path = self.path_for(name)
        if not path.exists():
            known = ", ".join(self.list()) or "none"
            raise FileNotFoundError(f"'{name}' is not in {self.directory} -- known: {known}")
        return path.read_text(encoding="utf-8")

    def write(self, name: str, text: str) -> Path:
        return_path = self.path_for(name)
        return_path.write_text(text, encoding="utf-8")
        return return_path

    def delete(self, name: str) -> None:
        self.path_for(name).unlink(missing_ok=True)

    def document(self, name: str):
        """The single-section Document parsed from `name`'s file. Every
        kind-specific store's `load` starts here, one object per file."""
        parsed = parse_document(self.read(name))
        if not parsed.sections:
            raise ValueError(f"'{name}' in {self.directory} has no heading to read an object from")
        return parsed


@dataclass
class SkillStore:
    """A named catalogue of Skills, one per file, under `directory`."""

    directory: Union[str, Path]

    def __post_init__(self) -> None:
        self.store = Store(self.directory)

    def list(self) -> list[str]:
        return self.store.list()

    def save(self, skill: Skill) -> Path:
        return self.store.write(skill.name, skill.to_markdown())

    def load(self, name: str) -> Skill:
        return next(iter(Loader._load_skills(self.store.document(name)).values()))

    def load_all(self) -> dict[str, Skill]:
        """Every catalogued skill, keyed by normalized name -- the shape
        `PersonaStore.load`/`load_all` expect for cross-referencing."""
        return {normalize(name): self.load(name) for name in self.store.list()}

    def delete(self, name: str) -> None:
        self.store.delete(name)


@dataclass
class PersonaStore:
    """A named catalogue of Personas, one per file. `Skills:` references
    inside a persona file are resolved against a `skills` catalogue the
    caller loads first -- from a SkillStore, or any other source of
    normalized-name-keyed Skills."""

    directory: Union[str, Path]

    def __post_init__(self) -> None:
        self.store = Store(self.directory)

    def list(self) -> list[str]:
        return self.store.list()

    def save(self, persona: Persona) -> Path:
        return self.store.write(persona.name, persona.to_markdown())

    def load(self, name: str, skills: Optional[dict[str, Skill]] = None) -> Persona:
        return next(iter(Loader._load_personas(self.store.document(name), skills or {}).values()))

    def load_all(self, skills: Optional[dict[str, Skill]] = None) -> dict[str, Persona]:
        return {normalize(name): self.load(name, skills) for name in self.store.list()}

    def delete(self, name: str) -> None:
        self.store.delete(name)


@dataclass
class TaskStore:
    """A named catalogue of TaskDefinitions -- atomic, SIPOC-shaped steps reusable
    across any number of Workflows. Unlike the other kinds, a TaskDefinition has no
    counterpart in the stacked single-runtime files (`Step` stays inline,
    disposable authoring); TaskDefinitions are new, so this store parses and renders
    them itself rather than delegating to Loader."""

    directory: Union[str, Path]

    def __post_init__(self) -> None:
        self.store = Store(self.directory)

    def list(self) -> list[str]:
        return self.store.list()

    def save(self, task: TaskDefinition) -> Path:
        return self.store.write(task.name, task.to_markdown())

    def load(self, name: str) -> TaskDefinition:
        return TaskDefinition.from_section(self.store.document(name).sections[0])

    def load_all(self) -> dict[str, TaskDefinition]:
        return {normalize(name): self.load(name) for name in self.store.list()}

    def delete(self, name: str) -> None:
        self.store.delete(name)


@dataclass
class WorkflowStore:
    """A named catalogue of Workflows. `Policies:` and delegated personas
    inside a workflow file are resolved against `personas`/`policies`
    catalogues the caller loads first."""

    directory: Union[str, Path]

    def __post_init__(self) -> None:
        self.store = Store(self.directory)

    def list(self) -> list[str]:
        return self.store.list()

    def save(self, workflow: Workflow) -> Path:
        return self.store.write(workflow.name, workflow.to_markdown())

    def load(
        self,
        name: str,
        personas: Optional[dict[str, Persona]] = None,
        policies: Optional[dict[str, Policy]] = None,
    ) -> Workflow:
        return next(
            iter(Loader._load_workflows(self.store.document(name), personas or {}, policies or {}).values())
        )

    def load_all(
        self,
        personas: Optional[dict[str, Persona]] = None,
        policies: Optional[dict[str, Policy]] = None,
    ) -> dict[str, Workflow]:
        return {normalize(name): self.load(name, personas, policies) for name in self.store.list()}

    def delete(self, name: str) -> None:
        self.store.delete(name)


@dataclass
class ProcessStore:
    """A named catalogue of Processes. `Workflows:` references inside a
    process file are resolved against a `workflows` catalogue the caller
    loads first."""

    directory: Union[str, Path]

    def __post_init__(self) -> None:
        self.store = Store(self.directory)

    def list(self) -> list[str]:
        return self.store.list()

    def save(self, process: Process) -> Path:
        return self.store.write(process.name, process.to_markdown())

    def load(self, name: str, workflows: Optional[dict[str, Workflow]] = None) -> Process:
        processes, _referenced = Loader._load_processes(self.store.document(name), workflows or {})
        return processes[0]

    def load_all(self, workflows: Optional[dict[str, Workflow]] = None) -> dict[str, Process]:
        return {normalize(name): self.load(name, workflows) for name in self.store.list()}

    def delete(self, name: str) -> None:
        self.store.delete(name)


@dataclass
class PolicyStore:
    """A named catalogue of Policies, one per file -- reusable governance
    text, independent of which runtime or workflow currently applies it
    (scope is decided at load time, not stored in the catalogue)."""

    directory: Union[str, Path]

    def __post_init__(self) -> None:
        self.store = Store(self.directory)

    def list(self) -> list[str]:
        return self.store.list()

    def save(self, policy: Policy) -> Path:
        return self.store.write(policy.name, policy.to_markdown())

    def load(self, name: str) -> Policy:
        policies, _scopes = Loader._load_policies(self.store.document(name))
        return next(iter(policies.values()))

    def load_all(self) -> dict[str, Policy]:
        return {normalize(name): self.load(name) for name in self.store.list()}

    def delete(self, name: str) -> None:
        self.store.delete(name)


@dataclass
class Stores:
    """All six catalogues rooted under one directory (`root/skills/`,
    `root/personas/`, `root/tasks/`, `root/workflows/`, `root/processes/`,
    `root/policies/`), and the one right order to load them in so every
    cross-reference resolves: skills, then personas, then policies, then
    tasks, then workflows, then processes."""

    root: Union[str, Path]

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.skills = SkillStore(self.root / "skills")
        self.personas = PersonaStore(self.root / "personas")
        self.tasks = TaskStore(self.root / "tasks")
        self.workflows = WorkflowStore(self.root / "workflows")
        self.processes = ProcessStore(self.root / "processes")
        self.policies = PolicyStore(self.root / "policies")

    def load_all(self) -> dict[str, dict]:
        """Every catalogued object, loaded in cross-reference order and
        keyed by kind then normalized name -- a ready-made set of
        catalogues to compose a Runtime's processes from by hand, or to
        resolve a Task/Workflow's persona/policy references against."""
        skills = self.skills.load_all()
        personas = self.personas.load_all(skills)
        policies = self.policies.load_all()
        tasks = self.tasks.load_all()
        workflows = self.workflows.load_all(personas, policies)
        processes = self.processes.load_all(workflows)
        return {
            "skills": skills,
            "personas": personas,
            "policies": policies,
            "tasks": tasks,
            "workflows": workflows,
            "processes": processes,
        }
