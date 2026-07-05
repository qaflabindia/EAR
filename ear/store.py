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

The named on-disk `Store` is the default and the fallback, never something
a deployment has to opt into: with no `Catalogue Store` section in
memory.md (or one that says `Store: false`), every kind-specific store
above is file-based, exactly as always. Only an explicit `Store: true`
plus a recognized `Backend:` (currently `apache-age`) swaps in
`PostgresAgeBackend` instead, for deployments whose catalogue has outgrown
a directory of files. `psycopg` is never imported unless that backend is
actually constructed, so the zero-dependency default is untouched by this
being available at all -- see `Loader._apply_strategy` for the wiring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Union, runtime_checkable

from .loader import Loader
from .persona import Persona
from .policy import Policy
from .process import Process
from .section import normalize, parse_document
from .skill import Skill
from .task import TaskDefinition
from .workflow import Workflow

_SLUG_UNSAFE = re.compile(r"[^a-z0-9]+")
_VALID_TABLE_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")


def _slug(name: str) -> str:
    """A filesystem-safe file stem for a catalogued name, folded the same
    way every cross-reference in the stack already is -- 'Credit Risk
    Guru', 'credit-risk-guru' and 'credit_risk_guru' all address the same
    file."""
    slug = _SLUG_UNSAFE.sub("-", normalize(name)).strip("-")
    return slug or "unnamed"


@runtime_checkable
class CatalogueBackend(Protocol):
    """The minimal operations any catalogue backend must support: list the
    catalogued names, check/read/write/delete one by name. The file-based
    `Store` and the optional `PostgresAgeBackend` both satisfy this
    structurally (PEP 544, no inheritance required), so every kind-specific
    store below (`SkillStore`, ...) works unchanged regardless of which one
    backs it -- swapping backends is a constructor argument, not a
    different code path."""

    def list(self) -> list[str]: ...
    def exists(self, name: str) -> bool: ...
    def read(self, name: str) -> str: ...
    def write(self, name: str, text: str) -> None: ...
    def delete(self, name: str) -> None: ...


def _document(backend: CatalogueBackend, name: str):
    """The single-section Document parsed from `name`'s stored text, from
    whichever backend it came from. Every kind-specific store's `load`
    starts here."""
    parsed = parse_document(backend.read(name))
    if not parsed.sections:
        raise ValueError(f"'{name}' has no heading to read an object from")
    return parsed


def _backend_or_files(directory: Optional[Union[str, Path]], backend: Optional[CatalogueBackend]) -> CatalogueBackend:
    if backend is not None:
        return backend
    if directory is None:
        raise ValueError("Provide either 'directory' (file-based) or 'backend' (e.g. a PostgresAgeBackend)")
    return Store(directory)


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
class PostgresAgeBackend:
    """An optional, database-backed catalogue backend: one Postgres table
    per kind, storing each object's `to_markdown()` blob in a `body`
    column alongside an indexed `slug`/`name` -- the same blob-first shape
    designed against the scalability discussion this backend exists for.
    Apache AGE mirrors the same rows into a graph (see `db/schema.sql`)
    for the multi-hop traversal queries a directory of files can't answer
    efficiently; that mirroring lives in the database (triggers), not
    here, so this class only ever needs to read and write the relational
    row.

    Strictly opt-in: constructing this is the only place `psycopg` is
    imported anywhere in the package, so a deployment that never sets
    `Store: true` in memory.md's Catalogue Store section never needs the
    driver installed at all -- the zero-dependency default is untouched.
    """

    connection: str
    table: str

    def __post_init__(self) -> None:
        if not _VALID_TABLE_NAME.match(self.table):
            raise ValueError(f"Invalid table name '{self.table}' for PostgresAgeBackend")
        try:
            import psycopg
        except ImportError as error:
            raise ImportError(
                "PostgresAgeBackend requires the optional 'psycopg' driver -- install it "
                "with `pip install ear[postgres]`, or remove/disable the memory.md "
                "Catalogue Store section to keep using the named on-disk catalogue."
            ) from error
        self._psycopg = psycopg
        self._ensure_table()

    def _connect(self):
        return self._psycopg.connect(self.connection)

    def _ensure_table(self) -> None:
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id bigserial PRIMARY KEY,
                    slug text NOT NULL UNIQUE,
                    name text NOT NULL,
                    body text NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            conn.commit()

    def list(self) -> list[str]:
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(f"SELECT name FROM {self.table} ORDER BY name")
            return [row[0] for row in cursor.fetchall()]

    def exists(self, name: str) -> bool:
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(f"SELECT 1 FROM {self.table} WHERE slug = %s", (_slug(name),))
            return cursor.fetchone() is not None

    def read(self, name: str) -> str:
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(f"SELECT body FROM {self.table} WHERE slug = %s", (_slug(name),))
            row = cursor.fetchone()
            if row is None:
                known = ", ".join(self.list()) or "none"
                raise FileNotFoundError(f"'{name}' is not in table '{self.table}' -- known: {known}")
            return row[0]

    def write(self, name: str, text: str) -> None:
        document = parse_document(text)
        display_name = document.sections[0].name if document.sections else name
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self.table} (slug, name, body, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, body = EXCLUDED.body, updated_at = now()
                """,
                (_slug(name), display_name, text),
            )
            conn.commit()

    def delete(self, name: str) -> None:
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM {self.table} WHERE slug = %s", (_slug(name),))
            conn.commit()


@dataclass
class SkillStore:
    """A named catalogue of Skills. File-based under `directory` by
    default; pass `backend` (e.g. a PostgresAgeBackend) to use a
    database-backed catalogue instead."""

    directory: Optional[Union[str, Path]] = None
    backend: Optional[CatalogueBackend] = None

    def __post_init__(self) -> None:
        self.store = _backend_or_files(self.directory, self.backend)

    def list(self) -> list[str]:
        return self.store.list()

    def save(self, skill: Skill) -> None:
        self.store.write(skill.name, skill.to_markdown())

    def load(self, name: str) -> Skill:
        return next(iter(Loader._load_skills(_document(self.store, name)).values()))

    def load_all(self) -> dict[str, Skill]:
        """Every catalogued skill, keyed by normalized name -- the shape
        `PersonaStore.load`/`load_all` expect for cross-referencing."""
        return {normalize(name): self.load(name) for name in self.store.list()}

    def delete(self, name: str) -> None:
        self.store.delete(name)


@dataclass
class PersonaStore:
    """A named catalogue of Personas. `Skills:` references inside a
    persona file are resolved against a `skills` catalogue the caller
    loads first -- from a SkillStore, or any other source of
    normalized-name-keyed Skills. File-based under `directory` by
    default; pass `backend` for a database-backed catalogue instead."""

    directory: Optional[Union[str, Path]] = None
    backend: Optional[CatalogueBackend] = None

    def __post_init__(self) -> None:
        self.store = _backend_or_files(self.directory, self.backend)

    def list(self) -> list[str]:
        return self.store.list()

    def save(self, persona: Persona) -> None:
        self.store.write(persona.name, persona.to_markdown())

    def load(self, name: str, skills: Optional[dict[str, Skill]] = None) -> Persona:
        return next(iter(Loader._load_personas(_document(self.store, name), skills or {}).values()))

    def load_all(self, skills: Optional[dict[str, Skill]] = None) -> dict[str, Persona]:
        return {normalize(name): self.load(name, skills) for name in self.store.list()}

    def delete(self, name: str) -> None:
        self.store.delete(name)


@dataclass
class TaskStore:
    """A named catalogue of TaskDefinitions -- atomic, SIPOC-shaped steps
    reusable across any number of Workflows. Unlike the other kinds, a
    TaskDefinition has no counterpart in the stacked single-runtime files
    (`Step` stays inline, disposable authoring); TaskDefinitions are new,
    so this store parses and renders them itself rather than delegating
    to Loader. File-based under `directory` by default; pass `backend`
    for a database-backed catalogue instead."""

    directory: Optional[Union[str, Path]] = None
    backend: Optional[CatalogueBackend] = None

    def __post_init__(self) -> None:
        self.store = _backend_or_files(self.directory, self.backend)

    def list(self) -> list[str]:
        return self.store.list()

    def save(self, task: TaskDefinition) -> None:
        self.store.write(task.name, task.to_markdown())

    def load(self, name: str) -> TaskDefinition:
        return TaskDefinition.from_section(_document(self.store, name).sections[0])

    def load_all(self) -> dict[str, TaskDefinition]:
        return {normalize(name): self.load(name) for name in self.store.list()}

    def delete(self, name: str) -> None:
        self.store.delete(name)


@dataclass
class WorkflowStore:
    """A named catalogue of Workflows. `Policies:` and delegated personas
    inside a workflow file are resolved against `personas`/`policies`
    catalogues the caller loads first. File-based under `directory` by
    default; pass `backend` for a database-backed catalogue instead."""

    directory: Optional[Union[str, Path]] = None
    backend: Optional[CatalogueBackend] = None

    def __post_init__(self) -> None:
        self.store = _backend_or_files(self.directory, self.backend)

    def list(self) -> list[str]:
        return self.store.list()

    def save(self, workflow: Workflow) -> None:
        self.store.write(workflow.name, workflow.to_markdown())

    def load(
        self,
        name: str,
        personas: Optional[dict[str, Persona]] = None,
        policies: Optional[dict[str, Policy]] = None,
    ) -> Workflow:
        return next(
            iter(Loader._load_workflows(_document(self.store, name), personas or {}, policies or {}).values())
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
    loads first. File-based under `directory` by default; pass `backend`
    for a database-backed catalogue instead."""

    directory: Optional[Union[str, Path]] = None
    backend: Optional[CatalogueBackend] = None

    def __post_init__(self) -> None:
        self.store = _backend_or_files(self.directory, self.backend)

    def list(self) -> list[str]:
        return self.store.list()

    def save(self, process: Process) -> None:
        self.store.write(process.name, process.to_markdown())

    def load(self, name: str, workflows: Optional[dict[str, Workflow]] = None) -> Process:
        processes, _referenced = Loader._load_processes(_document(self.store, name), workflows or {})
        return processes[0]

    def load_all(self, workflows: Optional[dict[str, Workflow]] = None) -> dict[str, Process]:
        return {normalize(name): self.load(name, workflows) for name in self.store.list()}

    def delete(self, name: str) -> None:
        self.store.delete(name)


@dataclass
class PolicyStore:
    """A named catalogue of Policies -- reusable governance text,
    independent of which runtime or workflow currently applies it (scope
    is decided at load time, not stored in the catalogue). File-based
    under `directory` by default; pass `backend` for a database-backed
    catalogue instead."""

    directory: Optional[Union[str, Path]] = None
    backend: Optional[CatalogueBackend] = None

    def __post_init__(self) -> None:
        self.store = _backend_or_files(self.directory, self.backend)

    def list(self) -> list[str]:
        return self.store.list()

    def save(self, policy: Policy) -> None:
        self.store.write(policy.name, policy.to_markdown())

    def load(self, name: str) -> Policy:
        policies, _scopes = Loader._load_policies(_document(self.store, name))
        return next(iter(policies.values()))

    def load_all(self) -> dict[str, Policy]:
        return {normalize(name): self.load(name) for name in self.store.list()}

    def delete(self, name: str) -> None:
        self.store.delete(name)


_DATABASE_BACKENDS = {"apache-age", "postgres", "postgresql"}


@dataclass
class Stores:
    """All six catalogues, and the one right order to load them in so
    every cross-reference resolves: skills, then personas, then policies,
    then tasks, then workflows, then processes.

    File-based under `root` (`root/skills/`, `root/personas/`, ...) by
    default -- this is the fallback, not something to opt into. Only when
    `backend_name` names a recognized database backend (currently
    `apache-age`, aliases `postgres`/`postgresql`) does each kind-specific
    store switch to a `PostgresAgeBackend` against `connection` instead.
    `Stores.from_strategy` is what `Loader` calls to make this choice from
    a memory.md `Catalogue Store` section rather than a direct argument."""

    root: Union[str, Path]
    connection: Optional[str] = None
    backend_name: str = "file"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.backend_name.lower() in _DATABASE_BACKENDS:
            self.skills = SkillStore(backend=PostgresAgeBackend(self.connection, "skills"))
            self.personas = PersonaStore(backend=PostgresAgeBackend(self.connection, "personas"))
            self.tasks = TaskStore(backend=PostgresAgeBackend(self.connection, "tasks"))
            self.workflows = WorkflowStore(backend=PostgresAgeBackend(self.connection, "workflows"))
            self.processes = ProcessStore(backend=PostgresAgeBackend(self.connection, "processes"))
            self.policies = PolicyStore(backend=PostgresAgeBackend(self.connection, "policies"))
        else:
            self.skills = SkillStore(self.root / "skills")
            self.personas = PersonaStore(self.root / "personas")
            self.tasks = TaskStore(self.root / "tasks")
            self.workflows = WorkflowStore(self.root / "workflows")
            self.processes = ProcessStore(self.root / "processes")
            self.policies = PolicyStore(self.root / "policies")

    @classmethod
    def from_strategy(cls, directory: Union[str, Path], strategy: Any) -> "Stores":
        """Build a Stores rooted at `directory`, switching to a database
        backend only if `strategy`'s Catalogue Store section opted in
        (`Store: true` and a recognized `Backend:`). Absent, disabled, or
        an unrecognized backend name all keep the named on-disk catalogue
        -- the fallback is the default, not a special case."""
        enabled = bool(getattr(strategy, "catalogue_store_enabled", False))
        backend = strategy.catalogue_backend if enabled else ""
        return cls(root=directory, connection=getattr(strategy, "catalogue_connection", "") or None, backend_name=backend or "file")

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
