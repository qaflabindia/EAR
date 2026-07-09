"""Loader -- build a whole Runtime from a directory of stacked markdown
files, written entirely in natural language. The user writes no code:

    skills.md    prompts stacked into Skills      (heading = skill, prose = prompt)
    persona.md   skills stacked into Personas     (prose = instructions, `Skills:` = stack)
    workflow.md  steps stacked into Workflows     (numbered steps, `(Persona)` delegates)
    process.md   workflows stacked into Processes (prose = description, `Workflows:` = stack)
    policy.md    governance, risk and controls    (prose = statement, `Applies to:` = scope)
    tenant.md    the org this stack belongs to    (`Org id:`, fiscal year, optional -- defaults
                                                   to the "default" tenant when absent)
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
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional, Union

from .contract import Contract
from .knowledge import Knowledge, KnowledgeSource
from .llm import LMError, fetch_text
from .persona import Persona
from .policy import Policy
from .process import Process
from .runtime import Runtime
from .section import Document, Section, normalize, parse_document
from .session_store import SessionStore
from .skill import Skill
from .strategy import Strategy, count_in_prose, days_in_prose
from .tenant import Tenant
from .workflow import Workflow

_FILE_CANDIDATES = {
    "skills": ("skills.md", "skill.md"),
    "personas": ("persona.md", "personas.md"),
    "workflows": ("workflow.md", "workflows.md"),
    "processes": ("process.md", "processes.md"),
    "policies": ("policy.md", "policies.md"),
    "tenant": ("tenant.md", "org.md"),
    "memory": ("memory.md",),
}

_TENANT_FIELD_KEYS = (
    "org id",
    "org",
    "fiscal year start",
    "fiscal year end",
    "timezone",
    "secret env var",
    "secret",
)

_RUNTIME_SCOPES = {"runtime", "the runtime", "all", "everything", "global", "the whole runtime"}
_TOOL_SCOPES = {"tools", "tool", "tool calls", "tool call", "tool invocations", "tool invocation", "any tool"}

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
        tenant_doc = self._parse("tenant")
        memory_text = self._read("memory")

        skills = Loader._load_skills(skills_doc)
        personas = Loader._load_personas(personas_doc, skills)
        policies, policy_scopes = Loader._load_policies(policies_doc)
        workflows = Loader._load_workflows(workflows_doc, personas, policies)
        processes, referenced = Loader._load_processes(processes_doc, workflows)
        tenant = Loader._load_tenant(tenant_doc)

        runtime = Runtime(name=name or processes_doc.title or self.directory.name)
        runtime.tenant = tenant
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

    @staticmethod
    def _load_skills(document: Document) -> dict[str, Skill]:
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

    @staticmethod
    def _load_personas(document: Document, skills: dict[str, Skill]) -> dict[str, Persona]:
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

    @staticmethod
    def _load_policies(document: Document) -> tuple[dict[str, Policy], dict[str, str]]:
        policies: dict[str, Policy] = {}
        scopes: dict[str, str] = {}
        for section in document.sections:
            body = section.body(
                field_keys=(
                    "fallback",
                    "fallback expression",
                    "applies to",
                    "applies",
                    "scope",
                    "approval",
                    "approvers",
                    "approver",
                    "escalate",
                    "escalation",
                )
            )
            statement = "\n".join(filter(None, [body.prose] + [f"- {bullet}" for bullet in body.bullets]))
            key = normalize(section.name)
            escalation = body.field("escalate", "escalation")
            escalation_days = days_in_prose(escalation) if escalation else None
            if escalation and escalation_days is None:
                # A deadline the author declared and the runner silently
                # can't read is a governance hole, not a default.
                raise ValueError(
                    f"Policy '{section.name}' declares Escalate '{escalation}' but no readable "
                    "period -- write 'Escalate: after 3 days'"
                )
            policies[key] = Policy(
                name=section.name,
                statement=statement,
                fallback_expression=body.field("fallback", "fallback expression"),
                approval_required=Loader._read_approval_field(section.name, body.field("approval")),
                approvers=_split_references(body.field("approvers", "approver")),
                escalation=escalation,
                escalation_days=escalation_days,
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

    @staticmethod
    def _load_workflows(
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
                last_workflow.contract = Loader._load_contract(section, last_workflow)
                continue
            body = section.body(
                field_keys=(
                    "persona",
                    "delegate to",
                    "delegate",
                    "policies",
                    "policy",
                    "pattern",
                    "routes",
                    "route",
                    "retries",
                    "retry",
                )
            )
            workflow = Workflow(
                name=section.name,
                pattern=body.field("pattern"),
                routes=body.field("routes", "route"),
            )
            retries = body.field("retries", "retry")
            if retries:
                workflow.retry_budget = count_in_prose(retries)
                if workflow.retry_budget is None:
                    raise ValueError(
                        f"Workflow '{section.name}' declares Retries '{retries}' but no readable "
                        "count -- write 'Retries: retry a failed leg twice'"
                    )
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

    @staticmethod
    def _load_processes(
        document: Document, workflows: dict[str, Workflow]
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

    @staticmethod
    def _load_tenant(document: Document) -> Tenant:
        """Read the stack's `tenant.md` -- one org record, the same
        heading-plus-fields shape as every other stacked file, just with
        exactly one section expected. No `tenant.md` at all (an empty
        Document, no sections) yields the default tenant: `org_id`
        `"default"`, no fiscal year, so workday notation falls back to the
        calendar year."""
        if not document.sections:
            return Tenant()
        section = document.sections[0]
        body = section.body(field_keys=_TENANT_FIELD_KEYS)
        org_id = body.field("org id", "org")
        if not org_id:
            raise ValueError(
                f"Tenant '{section.name}' declares no 'Org id:' -- every tenant.md needs one"
            )
        return Tenant(
            org_id=org_id,
            name=section.name,
            fiscal_year_start=Loader._parse_tenant_date(section.name, "Fiscal year start", body.field("fiscal year start")),
            fiscal_year_end=Loader._parse_tenant_date(section.name, "Fiscal year end", body.field("fiscal year end")),
            timezone=body.field("timezone") or None,
            secret_env_var=body.field("secret env var", "secret") or None,
        )

    @staticmethod
    def _parse_tenant_date(tenant_name: str, field_label: str, value: str) -> Optional[date]:
        if not value:
            return None
        try:
            return date.fromisoformat(value.strip())
        except ValueError as error:
            raise ValueError(
                f"Tenant '{tenant_name}' declares '{field_label}: {value}' but it isn't a readable "
                "date -- write it as YYYY-MM-DD"
            ) from error

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
                if normalize(target) in _TOOL_SCOPES:
                    if policy not in runtime.tool_policies:
                        runtime.tool_policies.append(policy)
                elif lowered in _RUNTIME_SCOPES or "runtime" in lowered:
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

        aux_binding = strategy.auxiliary_model_binding()
        if aux_binding is not None and (aux_binding.resolve_api_key() is not None or aux_binding.api_base):
            runtime.auxiliary_model_binding = aux_binding

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

        if strategy.sandbox_enabled:
            self._open_sandbox(runtime, strategy)

        if strategy.knowledge_sources:
            runtime.librarian.knowledge = self._load_knowledge(runtime, strategy)

        # Local import: `store.py` imports `Loader` (to reuse its per-kind
        # parsing), so a module-level import here would be circular.
        from .store import Stores

        runtime.stores = Stores.from_strategy(self.directory / "store", strategy)

        instructions = self.directory / ".ear" / "instructions.md"
        if instructions.exists():
            # A persisted, reviewable override of the shipped instructions
            # (see Optimizer.load_instructions for scope).
            runtime.optimizer.load_instructions(instructions)

        tools = self.directory / ".ear" / "tools.md"
        runtime.tools_path = str(tools)
        from .acquirer import Acquirer

        # Tools this runtime declared for itself on a prior run (see
        # Acquirer.create_tool) -- merged in, memory.md's own declarations
        # always winning on a name clash. When a Sandbox confines this
        # runtime, its acquired tools live *inside* the sandbox root (see
        # Acquirer's blast-radius note), so that -- not the stack-level
        # file -- is where the reload looks.
        if runtime.sandbox is not None:
            if runtime.sandbox.exists(".ear/tools.md"):
                Acquirer.load_tools(runtime.sandbox.resolve(".ear/tools.md"), strategy)
        elif tools.exists():
            Acquirer.load_tools(tools, strategy)
        if strategy.tool_acquisition:
            runtime.tool_binder.acquirer_tools = runtime.acquirer.as_tools(runtime)

        self._bind_basic_toolsets(runtime, strategy)

    # Terminal / Shell, Code Executor and Environment Admin are three
    # *names* over one physical capability (Sandbox.run) -- Toolsets
    # controls which name(s) are granted, never a fake per-name command
    # filter. A restricted allow-list per name would just be the static
    # per-language table this design already rejected, wearing a
    # different hat: the sandbox can run any command under any of these
    # names, so pretending "code_executor" is narrower than "terminal"
    # would be theater, not access control. What tools.md actually
    # controls is reachability -- which name is bound at all -- never
    # whether the underlying capability physically exists; it always
    # does, the moment a Sandbox is open.
    _SHELL_TOOLSET_NAMES = {
        "terminal": "Run a shell command inside the sandbox -- confined to the workspace and time-limited.",
        "code_executor": "Compile and/or run code inside the sandbox -- confined to the workspace and time-limited.",
        "environment_admin": "Provision the environment (install a package, set up a toolchain) inside the sandbox -- confined to the workspace and time-limited.",
    }

    def _bind_basic_toolsets(self, runtime: Runtime, strategy: Strategy) -> None:
        """Bind the basic toolsets declared enabled (Toolsets in
        memory.md, or the shipped defaults when the section is absent) --
        mechanical capabilities that ship ready rather than something the
        model derives itself each time. Reading/writing
        PDF/Markdown/DOCX/CSV/PPTX/XLSX and the MCP connector are covered
        by tools this Loader already binds elsewhere (Acquirer, native
        MCP client) or by a later pass; this wires web access/search,
        email, and the three sandbox-shell names."""
        from .tool_binder import BoundTool

        enabled = {name for name, on in strategy.toolsets.items() if on}

        from .web import WebAccess

        web = WebAccess(search_provider=strategy.search_provider, search_api_key_env_var=strategy.search_api_key_env_var)
        runtime.tool_binder.basic_tools = web.as_tools(enabled)

        from .mail import Mail

        mail = Mail(
            host=strategy.smtp_host,
            port=strategy.smtp_port,
            user_env_var=strategy.smtp_user_env_var,
            password_env_var=strategy.smtp_password_env_var,
        )
        runtime.tool_binder.basic_tools += mail.as_tools(enabled)

        if runtime.sandbox is not None:
            sandbox = runtime.sandbox

            def run_in_sandbox(command: str) -> str:
                return sandbox.run(command).render()

            for name, description in self._SHELL_TOOLSET_NAMES.items():
                if name in enabled:
                    runtime.tool_binder.basic_tools.append(BoundTool(name=name, description=description, handler=run_in_sandbox))

            if "environment_admin" in enabled:
                def check_environment(names: str = "") -> str:
                    requested = tuple(n.strip() for n in names.split(",") if n.strip()) or ("python3", "node", "npm", "pip")
                    report = sandbox.capabilities(requested)
                    return "\n".join(
                        f"{name}: available -- {info['version']} ({info['path']})"
                        if info["available"]
                        else f"{name}: not available"
                        for name, info in report.items()
                    )

                runtime.tool_binder.basic_tools.append(
                    BoundTool(
                        name="check_environment",
                        description=(
                            "Check which runtimes are actually installed and reachable in this "
                            "sandbox (default: python3, node, npm, pip) -- verified live, never assumed."
                        ),
                        handler=check_environment,
                    )
                )

    def _open_sandbox(self, runtime: Runtime, strategy: Strategy) -> None:
        """Give this runtime instance its own isolated workspace, declared
        in memory.md's Sandbox section. The root defaults under the stack's
        `.ear/`; when the author asks for confined tools ('with a shell',
        'read and write files'), the sandbox's file/shell tools are bound
        into the cycle's toolset. The opening is on the trail."""
        from .sandbox import Sandbox

        raw = Path(strategy.sandbox_root or ".ear/sandbox")
        root = raw if raw.is_absolute() else self.directory / raw
        sandbox = Sandbox.create(
            root=str(root),
            name=runtime.name,
            timeout=strategy.sandbox_timeout if strategy.sandbox_timeout is not None else 30.0,
            memory_mb=strategy.sandbox_memory_mb,
            ephemeral=strategy.sandbox_ephemeral,
        )
        runtime.sandbox = sandbox
        if strategy.sandbox_tools:
            runtime.tool_binder.sandbox_tools = sandbox.as_tools()
        runtime.reasoning_log.record(
            stage="sandbox",
            inputs={
                "root": str(sandbox.root),
                "timeout_s": sandbox.timeout,
                "memory_mb": sandbox.memory_mb,
                "ephemeral": sandbox.ephemeral,
                "tools": strategy.sandbox_tools,
            },
            output=f"sandbox '{runtime.name}' opened at {sandbox.root}",
            rationale=(
                "each runtime instance runs inside its own filesystem-confined, "
                "resource-limited workspace; spawned subagents nest their own"
            ),
        )
        runtime.reasoning_log.flush()

    def _load_knowledge(self, runtime: Runtime, strategy: Strategy) -> Knowledge:
        """Build the Librarian's corpus from the declared sources -- files
        resolved against the stack directory, URLs fetched over the native
        client and cached -- then attach the persisted gist index, writing
        gists for uncovered passages when a model is bound."""
        knowledge = Knowledge()
        for source in strategy.knowledge_sources:
            if source.url:
                path = self._fetch_knowledge(source)
                knowledge.add_document(source.name, path.name, path.read_text(encoding="utf-8"))
            else:
                for path in self._knowledge_paths(source.name, source.pattern):
                    knowledge.add_document(source.name, path.name, path.read_text(encoding="utf-8"))
        self._index_knowledge(runtime, knowledge)
        return knowledge

    def _fetch_knowledge(self, source: KnowledgeSource) -> Path:
        """The cached file for a URL source, refetching when the declared
        cadence says the cache is stale. A failed refresh falls back to
        the cached copy (stale beats absent; the next load retries); a
        failed first fetch fails loudly -- knowledge the author declared
        and the runtime silently doesn't have is a governance hole."""
        suffix = ".md" if source.url.partition("?")[0].endswith(".md") else ".txt"
        path = self.directory / ".ear" / "knowledge" / (normalize(source.name).replace(" ", "-") + suffix)
        if path.exists():
            age_days = (time.time() - path.stat().st_mtime) / 86400
            if source.refresh_days is None or age_days < source.refresh_days:
                return path
        try:
            text = fetch_text(source.url)
        except LMError as error:
            if path.exists():
                return path
            raise ValueError(
                f"Knowledge source '{source.name}' could not be fetched from {source.url} "
                f"and has no cached copy: {error}"
            ) from error
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def _index_knowledge(self, runtime: Runtime, knowledge: Knowledge) -> None:
        """Attach the persisted gist index and, with a model bound, gist
        the passages it doesn't cover -- on the record, like every other
        model call. Offline, BM25 over the raw text stands, and the
        retrieval record labels it as such."""
        from .reasoning_log import calls_so_far, usage_since

        index_path = self.directory / ".ear" / "index.md"
        knowledge.load_index(index_path)
        missing = knowledge.missing_gists()
        if not missing or runtime.model_binding is None:
            return
        lm = runtime.model_binding.activate()
        start = calls_so_far(lm)
        failure = ""
        try:
            knowledge.build_gists(lm)
        except LMError as error:
            # Gists built before the failure still persist; the rest are
            # retried on the next load.
            failure = str(error)
        gisted = len(missing) - len(knowledge.missing_gists())
        if gisted:
            knowledge.write_index(index_path, model_label=runtime.model_binding.model_id)
        runtime.reasoning_log.record(
            stage="indexing",
            inputs={"passages": len(knowledge), "gists_written": gisted},
            output=(
                f"gist index at {index_path}" if not failure else f"indexing interrupted: {failure}"
            ),
            rationale=(
                "one-line gists per passage, persisted by content hash so "
                "differently-phrased questions still find their passages"
            ),
            model=runtime.model_binding.model_id,
            usage=usage_since(lm, start),
        )
        runtime.reasoning_log.flush()

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
