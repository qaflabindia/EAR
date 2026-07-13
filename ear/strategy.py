"""Strategy -- the runtime's operating strategy, stacked in `memory.md`.

One markdown file, written in plain English, declares how the runtime
operates -- and every setting is *extracted from that prose*, never
hardcoded in Python:

    Context History      how much recent history stays verbatim before compression
    Cross-Session Data   where memory/experience/adaptations persist between sessions
    Subagent Spawning    whether subagents may be spawned, and how many
    Model Selection      which LLM provider/model reasons, and where its credential lives
    Reasoning Audit Trail where the ReasoningLog's JSONL trail is written
    MCP                  declared MCP servers (name: what it provides, `command`)
    Tools                declared tools (name: what it does, `command`)
    Skills Discovery     guidance for how the Discoverer ranks relevance
    Ontological Settings the vocabulary (term: meaning) reasoning works with

Section headings are matched by keyword ("Subagent Spawning", "Spawning",
"Sub-agents" all work), so authors write natural headings, not a schema.
The full prose of every section is kept and surfaced to the Reasoner via
`narrative()` -- extraction only pulls out the handful of values the
machinery itself needs (a capacity, a path, a model id, an env-var name),
and the model id's credential is always an environment-variable *name*,
never a key written in the file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .knowledge import KnowledgeSource
from .mcp_server import McpServer
from .model_binding import ModelBinding
from .ontology import Ontology
from .section import Body, parse_document
from .tool import Tool

_DISABLED = re.compile(
    r"\b(?:do not|don't|never|disabled?|forbidden|prohibited)\b|\bno\s+sub-?agents?\b|\bswitched?\s+off\b",
    re.IGNORECASE,
)
_INTEGER = re.compile(r"\b(\d+)\b")
_BACKTICKED = re.compile(r"`([^`]+)`")
_STORE_PATH = re.compile(r"(?<![\w/])((?:[\w.-]+/)*[\w.-]+\.(?:md|jsonl?|log|db|sqlite))\b")
_ENV_VAR = re.compile(r"\b([A-Z][A-Z0-9_]*(?:KEY|TOKEN)[A-Z0-9_]*)\b")
_MODEL_ID = re.compile(r"(?<![\w./])([A-Za-z][\w-]*)/([A-Za-z][\w.:-]*)(?![\w./])")
_MODEL_TOKEN = re.compile(r"\b([a-z][a-z0-9]*(?:[-.:][a-z0-9]+)+)\b")
_TEMPERATURE = re.compile(r"temperature\s*(?:of|=|:|at)?\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_MAX_TOKENS = re.compile(
    # 'max_tokens: 512' names tokens in the anchor itself, so no trailing
    # unit word is required; 'up to 8000 tokens' does need one, or 'up to
    # 3 subagents' elsewhere in the same prose would false-match.
    r"max[_ ]?tokens\s*(?:of|=|:)?\s*([\d,]+)|"
    r"max(?:imum)?\s+(?:output|response|reply)(?:\s+length)?\s*(?:of|=|:)?\s*([\d,]+)\s*tokens?|"
    r"up\s+to\s*(?:of|=|:)?\s*([\d,]+)\s*tokens?",
    re.IGNORECASE,
)
_URL = re.compile(r"\bhttps?://[^\s`,)]+", re.IGNORECASE)
_STORE_ENABLED = re.compile(r"\bstore\s*[:=]?\s*(?:true|yes|on|enabled|required)\b", re.IGNORECASE)
_CATALOGUE_BACKEND = re.compile(r"\bbackend\s*[:=]\s*([\w.-]+)", re.IGNORECASE)
_CONNECTION_URL = re.compile(r"\b(?:postgres(?:ql)?|age)://[^\s`,)]+", re.IGNORECASE)

# Vocabulary for reading a provider out of prose ("Reason with anthropic's
# claude-opus-4-8..."). The unambiguous form is the provider/model id with a slash
# ("anthropic/claude-opus-4-8"), which needs no vocabulary at all.
_PROVIDERS = (
    "anthropic",
    "openai",
    "azure",
    "gemini",
    "vertex",
    "google",
    "bedrock",
    "mistral",
    "cohere",
    "groq",
    "together",
    "ollama",
    "deepseek",
    "openrouter",
    "xai",
    "fireworks",
    "perplexity",
    "huggingface",
)

# The ten basic toolsets and their shipped defaults -- mirrors an
# operator's Tools Hub: mechanical capabilities enabled or disabled
# platform-wide, never something the model must reason its way into each
# time. An absent Toolsets section keeps these exactly as they are.
_DEFAULT_TOOLSETS = {
    "internet_access": True,
    "internet_search": False,
    "read_documents": True,
    "write_documents": False,
    "code_executor": True,
    "browser_automation": False,
    "terminal": False,
    "email_sender": False,
    "mcp_connector": False,
    "environment_admin": True,
}


def _toolset_key(name: str) -> str:
    """Fold a Toolsets bullet's name to one of the ten canonical keys --
    tolerant of the way people actually write these ('Internet Access
    (Web Fetch)', 'Terminal / Shell', 'Read PDF/Md/Docx/csv/pptx/xlsx')
    rather than demanding an exact identifier. An unrecognized name still
    becomes its own toggle (normalized), never an error -- authoring a
    novel toolset name is never refused."""
    lowered = name.lower()
    if "search" in lowered:
        return "internet_search"
    if "internet" in lowered or "web fetch" in lowered or lowered.strip() in {"fetch", "web"}:
        return "internet_access"
    _document_formats = ("pdf", "doc", "csv", "ppt", "xls", "md", "markdown")
    if any(fmt in lowered for fmt in _document_formats):
        return "write_documents" if "write" in lowered else "read_documents"
    if "code" in lowered:
        return "code_executor"
    if "browser" in lowered or "playwright" in lowered:
        return "browser_automation"
    if "terminal" in lowered or "shell" in lowered:
        return "terminal"
    if "email" in lowered or "mail" in lowered:
        return "email_sender"
    if "mcp" in lowered:
        return "mcp_connector"
    if "environment" in lowered or "stack setup" in lowered:
        return "environment_admin"
    return re.sub(r"[\s/]+", "_", lowered.strip())


@dataclass
class Strategy:
    """The operating strategy read from `memory.md`: every section's prose,
    plus the few values the machinery extracts from that prose."""

    # Context history -> Memory's verbatim window.
    context_history: str = ""
    history_capacity: Optional[int] = None

    # Cross-session data -> SessionStore persistence.
    cross_session: str = ""
    session_enabled: bool = False
    session_path: str = ""

    # Subagent spawning -> Spawner limits.
    subagent_spawning: str = ""
    subagents_configured: bool = False
    subagents_enabled: bool = True
    max_subagents: Optional[int] = None

    # Model selection -> ModelBinding.
    model_selection: str = ""
    provider: str = ""
    model: str = ""
    api_key_env_var: str = ""
    api_base: str = ""
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None

    # Auxiliary model -> a second, usually cheaper ModelBinding a runtime may
    # call for mechanical, non-judgment work (today: compressing a tool
    # result before it re-enters the native tool loop's gathered context).
    # Declaring none is the default and leaves every such feature a no-op.
    auxiliary_model_selection: str = ""
    auxiliary_provider: str = ""
    auxiliary_model: str = ""
    auxiliary_api_key_env_var: str = ""
    auxiliary_api_base: str = ""
    auxiliary_temperature: Optional[float] = None
    auxiliary_max_output_tokens: Optional[int] = None

    # Declared capabilities, surfaced to reasoning.
    mcp_servers: list[McpServer] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    # Whether the runtime may declare new tools for itself at runtime (see
    # ear/acquirer.py) -- on by default (a basic capability, not one the
    # author must opt into), turned off by disabling language under Tools
    # ("fixed toolset", "no new tools", "never create tools").
    tool_acquisition: bool = True

    # Basic toolsets (Toolsets in memory.md): mechanical capabilities --
    # fetch a URL, parse a known file format, send mail -- that ship ready
    # rather than something the model derives itself each time. Keyed by
    # canonical name (see _toolset_key), enabled/disabled per bullet;
    # absent bullets, or an absent section, keep _DEFAULT_TOOLSETS.
    toolsets: dict = field(default_factory=lambda: dict(_DEFAULT_TOOLSETS))
    # internet_search config: provider name (e.g. "tavily") and the
    # environment-variable *name* its API key is read from -- never a key
    # written in memory.md, the same rule Model Selection follows.
    search_provider: str = ""
    search_api_key_env_var: str = ""
    # email_sender config: SMTP host/port and credential env-var names.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user_env_var: str = ""
    smtp_password_env_var: str = ""

    # Skills discovery guidance -> Discoverer.
    skills_discovery: str = ""

    # Reasoning audit trail -> ReasoningLog persistence.
    audit_trail: str = ""
    audit_enabled: bool = False
    audit_path: str = ""
    # Retention declared in the same audit prose ("keep 90 days"): the
    # runner rotates cycles older than this out of the trail, on the record.
    retention_days: Optional[float] = None

    # Knowledge -> the Librarian's reference corpus.
    knowledge: str = ""
    knowledge_sources: list[KnowledgeSource] = field(default_factory=list)

    # Pricing -> dollars on usage records. Rates are the author's
    # declaration, never a price table shipped in code.
    pricing: str = ""
    input_rate_per_million: Optional[float] = None
    output_rate_per_million: Optional[float] = None

    # Execution resilience -> the Journey's leg retry budget, unless a
    # workflow declares its own.
    execution: str = ""
    leg_retry_budget: Optional[int] = None

    # Sandbox -> each runtime instance's isolated workspace and governed
    # command execution. Off unless a Sandbox section is declared.
    sandbox: str = ""
    sandbox_enabled: bool = False
    sandbox_root: str = ""
    sandbox_timeout: Optional[float] = None
    sandbox_memory_mb: Optional[int] = None
    sandbox_ephemeral: bool = False
    sandbox_tools: bool = False

    # Evolution -> governed self-modification (see ear/evolution.py).
    # Absent, or authored with disabling language, leaves the policy None
    # and the runtime refusing every proposed change -- evolution is off
    # unless the author raises the fence and says what fits inside it.
    evolution: str = ""
    evolution_policy: Optional[object] = None

    # Ontological settings -> the reasoning vocabulary.
    ontology: Ontology = field(default_factory=Ontology)

    # Catalogue store -> where Stores (SkillStore, PersonaStore, ...)
    # reads and writes named objects. Off (or absent) keeps the named
    # on-disk catalogue (a directory of markdown files) as the only
    # store, which is why file-based storage is the fallback rather
    # than something a user opts into: it is what happens when this
    # section says nothing. `Store: true` plus a `Backend:` opts into
    # a database-backed store instead; the on-disk catalogue is never
    # required to be replaced, only optionally supplemented.
    catalogue_store: str = ""
    catalogue_store_enabled: bool = False
    catalogue_backend: str = ""
    catalogue_connection: str = ""

    @classmethod
    def from_markdown(cls, text: str) -> "Strategy":
        strategy = cls()
        for section in parse_document(text).sections:
            heading = section.name.lower()
            body = section.body()
            prose = _full_text(body)
            if "ontolog" in heading or "vocabular" in heading:
                strategy._read_ontology(body)
            elif "catalogue" in heading or "catalog" in heading:
                # Checked before the audit/"log" branch below: "catalogue"
                # contains the substring "log" and would otherwise be
                # misread as an audit trail declaration.
                strategy._read_catalogue_store(prose)
            elif "audit" in heading or "log" in heading or "trace" in heading or "trail" in heading:
                strategy._read_audit(prose)
            elif "knowledge" in heading:
                strategy._read_knowledge(body)
            elif "pricing" in heading or "price" in heading or "cost" in heading:
                strategy._read_pricing(prose)
            elif "retry" in heading or "retries" in heading or "resilien" in heading or "execution" in heading:
                strategy._read_execution(prose)
            elif "sandbox" in heading or "isolat" in heading or "workspace" in heading:
                strategy._read_sandbox(prose)
            elif "evol" in heading or "self-modif" in heading or "self modif" in heading:
                strategy._read_evolution(body)
            elif "mcp" in heading:
                strategy._read_mcp(body)
            elif "discover" in heading:
                strategy.skills_discovery = prose
            elif "toolset" in heading:
                # Checked before the "tool" branch below: "toolset"
                # contains the substring "tool" and would otherwise be
                # misread as a Tools declaration.
                strategy._read_toolsets(body)
            elif "tool" in heading:
                strategy._read_tools(body)
            elif "auxiliary" in heading or "summar" in heading:
                # Checked before the plain "model" branch below: "Auxiliary
                # Model" contains the substring "model" and would otherwise
                # overwrite the primary Model Selection's fields.
                strategy._read_auxiliary_model(prose)
            elif "model" in heading:
                strategy._read_model(prose)
            elif "session" in heading or "cross" in heading or "persist" in heading:
                strategy._read_cross_session(prose)
            elif "spawn" in heading or "subagent" in heading or "sub-agent" in heading or "agent" in heading:
                strategy._read_subagents(prose)
            elif "history" in heading or "context" in heading or "memory" in heading:
                strategy._read_context_history(prose)
        return strategy

    # -- section readers ---------------------------------------------------

    def _read_context_history(self, prose: str) -> None:
        self.context_history = prose
        count = _INTEGER.search(prose)
        if count:
            self.history_capacity = int(count.group(1))

    def _read_cross_session(self, prose: str) -> None:
        self.cross_session = prose
        self.session_enabled = not _DISABLED.search(prose)
        self.session_path = _declared_path(prose)

    def _read_audit(self, prose: str) -> None:
        self.audit_trail = prose
        self.audit_enabled = not _DISABLED.search(prose)
        self.audit_path = _declared_path(prose)
        # A retention window declared in the same prose ("keep 90 days",
        # "retain for 6 months"); the phrase must name a period, and only a
        # retention/keep/retain sentence is read so an unrelated number in
        # the audit prose is never mistaken for a window.
        for sentence in prose.replace(";", ".").split("."):
            lowered = sentence.lower()
            if any(word in lowered for word in ("keep", "retain", "retention", "purge", "rotate")):
                days = days_in_prose(sentence)
                if days is not None:
                    self.retention_days = days
                    break

    def _read_knowledge(self, body: Body) -> None:
        self.knowledge = body.prose
        for bullet in body.bullets:
            name, description = _split_declaration(bullet)
            command, url, cleaned = _extract_reach(description)
            if url:
                # A URL source, fetched over the native client and cached;
                # its refresh cadence is declared in the same bullet
                # ("refetch weekly", "refresh every 3 days").
                self.knowledge_sources.append(
                    KnowledgeSource(name=name, url=url, refresh_days=days_in_prose(cleaned))
                )
                continue
            pattern = command or cleaned
            if not pattern:
                raise ValueError(
                    f"Knowledge source '{name}' declares no path -- write '- name: path-or-glob' or '- name: URL'"
                )
            self.knowledge_sources.append(KnowledgeSource(name=name, pattern=pattern))

    def _read_pricing(self, prose: str) -> None:
        """Read token rates from prose. The reliable form is per million:
        'Input tokens cost $3 per million; output tokens cost $15 per
        million.' -- each sentence names input or output, a $ amount, and
        the scale word (million / thousand / token)."""
        self.pricing = prose
        for sentence in prose.replace(";", ".").split("."):
            words = sentence.lower().split()
            amounts = []
            for word in words:
                cleaned = word.strip("$,()")
                if word.startswith("$"):
                    try:
                        amounts.append(float(cleaned))
                    except ValueError:
                        continue
            if not amounts:
                continue
            rate = amounts[0]
            if "thousand" in words or "1k" in words:
                rate *= 1000
            elif "token" in words and "million" not in words and "1m" not in words:
                rate *= 1_000_000
            if "input" in words:
                self.input_rate_per_million = rate
            if "output" in words:
                self.output_rate_per_million = rate

    def dollars(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> Optional[float]:
        """The declared cost of a token spend, or None when no pricing was
        authored -- a dollar figure nobody declared is never invented. Cached
        input is priced off the input rate at the provider-standard multipliers
        (a cache read ~0.1x, a cache write ~1.25x); `input_tokens` is the
        uncached remainder, so the three input counts never double-bill."""
        if self.input_rate_per_million is None and self.output_rate_per_million is None:
            return None
        cost = 0.0
        if self.input_rate_per_million is not None:
            rate = self.input_rate_per_million / 1_000_000
            cost += input_tokens * rate
            cost += cache_read_tokens * rate * 0.1
            cost += cache_write_tokens * rate * 1.25
        if self.output_rate_per_million is not None:
            cost += output_tokens * self.output_rate_per_million / 1_000_000
        return cost

    def _read_sandbox(self, prose: str) -> None:
        """Read the sandbox declaration: whether it is on, where its root
        is (a backticked path), the wall-clock timeout, a memory cap, and
        whether it exposes confined file/shell tools to the model."""
        self.sandbox = prose
        lowered = prose.lower()
        self.sandbox_enabled = not (
            _DISABLED.search(prose)
            or "no sandbox" in lowered
            or "without a sandbox" in lowered
            or "without sandbox" in lowered
        )
        self.sandbox_root = _declared_path(prose)
        self.sandbox_timeout = duration_seconds_in_prose(prose)
        self.sandbox_memory_mb = megabytes_in_prose(prose)
        self.sandbox_ephemeral = any(
            word in lowered for word in ("ephemeral", "temporary", "throwaway", "discard", "cleaned up")
        )
        self.sandbox_tools = any(
            word in lowered
            for word in ("shell", "bash", "command", "file tool", "read and write", "read/write", "expose")
        )

    def _read_evolution(self, body: Body) -> None:
        """Read the Evolution declaration: which kinds of self-modification
        are allowed, prohibited, or human-approval-gated, and which of the
        four requirements (sandbox, evaluation, explanation, rollback) the
        author relaxed -- all default on. Explicit off language ("evolution
        is disabled", "never evolve") leaves the policy None, exactly as if
        the section were absent; the general _DISABLED vocabulary is *not*
        used here because a healthy Evolution section legitimately says
        "prohibited" and "never" about individual kinds."""
        self.evolution = _full_text(body)
        if re.search(
            r"\bevolution\s+is\s+(?:disabled|off|forbidden)\b|\b(?:do not|don't|never)\s+evolve\b|\bno\s+evolution\b",
            body.prose,
            re.IGNORECASE,
        ):
            return
        from .evolution import EvolutionPolicy

        self.evolution_policy = EvolutionPolicy.from_prose(body)

    def _read_catalogue_store(self, prose: str) -> None:
        """Read the catalogue store declaration: 'Store: true' opts into a
        database-backed catalogue instead of the named on-disk one;
        'Backend:' names which (e.g. 'apache-age'); a declared connection
        string wires it up. Absent, or 'Store: false'/no readable 'true',
        leaves `catalogue_store_enabled` False -- the on-disk catalogue
        stays the only store, exactly as if this section did not exist."""
        self.catalogue_store = prose
        self.catalogue_store_enabled = bool(_STORE_ENABLED.search(prose)) and not _DISABLED.search(prose)
        backend = _CATALOGUE_BACKEND.search(prose)
        self.catalogue_backend = backend.group(1).strip().lower() if backend else ""
        self.catalogue_connection = _declared_connection(prose)

    def _read_execution(self, prose: str) -> None:
        """Read the leg retry budget from prose: 'Retry a failed leg twice
        before giving up.' A section with no readable count declares no
        budget -- a journey then keeps its crash-and-resume semantics."""
        self.execution = prose
        self.leg_retry_budget = count_in_prose(prose)

    def _read_subagents(self, prose: str) -> None:
        self.subagent_spawning = prose
        self.subagents_configured = True
        self.subagents_enabled = not _DISABLED.search(prose)
        if self.subagents_enabled:
            count = _INTEGER.search(prose)
            if count:
                self.max_subagents = int(count.group(1))

    @staticmethod
    def _parse_model_prose(prose: str) -> dict:
        """The provider/model id, credential env-var name, API base and
        temperature readable out of a model-declaring section's prose --
        shared by the primary Model Selection and the Auxiliary Model
        sections so both are read by exactly one rule, never two drifting
        copies of it."""
        parsed: dict = {}
        url = _URL.search(prose)
        if url:
            parsed["api_base"] = url.group(0).rstrip(".,;")
        env_var = _ENV_VAR.search(prose)
        if env_var:
            parsed["api_key_env_var"] = env_var.group(1)
        temperature = _TEMPERATURE.search(prose)
        if temperature:
            parsed["temperature"] = float(temperature.group(1))
        max_tokens = _MAX_TOKENS.search(prose)
        if max_tokens:
            digits = next(g for g in max_tokens.groups() if g is not None)
            parsed["max_output_tokens"] = int(digits.replace(",", ""))
        for match in _MODEL_ID.finditer(prose):
            # A model id written at the end of a sentence must not swallow
            # the sentence's period.
            left, right = match.group(1), match.group(2).rstrip(".")
            # A model id either names a known provider or carries a digit
            # ("claude-opus-4-8"); this keeps prose like "approve/decline"
            # from being mistaken for one.
            if left.lower() in _PROVIDERS or any(ch.isdigit() for ch in right):
                parsed["provider"], parsed["model"] = left.lower(), f"{left.lower()}/{right}"
                return parsed
        lowered = prose.lower()
        provider = next((p for p in _PROVIDERS if re.search(rf"\b{p}\b", lowered)), "")
        if provider:
            for token in _MODEL_TOKEN.findall(lowered):
                if token != provider and any(ch.isdigit() for ch in token):
                    parsed["provider"], parsed["model"] = provider, token
                    return parsed
        return parsed

    def _read_model(self, prose: str) -> None:
        self.model_selection = prose
        parsed = self._parse_model_prose(prose)
        self.provider = parsed.get("provider", self.provider)
        self.model = parsed.get("model", self.model)
        self.api_key_env_var = parsed.get("api_key_env_var", self.api_key_env_var)
        self.api_base = parsed.get("api_base", self.api_base)
        if "temperature" in parsed:
            self.temperature = parsed["temperature"]
        if "max_output_tokens" in parsed:
            self.max_output_tokens = parsed["max_output_tokens"]

    def _read_auxiliary_model(self, prose: str) -> None:
        """A second, usually cheaper model a runtime may call for mechanical
        work -- not judgment -- such as compressing a tool result before it
        re-enters the native tool loop's gathered context. Read by exactly
        the same rule as the primary Model Selection (`_parse_model_prose`),
        into its own fields so the two never collide."""
        self.auxiliary_model_selection = prose
        parsed = self._parse_model_prose(prose)
        self.auxiliary_provider = parsed.get("provider", self.auxiliary_provider)
        self.auxiliary_model = parsed.get("model", self.auxiliary_model)
        self.auxiliary_api_key_env_var = parsed.get("api_key_env_var", self.auxiliary_api_key_env_var)
        self.auxiliary_api_base = parsed.get("api_base", self.auxiliary_api_base)
        if "temperature" in parsed:
            self.auxiliary_temperature = parsed["temperature"]
        if "max_output_tokens" in parsed:
            self.auxiliary_max_output_tokens = parsed["max_output_tokens"]

    def _read_mcp(self, body: Body) -> None:
        for bullet in body.bullets:
            name, description = _split_declaration(bullet)
            command, url, description = _extract_reach(description)
            self.mcp_servers.append(McpServer(name=name, description=description, command=command, url=url))

    def _read_tools(self, body: Body) -> None:
        for bullet in body.bullets:
            name, description = _split_declaration(bullet)
            command, _url, description = _extract_reach(description)
            self.tools.append(Tool(name=name, description=description, command=command))
        prose = _full_text(body).lower()
        if prose and (_DISABLED.search(prose) or "fixed toolset" in prose or "no new tools" in prose):
            self.tool_acquisition = False

    def _read_toolsets(self, body: Body) -> None:
        """Read the Toolsets declaration: one bullet per basic toolset,
        'name: enabled' or 'name: disabled' -- mechanical capabilities
        (fetch a URL, parse a known file format, send mail) that ship
        ready rather than something the model derives itself each time.
        Absent bullets, or an absent section entirely, keep the shipped
        defaults (_DEFAULT_TOOLSETS). internet_search's provider/API-key
        env var and email_sender's SMTP config may ride in the same
        bullets' free text."""
        self.toolsets = dict(_DEFAULT_TOOLSETS)
        for bullet in body.bullets:
            name, description = _split_declaration(bullet)
            key = _toolset_key(name)
            lowered = description.lower()
            if _DISABLED.search(description) or "disabled" in lowered or " off" in f" {lowered}":
                self.toolsets[key] = False
            elif "enabled" in lowered or " on" in f" {lowered}":
                self.toolsets[key] = True
        prose = _full_text(body)
        provider = re.search(r"\bprovider\s+(\w+)", prose, re.IGNORECASE)
        if provider:
            self.search_provider = provider.group(1).lower()
        key_env = re.search(r"\bkey env var\s+([A-Z][A-Z0-9_]*)", prose)
        if key_env:
            self.search_api_key_env_var = key_env.group(1)
        host = re.search(r"\bsmtp host\s+(\S+)", prose, re.IGNORECASE)
        if host:
            self.smtp_host = host.group(1).rstrip(".,;")
        port = re.search(r"\bport\s+(\d+)", prose, re.IGNORECASE)
        if port:
            self.smtp_port = int(port.group(1))
        user_env = re.search(r"\buser env var\s+([A-Z][A-Z0-9_]*)", prose)
        if user_env:
            self.smtp_user_env_var = user_env.group(1)
        password_env = re.search(r"\bpassword env var\s+([A-Z][A-Z0-9_]*)", prose)
        if password_env:
            self.smtp_password_env_var = password_env.group(1)

    def _read_ontology(self, body: Body) -> None:
        for bullet in body.bullets:
            term, meaning = _split_declaration(bullet)
            if meaning:
                self.ontology.define(term, meaning)
            else:
                self.ontology.notes = (self.ontology.notes + "\n" + term).strip()
        if body.prose:
            self.ontology.notes = (self.ontology.notes + "\n" + body.prose).strip()

    # -- what the machinery consumes ---------------------------------------

    def model_binding(self) -> Optional[ModelBinding]:
        """The ModelBinding this strategy declares, or None when no model
        was named (the runtime then stays on its deterministic fallback)."""
        if not self.model:
            return None
        params: dict = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            params["max_tokens"] = self.max_output_tokens
        return ModelBinding(
            provider=self.provider,
            model=self.model,
            api_key_env_var=self.api_key_env_var or None,
            api_base=self.api_base or None,
            params=params,
        )

    def auxiliary_model_binding(self) -> Optional[ModelBinding]:
        """The ModelBinding an Auxiliary Model section declares, or None
        when none was named -- the default, which leaves every feature
        that would use it (today: tool-result compression) a no-op."""
        if not self.auxiliary_model:
            return None
        params: dict = {}
        if self.auxiliary_temperature is not None:
            params["temperature"] = self.auxiliary_temperature
        if self.auxiliary_max_output_tokens is not None:
            params["max_tokens"] = self.auxiliary_max_output_tokens
        return ModelBinding(
            provider=self.auxiliary_provider,
            model=self.auxiliary_model,
            api_key_env_var=self.auxiliary_api_key_env_var or None,
            api_base=self.auxiliary_api_base or None,
            params=params,
        )

    def narrative(self) -> str:
        """The strategy rendered for the reasoning prompt: the ontology,
        the declared tools and MCP servers, and the discovery guidance --
        so the model reasons with the enterprise's own vocabulary and knows
        what capabilities are available to it."""
        parts: list[str] = []
        ontology = self.ontology.render()
        if ontology:
            parts.append(ontology)
        if self.tools:
            parts.append("Declared tools:\n" + "\n".join(f"- {tool.describe()}" for tool in self.tools))
        if self.mcp_servers:
            parts.append(
                "MCP servers available:\n" + "\n".join(f"- {server.describe()}" for server in self.mcp_servers)
            )
        if self.skills_discovery:
            parts.append(f"Discovery guidance: {self.skills_discovery}")
        return "\n\n".join(parts)


def _full_text(body: Body) -> str:
    return "\n".join(filter(None, [body.prose] + body.bullets + body.numbered))


def _declared_path(prose: str) -> str:
    """The store path a section declares. Backticked paths win; among bare
    mentions, one with a directory part wins, so prose that merely mentions
    a stack file like `memory.md` is never mistaken for the store."""
    for candidate in _BACKTICKED.findall(prose):
        if _STORE_PATH.fullmatch(candidate) or "/" in candidate:
            return candidate
    matches = [match.group(1) for match in _STORE_PATH.finditer(prose)]
    for match in matches:
        if "/" in match:
            return match
    return matches[0] if matches else ""


def _declared_connection(prose: str) -> str:
    """The database connection string a Catalogue Store section declares
    ('Connection: postgresql://user:pass@host/db'), backticked or bare."""
    for candidate in _BACKTICKED.findall(prose):
        if _CONNECTION_URL.fullmatch(candidate):
            return candidate
    match = _CONNECTION_URL.search(prose)
    return match.group(0) if match else ""


# Refresh cadences a URL knowledge source may declare in prose, in days.
# Plain word scanning -- "refetch weekly", "refresh every 3 days" -- so the
# author writes a sentence, not a schema.
_CADENCE_DAYS = {
    "hourly": 1 / 24,
    "daily": 1.0,
    "nightly": 1.0,
    "weekly": 7.0,
    "fortnightly": 14.0,
    "monthly": 30.0,
    "quarterly": 91.0,
    "yearly": 365.0,
    "annually": 365.0,
}
_UNIT_DAYS = {"hour": 1 / 24, "day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0}


def days_in_prose(text: str) -> Optional[float]:
    """The period a sentence of prose declares, in days -- a cadence word
    ("weekly"), or a count with a unit ("every 3 days", "after 2 weeks").
    None when no period was authored. Used for knowledge refresh cadences
    and approval escalation deadlines alike."""
    words = [word.strip(".,;:()").lower() for word in text.split()]
    for word in words:
        if word in _CADENCE_DAYS:
            return _CADENCE_DAYS[word]
    for position, word in enumerate(words[:-1]):
        try:
            count = float(word)
        except ValueError:
            continue
        unit = words[position + 1].rstrip("s")
        if unit in _UNIT_DAYS:
            return count * _UNIT_DAYS[unit]
    return None


# Spoken counts a declaration may use ("retry twice", "no retries").
_COUNT_WORDS = {
    "no": 0,
    "never": 0,
    "zero": 0,
    "once": 1,
    "one": 1,
    "twice": 2,
    "two": 2,
    "thrice": 3,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def count_in_prose(text: str) -> Optional[int]:
    """The first count a sentence of prose declares -- a digit or a spoken
    number ("retry a failed leg twice" -> 2, "no retries" -> 0). None when
    no count was authored."""
    for word in (word.strip(".,;:()").lower() for word in text.split()):
        if word.isdigit():
            return int(word)
        if word in _COUNT_WORDS:
            return _COUNT_WORDS[word]
    return None


_TIME_UNITS = {
    "second": 1,
    "seconds": 1,
    "sec": 1,
    "secs": 1,
    "minute": 60,
    "minutes": 60,
    "min": 60,
    "mins": 60,
    "hour": 3600,
    "hours": 3600,
}


def duration_seconds_in_prose(text: str) -> Optional[float]:
    """A wall-clock duration in seconds from prose -- "time out after 30
    seconds", "2 minutes", or a bare number (read as seconds). None when
    no number is present."""
    words = [word.strip(".,;:()").lower() for word in text.split()]
    for position, word in enumerate(words[:-1]):
        try:
            count = float(word)
        except ValueError:
            continue
        if words[position + 1] in _TIME_UNITS:
            return count * _TIME_UNITS[words[position + 1]]
    for word in words:
        if word.replace(".", "", 1).isdigit():
            return float(word)
    return None


def megabytes_in_prose(text: str) -> Optional[int]:
    """A memory cap in megabytes from prose -- "limit memory to 512 MB",
    "1 GB", or glued forms like "512mb". None when none is declared."""
    tokens = text.lower().replace("mb", " mb ").replace("gb", " gb ").split()
    for position, token in enumerate(tokens):
        try:
            amount = float(token.strip(".,;:()"))
        except ValueError:
            continue
        unit = tokens[position + 1] if position + 1 < len(tokens) else ""
        if unit == "gb":
            return int(amount * 1024)
        if unit == "mb":
            return int(amount)
    return None


def _split_declaration(bullet: str) -> tuple[str, str]:
    """Split a declaration bullet into (name, description) on the first
    ':' or dash separator; a bullet with no separator is all name."""
    for separator in (":", "—", "–", " -- ", " - "):
        if separator in bullet:
            name, description = bullet.split(separator, 1)
            if name.strip() and len(name.strip()) <= 60:
                return name.strip(), description.strip()
    return bullet.strip(), ""


def _extract_reach(description: str) -> tuple[str, str, str]:
    """Pull a backticked command and/or a URL out of a declaration's
    description, returning (command, url, cleaned description)."""
    command = ""
    backticked = _BACKTICKED.search(description)
    if backticked:
        command = backticked.group(1).strip()
    url = ""
    url_match = _URL.search(description)
    if url_match:
        url = url_match.group(0).rstrip(".,;")
    cleaned = _BACKTICKED.sub("", description)
    cleaned = _URL.sub("", cleaned)
    cleaned = re.sub(r"[,;]?\s*(?:via|using|through|over|at)\s*$", "", cleaned.strip(), flags=re.IGNORECASE)
    return command, url, cleaned.strip(" ,;")
