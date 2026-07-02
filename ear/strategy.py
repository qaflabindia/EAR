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
_URL = re.compile(r"\bhttps?://[^\s`,)]+", re.IGNORECASE)

# Vocabulary for reading a provider out of prose ("Reason with anthropic's
# claude-opus-4-8..."). The unambiguous form is the LiteLLM id with a slash
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

    # Declared capabilities, surfaced to reasoning.
    mcp_servers: list[McpServer] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)

    # Skills discovery guidance -> Discoverer.
    skills_discovery: str = ""

    # Reasoning audit trail -> ReasoningLog persistence.
    audit_trail: str = ""
    audit_enabled: bool = False
    audit_path: str = ""

    # Ontological settings -> the reasoning vocabulary.
    ontology: Ontology = field(default_factory=Ontology)

    @classmethod
    def from_markdown(cls, text: str) -> "Strategy":
        strategy = cls()
        for section in parse_document(text).sections:
            heading = section.name.lower()
            body = section.body()
            prose = _full_text(body)
            if "ontolog" in heading or "vocabular" in heading:
                strategy._read_ontology(body)
            elif "audit" in heading or "log" in heading or "trace" in heading or "trail" in heading:
                strategy._read_audit(prose)
            elif "mcp" in heading:
                strategy._read_mcp(body)
            elif "discover" in heading:
                strategy.skills_discovery = prose
            elif "tool" in heading:
                strategy._read_tools(body)
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

    def _read_subagents(self, prose: str) -> None:
        self.subagent_spawning = prose
        self.subagents_configured = True
        self.subagents_enabled = not _DISABLED.search(prose)
        if self.subagents_enabled:
            count = _INTEGER.search(prose)
            if count:
                self.max_subagents = int(count.group(1))

    def _read_model(self, prose: str) -> None:
        self.model_selection = prose
        url = _URL.search(prose)
        if url:
            self.api_base = url.group(0).rstrip(".,;")
        env_var = _ENV_VAR.search(prose)
        if env_var:
            self.api_key_env_var = env_var.group(1)
        temperature = _TEMPERATURE.search(prose)
        if temperature:
            self.temperature = float(temperature.group(1))
        for match in _MODEL_ID.finditer(prose):
            left, right = match.group(1), match.group(2)
            # A model id either names a known provider or carries a digit
            # ("claude-opus-4-8"); this keeps prose like "approve/decline"
            # from being mistaken for one.
            if left.lower() in _PROVIDERS or any(ch.isdigit() for ch in right):
                self.provider, self.model = left.lower(), f"{left.lower()}/{right}"
                return
        lowered = prose.lower()
        provider = next((p for p in _PROVIDERS if re.search(rf"\b{p}\b", lowered)), "")
        if provider:
            for token in _MODEL_TOKEN.findall(lowered):
                if token != provider and any(ch.isdigit() for ch in token):
                    self.provider, self.model = provider, token
                    return

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
        return ModelBinding(
            provider=self.provider,
            model=self.model,
            api_key_env_var=self.api_key_env_var or None,
            api_base=self.api_base or None,
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
