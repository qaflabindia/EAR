"""MCPToolset -- reach an MCP (Model Context Protocol) server's tools as
governed EAR Tools.

The Tool/ToolPolicy/Invoker foundation (see `ear/tool.py`) is what makes
this cheap: an MCPToolset needs no governance story of its own, because
every tool it discovers is wrapped as an ordinary `Tool`, gated by the same
ToolPolicy/Governor/Invoker path as any tool declared by hand. This is the
payoff of building governed actions first -- MCP becomes "provider-agnostic
tools" the same way `ModelBinding`/`Router` are "provider-agnostic models":
reach a whole ecosystem of tools without EAR authoring each one, and every
call still clears the Governor and lands in Evidence.

Config is env-driven, never hardcoded, matching the rest of the package --
`from_env`/`from_spec` mirror `Router.from_env`/`from_spec`.

The `mcp` SDK is a lazy import: `import ear`, declaring an `MCPToolset`,
and even calling `Tool.describe()` on one of its wrapped tools all work
without it installed. Only `.tools()` (which connects to the server) or
invoking a discovered tool's handler needs `pip install -e '.[mcp]'`.

Each call to `.tools()` or a wrapped tool's handler opens a fresh
connection, does the one operation, and closes it. That trades connection
reuse for the simplest, statelessness-preserving design -- no session
lifecycle for a Runtime to manage -- which fits how the rest of EAR is
built (a Runtime is a plain dataclass, not a service with a lifecycle)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .tool import Tool


@dataclass
class MCPToolset:
    """Connects to one MCP server and exposes its tools as governed EAR
    Tools. Give it `url` (HTTP, streamable) for a remote server, or
    `command` (+ `args`) to launch a local stdio server -- exactly one of
    the two."""

    url: Optional[str] = None
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    api_key_env_var: Optional[str] = None
    label: str = ""

    @classmethod
    def from_env(cls, var: str = "EAR_MCP_SERVERS") -> list["MCPToolset"]:
        """Build one MCPToolset per server described in the environment
        variable `var` -- config lives in the environment, never
        hardcoded, exactly like `Router.from_env`."""
        spec = os.environ.get(var)
        if not spec or not spec.strip():
            raise ValueError(f"No MCP server spec found in environment variable {var!r}")
        return cls.from_spec(spec)

    @classmethod
    def from_spec(cls, spec: str) -> list["MCPToolset"]:
        """Parse a JSON array of server specs into MCPToolsets, e.g.

        ``[{"label": "files", "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]},
           {"label": "search", "url": "https://mcp.example.com/mcp",
            "api_key_env_var": "SEARCH_MCP_KEY"}]``
        """
        entries = json.loads(spec)
        return [
            cls(
                url=entry.get("url"),
                command=entry.get("command"),
                args=entry.get("args", []),
                api_key_env_var=entry.get("api_key_env_var"),
                label=entry.get("label", ""),
            )
            for entry in entries
        ]

    @classmethod
    def tools_from_env(cls, var: str = "EAR_MCP_SERVERS") -> list[Tool]:
        """Convenience: every tool from every server named in `var`,
        flattened into one list -- `persona.tools.extend(...)`."""
        return [tool for toolset in cls.from_env(var) for tool in toolset.tools()]

    def resolve_api_key(self) -> Optional[str]:
        if self.api_key_env_var is None:
            return None
        return os.environ.get(self.api_key_env_var)

    def tools(self) -> list[Tool]:
        """Connect to the server, list its tools, and wrap each as a
        governed EAR Tool whose handler calls back into the server."""
        descriptions = asyncio.run(self._list_tool_descriptions())
        return [self._wrap(description) for description in descriptions]

    def _wrap(self, description: Any) -> Tool:
        """Wrap one MCP tool description (an object with `name` and
        `description` attributes -- `mcp.types.Tool`, or any stand-in with
        the same shape) into a governed EAR Tool. Pure and dependency-free:
        takes no `mcp` import, so it is testable without the package or a
        live server."""
        name = description.name
        contract = getattr(description, "description", None) or name

        def handler(**arguments: Any) -> Any:
            return asyncio.run(self._call_tool(name, arguments))

        return Tool(name=name, contract=contract, handler=handler, permissions=[f"mcp:{self.label or name}"])

    async def _open(self, stack: contextlib.AsyncExitStack) -> Any:
        import mcp

        if self.url:
            from mcp.client.streamable_http import streamablehttp_client

            headers = {"Authorization": f"Bearer {self.resolve_api_key()}"} if self.resolve_api_key() else {}
            read, write, _ = await stack.enter_async_context(streamablehttp_client(self.url, headers=headers))
        elif self.command:
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(command=self.command, args=self.args)
            read, write = await stack.enter_async_context(stdio_client(params))
        else:
            raise ValueError("MCPToolset needs either `url` or `command` set")
        session = await stack.enter_async_context(mcp.ClientSession(read, write))
        await session.initialize()
        return session

    async def _list_tool_descriptions(self) -> list[Any]:
        async with contextlib.AsyncExitStack() as stack:
            session = await self._open(stack)
            result = await session.list_tools()
            return list(result.tools)

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        # Rendered (and, on an MCP-side error, raised) only after the stack
        # has closed: raising while the transport's TaskGroup is still
        # tearing down gets anyio to wrap it in an opaque ExceptionGroup
        # instead of the clean RuntimeError callers should see.
        async with contextlib.AsyncExitStack() as stack:
            session = await self._open(stack)
            result = await session.call_tool(name, arguments)
        return self._render_result(result)

    @staticmethod
    def _render_result(result: Any) -> str:
        """Flatten an MCP CallToolResult's content blocks into one string.
        Duck-typed on `.isError`/`.content`/block `.text`, so it is
        testable with a plain fake result -- no `mcp` import needed."""
        if getattr(result, "isError", False):
            raise RuntimeError(f"MCP tool call failed: {result.content}")
        parts = [getattr(block, "text", None) or str(block) for block in getattr(result, "content", [])]
        return "\n".join(parts) if parts else str(result)
