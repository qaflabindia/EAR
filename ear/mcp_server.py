"""McpServer -- an MCP (Model Context Protocol) server declared to the
runtime in plain English.

MCP servers are stacked in `memory.md` under the MCP strategy section: one
bullet per server, `name: what it provides`, with the launch command
backticked or a URL written inline. Like Tools, they are surfaced to the
Reasoner as part of the operating strategy so the model reasons about them
in natural language; the declaration records how to reach the server for
integrations that connect to it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class McpServer:
    """One declared MCP server: a name, a plain-English description of what
    it provides, and how to reach it (a launch command or a URL)."""

    name: str
    description: str = ""
    command: str = ""
    url: str = ""

    def describe(self) -> str:
        line = self.name
        if self.description:
            line += f": {self.description}"
        reach = self.command or self.url
        if reach:
            line += f" (reached via `{reach}`)"
        return line
