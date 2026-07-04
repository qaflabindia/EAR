"""A tiny local MCP server used only by tests, over stdio -- no network.

Exercises the real `mcp` SDK on the server side so `MCPToolset` is tested
against a genuine MCP round-trip, not a mock, while staying fully offline.
"""

from mcp.server.fastmcp import FastMCP

server = FastMCP("ear-test-server")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def fail() -> str:
    """Always raises, to exercise the MCP error path."""
    raise ValueError("boom")


if __name__ == "__main__":
    server.run(transport="stdio")
