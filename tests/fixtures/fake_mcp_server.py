"""A minimal MCP server over stdio for tests -- stdlib only, no EAR import.

Speaks just enough JSON-RPC to exercise EAR's native McpClient: it answers
`initialize`, `tools/list` (one tool, `add`, with an integer schema) and
`tools/call` (adds two numbers, or reports an error for a bad tool).
"""
import json
import sys


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        method = message.get("method")
        if method == "notifications/initialized":
            continue
        request_id = message.get("id")
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "fake"}}
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "add",
                        "description": "Add two integers a and b.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        },
                    }
                ]
            }
        elif method == "tools/call":
            params = message.get("params") or {}
            if params.get("name") != "add":
                result = {"content": [{"type": "text", "text": "unknown tool"}], "isError": True}
            else:
                args = params.get("arguments") or {}
                total = int(args.get("a", 0)) + int(args.get("b", 0))
                result = {"content": [{"type": "text", "text": f"sum is {total}"}]}
        else:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "no"}}) + "\n")
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
