"""tools_cli -- list, view, create and retire a stack's tools from the
command line: the same four operations the Acquirer exposes to the model
itself (`ear/acquirer.py`), for a human operating on a stack directly.

    python -m ear.tools_cli list   <stack-dir>
    python -m ear.tools_cli view   <stack-dir> <name>
    python -m ear.tools_cli create <stack-dir> <name> <description> [command]
    python -m ear.tools_cli retire <stack-dir> <name> [reason]

Exit code is 0 on success, 2 on a usage error."""

from __future__ import annotations

import sys
from typing import Optional


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    action, stack, rest = argv[0], argv[1], argv[2:]

    from .loader import load_runtime

    runtime = load_runtime(stack)
    acquirer = runtime.acquirer

    if action == "list":
        print(acquirer.list_tools(runtime))
    elif action == "view":
        if not rest:
            print("usage: view <stack-dir> <name>", file=sys.stderr)
            return 2
        print(acquirer.view_tool(runtime, rest[0]))
    elif action == "create":
        if len(rest) < 2:
            print("usage: create <stack-dir> <name> <description> [command]", file=sys.stderr)
            return 2
        name, description = rest[0], rest[1]
        command = rest[2] if len(rest) > 2 else ""
        print(acquirer.create_tool(runtime, name, description, command))
    elif action == "retire":
        if not rest:
            print("usage: retire <stack-dir> <name> [reason]", file=sys.stderr)
            return 2
        name = rest[0]
        reason = rest[1] if len(rest) > 1 else ""
        print(acquirer.retire_tool(runtime, name, reason))
    else:
        print(f"unknown command '{action}' -- expected list, view, create, or retire", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
