"""bind_langchain_tool -- put a LangChain tool behind a declared EAR tool.

The ToolBinder's rule holds for platforms too: the natural-language stack
declares what exists (a Tools bullet in memory.md), and this adapter only
supplies the executable behind that declaration. The LangChain tool is
duck-typed (`.name`, `.description`, `.invoke` or `.run`), so any
LangChain community tool -- or anything shaped like one -- binds without
importing LangChain here; install what you use (`pip install
'ear[langchain]'` conventions apply to your own tool's dependencies).

The bound handler takes one plain-text query and returns the tool's text
result; the ToolBinder wraps it so every invocation lands on the reasoning
trail and a failure returns to the model as text.
"""

from __future__ import annotations

from typing import Any, Optional


def bind_langchain_tool(binder: Any, langchain_tool: Any, name: Optional[str] = None) -> Any:
    """Bind `langchain_tool` onto the declared tool called `name` (the
    LangChain tool's own name by default). Returns the binder."""
    tool_name = name or getattr(langchain_tool, "name", None) or type(langchain_tool).__name__

    def call(query: str) -> Any:
        invoke = getattr(langchain_tool, "invoke", None)
        if callable(invoke):
            return invoke(query)
        return langchain_tool.run(query)

    call.__doc__ = getattr(langchain_tool, "description", "") or f"The LangChain tool '{tool_name}'."
    return binder.bind(tool_name, call)