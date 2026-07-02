"""compile_to_graph / runtime_node -- run the stack as a LangGraph.

Two shapes, both built on one rule: **a graph node never bypasses
governance.** Every node runs a full governed cycle on the runtime --
Governor gates (approval gates included), the Librarian's knowledge,
bound tools, the ReasoningLog and Memory all apply inside each node
exactly as they do natively. LangGraph contributes what it is best at:
checkpointing between steps, halt-on-block routing, and interop with
graphs a team already runs.

- `runtime_node(runtime)` exposes the whole runtime as one node for a
  larger LangGraph app: state in (`intent`, `context`), state out
  (`decision`, `status`) -- a refusal or a parked approval is a status,
  not an exception, so the surrounding graph can route on it.
- `compile_to_graph(runtime, checkpointer=...)` compiles the authored
  stack itself: one node per workflow step, in authored order, each step
  reasoned as its own governed cycle with the overall intent and the
  earlier steps' conclusions carried in the state. Any non-decided status
  (BLOCKED, PENDING APPROVAL) routes the graph to END -- a gate that
  stops a cycle stops the graph.

Requires `pip install 'ear[langgraph]'`. Pass any LangGraph checkpointer
(e.g. `MemorySaver`) to persist and resume between steps; a
SessionStore-backed checkpointer remains on the plan.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypedDict

from ..approval import ApprovalRequired
from ..intent import Intent


class StackState(TypedDict, total=False):
    """The state a compiled stack graph carries between nodes."""

    intent: str
    context: dict
    steps: list
    decision: str
    status: str


def runtime_node(runtime: Any) -> Callable[[StackState], StackState]:
    """The whole runtime as one LangGraph node: a full governed cycle per
    invocation, with refusals and parked approvals returned as `status`."""

    def node(state: StackState) -> StackState:
        intent = Intent(text=str(state.get("intent", "")), context=dict(state.get("context") or {}))
        decision, status = _governed_cycle(runtime, intent)
        return {"decision": decision, "status": status}

    return node


def compile_to_graph(runtime: Any, checkpointer: Optional[Any] = None) -> Any:
    """Compile the authored stack into a LangGraph: one node per workflow
    step in authored order, each a full governed cycle, halting at the
    first non-decided status."""
    from langgraph.graph import END, StateGraph

    authored = [
        (workflow, number, step)
        for process in getattr(runtime, "processes", [])
        for workflow in process.workflows
        for number, step in enumerate(workflow.steps, start=1)
    ]
    if not authored:
        raise ValueError(f"Runtime '{runtime.name}' has no workflow steps to compile into a graph")

    graph = StateGraph(StackState)
    names: list[str] = []
    for workflow, number, step in authored:
        name = _node_name(workflow.name, number)
        graph.add_node(name, _step_node(runtime, workflow, step))
        names.append(name)
    graph.set_entry_point(names[0])
    for current, following in zip(names, names[1:]):
        graph.add_conditional_edges(current, _proceed_or_end, {"proceed": following, "end": END})
    graph.add_edge(names[-1], END)
    return graph.compile(checkpointer=checkpointer)


def _step_node(runtime: Any, workflow: Any, step: Any) -> Callable[[StackState], StackState]:
    def node(state: StackState) -> StackState:
        prior = list(state.get("steps") or [])
        text = step.instruction
        overall = str(state.get("intent", ""))
        if overall:
            text += f"\n\nThe overall intent this step serves: {overall}"
        if prior:
            transcript = "\n".join(f"- {entry['step']} -> {entry['decision']}" for entry in prior)
            text += f"\n\nEarlier steps concluded:\n{transcript}"
        intent = Intent(text=text, context=dict(state.get("context") or {}))
        decision, status = _governed_cycle(runtime, intent)
        entry = {"workflow": workflow.name, "step": step.instruction, "decision": decision, "status": status}
        return {"steps": prior + [entry], "decision": decision, "status": status}

    return node


def _governed_cycle(runtime: Any, intent: Intent) -> tuple[str, str]:
    """One cycle through the runtime, with governance outcomes expressed
    as graph-routable statuses instead of exceptions -- everything still
    lands on the trail exactly as it does natively."""
    try:
        return str(runtime.reason(intent)), "decided"
    except ApprovalRequired as parked:
        return str(parked), "PENDING APPROVAL"
    except PermissionError as blocked:
        return str(blocked), "BLOCKED"


def _proceed_or_end(state: StackState) -> str:
    return "proceed" if state.get("status") == "decided" else "end"


def _node_name(workflow_name: str, number: int) -> str:
    mapped = "".join(ch if ch.isalnum() else "_" for ch in workflow_name.strip().lower())
    return f"{mapped or 'workflow'}_step_{number}"