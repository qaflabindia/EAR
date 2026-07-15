"""Tests for Phase 2 of the Enterprise AGI binding:

* `ear/compiler.py` -- compiling a whole command centre into an EAR stack.
* `ear/mcp_command_centre.py` -- serving a centre as a native MCP server.

All offline: the compiler is structural and the MCP server runs the
constitution's deterministic fallbacks, so nothing here needs a model or a
credential. The one live path (a compiled centre reasoning against a real
model) is already covered by the loader's own live tests.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from ear import Governor, Intent
from ear.compiler import CompiledStack, StackCompiler, compile_command_centre
from ear.mcp_command_centre import CommandCentreServer

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "command_centres"
AFCC = FIXTURES / "afcc"
AGCC = FIXTURES / "agcc"


# ---------------------------------------------------------------------------
# The Centre -> EAR-stack compiler.
# ---------------------------------------------------------------------------


def test_compile_writes_the_full_stack(tmp_path):
    stack = compile_command_centre(AFCC, tmp_path, verify=True)
    assert isinstance(stack, CompiledStack)
    for filename in ("skills.md", "persona.md", "policy.md", "workflow.md", "process.md", "memory.md"):
        assert (tmp_path / filename).exists(), filename


def test_capabilities_become_skills(tmp_path):
    stack = compile_command_centre(AFCC, tmp_path)
    assert stack.skills == ["classify_expense", "check_budget", "decide_expense", "write_requester_note"]
    skills_text = (tmp_path / "skills.md").read_text()
    assert "## classify_expense" in skills_text
    assert "expense taxonomy" in skills_text


def test_procedures_become_a_workflow_delegated_to_the_persona(tmp_path):
    compile_command_centre(AFCC, tmp_path)
    workflow = (tmp_path / "workflow.md").read_text()
    assert "## Expenditure Approval Workflow" in workflow
    # Every step is delegated to the single compiled persona.
    assert workflow.count("(Finance Operator)") == 4


def test_constitution_becomes_policy_md(tmp_path):
    compile_command_centre(AFCC, tmp_path)
    policy = (tmp_path / "policy.md").read_text()
    assert "CR-FIN-01" in policy
    assert "Applies to: runtime" in policy


def test_references_become_knowledge_documents(tmp_path):
    stack = compile_command_centre(AFCC, tmp_path)
    assert set(stack.knowledge) == {"expense-taxonomy.md", "approval-matrix.md"}
    assert (tmp_path / "knowledge" / "expense-taxonomy.md").exists()
    # The constitution is compiled to policy.md, never duplicated as knowledge.
    assert not (tmp_path / "knowledge" / "constitutional_rules.md").exists()
    # memory.md declares the knowledge sources.
    memory = (tmp_path / "memory.md").read_text()
    assert "knowledge/expense-taxonomy.md" in memory


def test_frontmatter_org_becomes_tenant(tmp_path):
    compile_command_centre(AFCC, tmp_path)
    tenant = (tmp_path / "tenant.md").read_text()
    assert "Org id: acme-corp" in tenant
    assert "Fiscal year start: 2026-04-01" in tenant


def test_triggers_prose_folds_into_the_persona_not_dropped(tmp_path):
    # A section the compiler does not consume structurally (Triggers) is
    # folded into the persona's instructions rather than silently dropped.
    compile_command_centre(AFCC, tmp_path)
    persona = (tmp_path / "persona.md").read_text()
    assert "Triggers:" in persona
    assert "purchase-order approval request" in persona
    # ...and folded as prose, never as bullets (which the loader would read
    # as skill references).
    persona_body = persona.split("## Finance Operator", 1)[1].split("Skills:", 1)[0]
    assert "\n- " not in persona_body


def test_compiled_stack_loads_and_enforces_its_constitution(tmp_path):
    stack = compile_command_centre(AFCC, tmp_path, verify=True)
    runtime = stack.load()
    assert len(runtime.policies) == 5
    assert len(runtime.processes) == 1

    # An expense above $25,000 parks under CR-FIN-03 on the compiled stack.
    intent = Intent(
        text="approve a large expense",
        context={"expense_amount": 40000, "remaining_budget": 90000, "documents_complete": True},
    )
    parked = Governor().govern(runtime, intent)
    assert any(p.name.startswith("CR-FIN-03") for p in parked)


def test_compiled_stack_clears_a_clean_expense(tmp_path):
    stack = compile_command_centre(AFCC, tmp_path)
    runtime = stack.load()
    intent = Intent(
        text="a small in-budget expense with complete documents",
        context={"expense_amount": 1200, "remaining_budget": 90000, "documents_complete": True},
    )
    assert Governor().govern(runtime, intent) == []


def test_mapping_report_names_every_produced_file(tmp_path):
    stack = compile_command_centre(AFCC, tmp_path)
    assert stack.mapping["policy.md"] == "references/constitutional_rules.md"
    assert stack.mapping["skills.md"] == "SKILL.md ## Capabilities"
    assert "workflow.md" in stack.mapping


def test_centre_without_procedures_still_compiles_a_runnable_workflow(tmp_path):
    # A centre carrying capabilities but no ## Procedures still yields one
    # workflow, so it loads and runs.
    centre = tmp_path / "minc"
    (centre / "references").mkdir(parents=True)
    (centre / "SKILL.md").write_text(
        "---\nname: Minimal Centre\n---\n\n# Minimal Centre\n\nDo the work.\n\n"
        "## Capabilities\n\n### do_it\n\nDo the one thing.\n",
        encoding="utf-8",
    )
    (centre / "references" / "constitutional_rules.md").write_text(
        "# Rules\n\n## CR-M1 -- Always audit\n\nVerdict: HALT\n\nRecord before acting.\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    stack = compile_command_centre(centre, out, verify=True)
    assert len(stack.workflows) == 1
    assert stack.load().processes


def test_compiler_reused_across_centres(tmp_path):
    # AGCC (governance) compiles too -- it has a constitution but no
    # capabilities/procedures, so it degrades to a single delegating workflow.
    stack = compile_command_centre(AGCC, tmp_path, verify=True)
    runtime = stack.load()
    assert len(runtime.policies) == 8


# ---------------------------------------------------------------------------
# MCP packaging -- a centre served as an MCP server.
# ---------------------------------------------------------------------------


def _afcc_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "afcc"
    shutil.copytree(AFCC, destination)
    return destination


def test_server_advertises_the_pentad():
    server = CommandCentreServer.load(AFCC)
    names = {tool["name"] for tool in server.tools()}
    assert names == {"list_state", "load_state", "update_state", "evaluate", "audit"}


def test_list_and_load_state():
    server = CommandCentreServer.load(AFCC)
    text, is_error = server.call("list_state", {})
    assert not is_error
    assert "budgets" in text
    loaded, _ = server.call("load_state", {"name": "budgets"})
    assert json.loads(loaded)["categories"]["travel"]["remaining"] == 70000


def test_evaluate_runs_the_constitution_fallbacks():
    server = CommandCentreServer.load(AFCC)
    text, is_error = server.call(
        "evaluate",
        {"context": json.dumps({"expense_amount": 40000, "remaining_budget": 5000, "documents_complete": True})},
    )
    assert not is_error
    report = json.loads(text)
    assert report["passed"] is False
    assert any("CR-FIN-02" in name for name in report["violations"])


def test_update_state_round_trips(tmp_path):
    centre = _afcc_copy(tmp_path)
    server = CommandCentreServer.load(centre)
    _text, is_error = server.call("update_state", {"name": "budgets", "value": json.dumps({"note": "frozen"})})
    assert not is_error
    assert server.centre.state.read_json("budgets") == {"note": "frozen"}


def test_audit_appends_to_the_ledger(tmp_path):
    centre = _afcc_copy(tmp_path)
    server = CommandCentreServer.load(centre)
    before = server.centre.state.audit_trail_path.read_text().count("\n")
    server.call("audit", {"entry": json.dumps({"action": "test", "verdict": "EXECUTE"})})
    after = server.centre.state.audit_trail_path.read_text().count("\n")
    assert after == before + 1


def test_unknown_tool_is_a_loud_error():
    server = CommandCentreServer.load(AFCC)
    text, is_error = server.call("nonexistent", {})
    assert is_error
    assert "unknown tool" in text


def test_missing_state_is_a_loud_error():
    server = CommandCentreServer.load(AFCC)
    text, is_error = server.call("load_state", {"name": "does-not-exist"})
    assert is_error


def test_served_over_the_native_mcp_client():
    # The real round trip: launch the centre as a subprocess and drive it
    # over EAR's own stdio JSON-RPC McpClient.
    from ear.mcp_client import McpClient

    command = [sys.executable, "-m", "ear.mcp_command_centre", str(AFCC)]
    with McpClient(command=command) as client:
        tools = {tool.name for tool in client.list_tools()}
        assert "evaluate" in tools
        loaded = client.call_tool("load_state", {"name": "vendor_registry"})
        assert "Skyline Travel" in loaded
        report = json.loads(
            client.call_tool(
                "evaluate",
                {"context": json.dumps({"expense_amount": 100, "remaining_budget": 90000, "documents_complete": True})},
            )
        )
        assert report["passed"] is True
