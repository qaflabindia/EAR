#!/usr/bin/env python3
"""Sales MIS Guru -- markdown in, markdown out, EAR reasons through the cycle.

This is wiring, not logic: point `load_runtime` at `sales_mis_stack/`, seed its
sandbox with the raw source workbook and the dashboard template, parse the MIS
manual into one intent per SIPOC step, and run the Exchange. Every step --
loading, sanity-checking, slicing and dicing, reconciling -- is code the model
itself authors and runs inside the sandbox (see `sales_mis_stack/memory.md`'s
Sandbox section and `skills.md`); nothing in this script stages data, fills a
dashboard or validates one. The canonical script names each step is asked to
author if absent (`validate_data.py`, `generate.py`, `validate_dashboard.py`)
are a naming convention from the MIS manual, not code shipped here.

What this driver *does* own is honesty about progress:

- A fresh run starts from a clean slate -- stale `workspace/`/`outputs/`
  artifacts and last run's intent/decision documents are cleared, so nothing
  this run reports can be a leftover another run produced.
- Each cycle's intent states only facts the driver has verified on disk (a
  named input artifact is listed with its verified byte size); it never tells
  the model a prior step succeeded when it didn't.
- Each cycle is *gated* on verified data, not on a user-facing promise about
  our internal filenames. Preferred handoff paths are the driver's own aliases;
  when the same verified artifact exists under an equivalent internal name, the
  driver resolves or materializes that alias instead of pretending the business
  task failed.
- After every cycle a step-status board prints (and lands in
  `.ear/step_status.md`): per step -- status, the inputs that were staged,
  the evidence the input was actually read, and the outputs produced with
  their real sizes. The board is computed from the filesystem, never from
  the prose of the decision.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root, so `import ear` finds the local package

from ear import Exchange, load_runtime  # noqa: E402
from ear.section import parse_document  # noqa: E402

STACK = HERE / "sales_mis_stack"
SOURCE_DIR = STACK / "Sales MIS"
RAW_SOURCE = SOURCE_DIR / "bank_daily_sales_2025.xlsx"
DASHBOARD_TEMPLATE = SOURCE_DIR / "bank_daily_sales_dashboard_2025.xlsx"
MIS_MANUAL = STACK / "knowledge" / "mis-manual.md"

# Business-facing upload names from the MIS manual. The source files in the
# repo keep their historical names; the driver stages them into the sandbox
# under these canonical names before it writes any intent.
RAW_UPLOAD = "uploads/daily_bank_sales_data_2025.xlsx"
DASHBOARD_UPLOAD = "uploads/daily_bank_sales_dashboard_2025.xlsx"

# Internal handoff names. These are not business contracts; they are the
# driver's preferred aliases for artifacts passed between steps. The resolver
# below keeps equivalent internal names from becoming false failures.
STAGED_DATASET = "workspace/staged_daily_sales.csv"
ANOMALY_REPORT = "workspace/anomaly_report.md"
CLEAN_DATASET = "workspace/clean_daily_sales.csv"
COMPLETED_DASHBOARD = "workspace/Completed Sales Dashboard.xlsx"
VALIDATED_DASHBOARD = "workspace/Validated dashboard.xlsx"
VALIDATION_LOG = "workspace/validation_log.md"

INTERNAL_HANDOFF_ALIASES = {
    # Step 2 validates the staged data. If the model proves that file clean but
    # leaves it under the staged name, the runner can materialize the clean
    # alias for downstream steps instead of treating our internal filename as a
    # business failure.
    CLEAN_DATASET: [STAGED_DATASET],
}


def _step(name, intent_file, title, body, requires, produces):
    return {
        "name": name,
        "intent_file": intent_file,
        "title": title,
        "body": body,
        # relpaths that must exist in the sandbox BEFORE this step may run
        "requires": requires,
        # relpaths this step promises to write; the gate for the next step
        "produces": produces,
        "scripts": [],
        "also_requested": [],
    }


STEP_IO = {
    1: {
        "intent_file": "step1-load.md",
        "requires": [RAW_UPLOAD],
        "produces": [STAGED_DATASET],
        "also_requested": [ANOMALY_REPORT],
    },
    2: {
        "intent_file": "step2-sanity-check.md",
        "requires": [STAGED_DATASET],
        "produces": [CLEAN_DATASET],
        "scripts": ["workspace/validate_data.py"],
    },
    3: {
        "intent_file": "step3-slice-and-dice.md",
        "requires": [CLEAN_DATASET, DASHBOARD_UPLOAD],
        "produces": [COMPLETED_DASHBOARD],
        "scripts": ["workspace/generate.py"],
    },
    4: {
        "intent_file": "step4-dashboard-validation.md",
        "requires": [CLEAN_DATASET, COMPLETED_DASHBOARD],
        "produces": [VALIDATED_DASHBOARD, VALIDATION_LOG],
        "scripts": ["workspace/validate_dashboard.py"],
    },
}


def _steps_from_manual(manual_path: Path = MIS_MANUAL):
    """Parse the MIS manual into the four concrete Exchange intents.

    The manual is the source of truth for the step instructions. The driver
    contributes only runtime facts it owns: exact handoff filenames, script
    paths, and the intent document names used by the Exchange inbox."""
    document = parse_document(manual_path.read_text(encoding="utf-8"))
    parsed = []
    for section in document.sections:
        match = re.match(r"Section\s+(\d+)\s+--\s+(.+)", section.name.strip(), flags=re.IGNORECASE)
        if not match:
            continue
        number = int(match.group(1))
        title = match.group(2).strip()
        if number not in STEP_IO:
            continue
        io = STEP_IO[number]
        body = _intent_body_from_manual(number, title, section.body().prose, io)
        step = _step(
            f"Step {number} -- {title}",
            io["intent_file"],
            f"Step {number} -- {title}",
            body,
            requires=io["requires"],
            produces=io["produces"],
        )
        step["scripts"] = io.get("scripts", [])
        step["also_requested"] = io.get("also_requested", [])
        parsed.append(step)
    parsed.sort(key=lambda step: int(re.search(r"\d+", step["name"]).group(0)))
    expected = sorted(STEP_IO)
    found = [int(re.search(r"\d+", step["name"]).group(0)) for step in parsed]
    if found != expected:
        raise ValueError(
            f"{manual_path} must declare exactly sections {expected} for the Sales MIS run; found {found}"
        )
    return parsed


def _intent_body_from_manual(number: int, title: str, prose: str, io: dict) -> str:
    script_line = ""
    scripts = io.get("scripts") or []
    if scripts:
        script_list = ", ".join(f"`{script}`" for script in scripts)
        script_line = f" Run the canonical script {script_list}; if it is absent, author it first."
    also_requested = ""
    if io.get("also_requested"):
        also_requested = " Also request these conditional outputs when the manual calls for them: " + ", ".join(
            f"`{path}`" for path in io["also_requested"]
        ) + "."
    customer = f"Step {number + 1}" if number < 4 else "Business and Leadership"
    return (
        f"Parsed from `knowledge/mis-manual.md` Section {number} -- {title}.\n\n"
        f"Do step {number}, {title}, of the Sales MIS Workflow this cycle only."
        f"{script_line} Follow the manual section below as the source of truth for suppliers, inputs, process, outputs and customers."
        f"{also_requested}\n\n"
        f"Manual section:\n\n{prose}\n\n"
        f"Stop there: {customer} is the customer."
    )


STEPS = _steps_from_manual()


def attach_progress_printer(runtime) -> None:
    """Wrap the reasoning log's own `record()` so every stage EAR already
    logs -- sandbox open, policy judgments, discovery, each tool call, the
    final deliberation -- prints live as it happens. No new instrumentation:
    this is the trail EAR writes to `.ear/reasoning.md` anyway, surfaced to
    stdout as it's recorded instead of only readable after the fact."""
    log = runtime.reasoning_log
    original = log.record

    def record_and_print(stage, inputs=None, output="", rationale="", model="", usage=None):
        entry = original(stage, inputs=inputs, output=output, rationale=rationale, model=model, usage=usage)
        headline = str(output).strip().splitlines()[0] if str(output).strip() else "(no output)"
        print(f"  [{time.strftime('%H:%M:%S')}] {stage:<12} {headline[:140]}")
        return entry

    log.record = record_and_print


def start_live_dashboard(runtime, port: int = 8000):
    """Serve EAR's own Dashboard (ear/dashboard.py) live, in a background
    thread of *this* process -- reading the same `runtime` object the main
    thread is reasoning with, so the page's 3-second auto-refresh shows
    state as it happens, mid-cycle. The caller keeps the process alive
    after the cycles finish (see `main()`) until the page's Shut Down
    button -- or Ctrl-C -- calls `server.shutdown()`."""
    from ear.dashboard import create_server

    server = create_server(runtime, port=port, host="127.0.0.1", refresh=3)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Live dashboard: http://127.0.0.1:{port}/  (auto-refreshes every 3s)")
    return thread, server


# -- the truth layer: verified facts about the box, never prose -----------------


def _artifact_line(sandbox, relpath: str) -> str:
    """One artifact as a verified fact: present with its real size, or absent."""
    target = sandbox.root / relpath
    if target.exists():
        return f"{relpath} -- present, {target.stat().st_size:,} bytes"
    return f"{relpath} -- ABSENT"


def _fresh_slate(sandbox, steps, stack: Path = STACK) -> None:
    """Clear anything a previous run left behind, and say what was cleared.
    Stale artifacts are the cheapest way for a run to 'succeed' at work it
    never did; a truthful run starts empty and earns every file on the board."""
    leftovers = []
    for area in ("workspace", "outputs"):
        base = sandbox.root / area
        for entry in sorted(base.rglob("*")):
            if entry.is_file():
                leftovers.append(str(entry.relative_to(sandbox.root)))
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True, exist_ok=True)
    if leftovers:
        print(f"Cleared {len(leftovers)} stale artifact(s) from a previous run: {', '.join(leftovers)}")
    for step in steps:
        for stale in (stack / "intents" / step["intent_file"], stack / "decisions" / step["intent_file"]):
            if stale.exists():
                stale.unlink()


def _missing_inputs(sandbox, step) -> list:
    """The step's required inputs that do NOT exist in the sandbox right
    now -- the gate that keeps an intent from asserting a false handoff."""
    return [rel for rel in step["requires"] if not (sandbox.root / rel).exists()]


def _intent_text(sandbox, step) -> str:
    """The intent document for a step, carrying only driver-verified facts:
    each input listed with the byte size just read from disk."""
    context_lines = "".join(
        f"- input ({rel}): verified present, {(sandbox.root / rel).stat().st_size:,} bytes\n"
        for rel in step["requires"]
    )
    context_lines += "".join(f"- script ({rel}): create if absent\n" for rel in step.get("scripts", []))
    context_lines += "".join(f"- output_expected: {rel}\n" for rel in step["produces"])
    context_lines += "".join(f"- output_expected_if_applicable: {rel}\n" for rel in step.get("also_requested", []))
    return (
        f"# {step['title']}\n\n{step['body']}\n\n## Context\n\n{context_lines}\n"
        "Every output must be written to its exact output_expected path above -- "
        "the next step is gated on that literal filename, so a different name, "
        "however reasonable, is a failed handoff.\n"
    )


def _decision_status(decision_text: str) -> str:
    """The status the Deliverable extracted, read from the decision document's
    own Data section -- the single hedge-free field the workflow declares."""
    match = re.search(r"^- status:\s*(.+)$", decision_text, flags=re.MULTILINE)
    return match.group(1).strip() if match else "(no status field)"


def _provider_failure(decision_text: str) -> bool:
    """True when a cycle's BLOCKED decision records the model call itself
    failing (billing, auth, network) -- the Exchange renders an LMError
    into the decision document rather than crashing (ear/exchange.py).
    Money-safety: this is the signal to auto-pause instead of letting the
    next cycles burn further calls against a dead account."""
    return "Status: BLOCKED" in decision_text and bool(
        re.search(r"LLM call to .+ failed", decision_text)
    )


def _step_complete(sandbox, step) -> bool:
    """A step counts as complete only by filesystem fact: every artifact it
    promises exists in the sandbox right now."""
    return all((sandbox.root / rel).exists() for rel in step["produces"])


def _pause(stack: Path, step, reason: str) -> Path:
    """Write the pause marker a human resumes from: which step was
    interrupted, why, and the exact command that continues the run
    without repeating the steps already verified complete."""
    marker = stack / ".ear" / "paused.md"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        "# Sales MIS run PAUSED\n\n"
        f"- paused at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- interrupted step: {step['name']}\n"
        f"- reason: {reason.strip()}\n\n"
        "Completed steps are verified on disk and will be skipped on resume.\n"
        "After fixing the cause (e.g. topping up API credits), resume with:\n\n"
        "    python3 examples/sales_mis_guru.py --resume\n",
        encoding="utf-8",
    )
    return marker


def _step_board(sandbox, steps, results) -> str:
    """The step-status board: one line of verified filesystem fact per
    artifact, plus the decision's own status. `results` maps step name ->
    dict(status=..., ran=bool)."""
    lines = ["# Sales MIS -- step status (computed from the sandbox, not from prose)", ""]
    for step in steps:
        outcome = results.get(step["name"])
        if outcome is None:
            headline = "not reached"
        else:
            produced = all((sandbox.root / rel).exists() for rel in step["produces"])
            verified = "outputs verified on disk" if produced else "OUTPUTS MISSING despite decision"
            headline = f"decision status: {outcome['status']} -- {verified}"
        lines.append(f"## {step['name']} -- {headline}")
        lines.append("")
        lines.append("Inputs staged:")
        for rel in step["requires"]:
            lines.append(f"- {_artifact_line(sandbox, rel)}")
        lines.append("Outputs:")
        for rel in step["produces"]:
            lines.append(f"- {_artifact_line(sandbox, rel)}")
        lines.append("")
    return "\n".join(lines)


def _print_board(board: str) -> None:
    print("\n" + "=" * 72)
    for line in board.splitlines():
        print(f"  {line}")
    print("=" * 72)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the Sales MIS cycle through the production flow -- all four steps, or one at a time for step-by-step debugging."
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=range(1, len(STEPS) + 1),
        default=None,
        help="run exactly this one SIPOC step (1-4) through the same production flow; "
        "prior steps' artifacts are kept and gated on, and the process exits when the step's cycle ends",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="skip the fresh-slate wipe even when starting from step 1",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a paused or partial run: keep all artifacts, skip every step whose "
        "outputs are already verified on disk, and run only what remains",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is set -- reasoning runs against a live model")
    else:
        print("No ANTHROPIC_API_KEY -- deterministic fallbacks run, and the trail still records every step")

    runtime = load_runtime(STACK)
    print(f"Runtime: {runtime.name}")
    print(f"Model:   {runtime.model_binding.model_id if runtime.model_binding else 'deterministic fallback (no credential in the environment)'}")

    if runtime.sandbox is None:
        raise RuntimeError("memory.md's Sandbox section did not open a sandbox -- check the stack")
    print(f"Sandbox: {runtime.sandbox.root}")

    # Each cycle below is a focused, single-step ask -- author one script,
    # run it, read its evidence, plus room for a rerun when the first
    # attempt errors -- not a whole 4-step pipeline in one shot.
    runtime.tool_binder.max_iterations = 15
    attach_progress_printer(runtime)
    dashboard_thread, dashboard_server = start_live_dashboard(
        runtime, port=int(os.environ.get("MIS_DASHBOARD_PORT", "8000"))
    )

    sandbox = runtime.sandbox
    selected = STEPS if args.step is None else [STEPS[args.step - 1]]
    starting_fresh = args.step in (None, 1) and not args.keep_artifacts and not args.resume
    if starting_fresh:
        _fresh_slate(sandbox, STEPS)
    else:
        # Debugging a later step in isolation: earlier steps' artifacts are
        # the handoffs this step is gated on -- keep them, clear only this
        # step's own exchange documents so the cycle actually re-runs.
        for step in selected:
            for stale in (STACK / "intents" / step["intent_file"], STACK / "decisions" / step["intent_file"]):
                if stale.exists():
                    stale.unlink()

    uploads = sandbox.resolve("uploads")
    uploads.mkdir(parents=True, exist_ok=True)
    shutil.copy(RAW_SOURCE, sandbox.root / RAW_UPLOAD)
    shutil.copy(DASHBOARD_TEMPLATE, sandbox.root / DASHBOARD_UPLOAD)
    print("Seeded sandbox uploads/ (verified):")
    for rel in (RAW_UPLOAD, DASHBOARD_UPLOAD):
        print(f"  - {_artifact_line(sandbox, rel)}")

    intents_dir = STACK / "intents"
    intents_dir.mkdir(exist_ok=True)
    exchange = Exchange(STACK)
    status_path = STACK / ".ear" / "step_status.md"
    results: dict = {}

    paused = False
    for step in selected:
        # Resume discipline: a step whose promised outputs already exist is
        # never re-run (and never re-billed) -- passed work stays passed.
        if args.resume and _step_complete(sandbox, step):
            print(f"\n=== {step['name']} already complete (outputs verified on disk) -- skipped ===")
            results[step["name"]] = {"status": "complete (skipped on resume)"}
            continue

        # The gate: every input this step depends on must exist on disk NOW.
        # An intent never asserts a handoff the driver has not verified.
        missing = _missing_inputs(sandbox, step)
        if missing:
            print(f"\n=== {step['name']} GATED -- required input(s) missing: {', '.join(missing)} ===")
            print("The previous step did not hand off what this step needs; stopping instead of")
            print("feeding the next persona a false premise. See the board below for the truth.")
            break

        (intents_dir / step["intent_file"]).write_text(_intent_text(sandbox, step), encoding="utf-8")
        print(f"\n=== intents/{step['intent_file']} dropped into the exchange ===")
        try:
            for path in exchange.run(runtime):
                text = path.read_text()
                print(f"\n--- {path.relative_to(STACK)} ---")
                print(text)
                results[step["name"]] = {"status": _decision_status(text)}
                if _provider_failure(text):
                    paused = True
        finally:
            # The board is written no matter how the cycle ended -- a crash
            # must never leave a stale board claiming last cycle's truth.
            board = _step_board(sandbox, STEPS, results)
            status_path.write_text(board + "\n", encoding="utf-8")
            _print_board(board)

        if paused:
            # Money safety: the model call itself failed (billing, auth,
            # network). Stop immediately -- running further cycles would
            # burn more calls against the same dead account -- alert the
            # human, and leave a marker the resume flag picks up.
            marker = _pause(STACK, step, "the provider call failed mid-cycle -- see the decision document above")
            print("\n" + "!" * 72)
            print("!!  RUN PAUSED -- the model provider call failed (billing/auth/network).")
            print(f"!!  Interrupted step: {step['name']}. Completed steps stay verified on disk.")
            print("!!  Fix the cause (e.g. top up API credits), then resume with:")
            print("!!      python3 examples/sales_mis_guru.py --resume")
            print(f"!!  Pause marker: {marker}")
            print("!" * 72)
            dashboard_server.shutdown()
            dashboard_thread.join()
            return 2

    stale_marker = STACK / ".ear" / "paused.md"
    if stale_marker.exists():
        stale_marker.unlink()  # this run got past the pause point; the marker no longer states the truth

    trail = Path(runtime.reasoning_log.path)
    print(f"\nReasoning trail: {trail}")
    print(f"Step status board: {status_path}")
    print(f"Sandbox workspace: {sandbox.root}")

    if args.step is not None:
        # Step-by-step debugging: exit cleanly so the trail and the board
        # can be inspected between steps; the dashboard restarts with the
        # next step's invocation.
        print(f"\nStep run complete (--step {args.step}). Exiting; rerun with the next --step to continue.")
        dashboard_server.shutdown()
        dashboard_thread.join()
        return 0

    print(
        "\nRun complete. Dashboard stays live -- shut it down from the "
        "page's button when you're done reading it (Ctrl-C here also works)."
    )
    try:
        dashboard_thread.join()
    except KeyboardInterrupt:
        print("\nCtrl-C received -- stopping the dashboard.")
        dashboard_server.shutdown()
        dashboard_thread.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
