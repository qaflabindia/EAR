#!/usr/bin/env python3
"""Sales MIS Guru -- markdown in, markdown out, EAR reasons through the cycle.

This is wiring, not logic: point `load_runtime` at `sales_mis_stack/`, seed its
sandbox with the raw source workbook and the dashboard template, drop in one
intent per persona, and run the Exchange. Every step -- loading, sanity-
checking, slicing and dicing, reconciling -- is code the model itself authors
and runs inside the sandbox (see `sales_mis_stack/memory.md`'s Sandbox section
and `skills.md`); nothing in this script does that work.

Why three intents instead of one: EAR composes an entire workflow -- however
many steps it narrates -- into a *single* deliberation call (see
`ear/performer.py`: `Deliberator.deliberate` runs once per intent). That one
call's native tool loop is bounded by `ToolBinder.max_iterations` (default 6,
sized for a quick lookup). Asking for all four steps of a real load/validate/
generate/reconcile cycle in one shot starves that budget before the model can
act on its own (correct) plan. Splitting into one focused intent per persona
-- load+sanity-check, then slice-and-dice, then validate -- gives each a
sane, justified budget instead of an arbitrarily large one, and it means
progress lands as three separate decision documents instead of one silent
block.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root, so `import ear` finds the local package

from ear import Exchange, load_runtime  # noqa: E402

STACK = HERE / "sales_mis_stack"
SOURCE_DIR = STACK / "Sales MIS"
RAW_SOURCE = SOURCE_DIR / "bank_daily_sales_2025.xlsx"
DASHBOARD_TEMPLATE = SOURCE_DIR / "bank_daily_sales_dashboard_2025.xlsx"

# The only "orchestration contract" this driver imposes: where each cycle
# hands its output to the next. What each script *does* with these paths is
# entirely the model's call -- these are handoff points, not logic.
STAGED_DATASET = "workspace/staged_daily_sales.csv"
ANOMALY_REPORT = "workspace/anomaly_report.md"
COMPLETED_DASHBOARD = "outputs/bank_daily_sales_dashboard_2025_completed.xlsx"
VALIDATION_LOG = "outputs/validation_log.md"

CYCLES = [
    (
        "load-and-sanity-check.md",
        "# Load and sanity-check the raw FY2025 daily sales data\n\n"
        "Do steps 1 and 2 of the Sales MIS Workflow this cycle only: load the "
        "raw workbook and run the sanity check it describes. Stop once the "
        "clean staged dataset and the anomaly report are written -- the "
        "dashboard is a later cycle's job.\n\n"
        "## Context\n\n"
        f"- raw_source_workbook: uploads/{RAW_SOURCE.name}\n"
        f"- staged_dataset_output: {STAGED_DATASET}\n"
        f"- anomaly_report_output: {ANOMALY_REPORT}\n",
    ),
    (
        "slice-and-dice.md",
        "# Slice and dice the clean data into the FY2025 dashboard\n\n"
        "Do step 3 of the Sales MIS Workflow this cycle only: the raw load "
        "and sanity check already ran in a prior cycle and the staged "
        "dataset below is the clean result. Read it and fill the dashboard "
        "template, saving the completed workbook. Validation is a later "
        "cycle's job.\n\n"
        "## Context\n\n"
        f"- staged_dataset: {STAGED_DATASET}\n"
        f"- dashboard_template: uploads/{DASHBOARD_TEMPLATE.name}\n"
        f"- completed_dashboard_output: {COMPLETED_DASHBOARD}\n",
    ),
    (
        "validate-dashboard.md",
        "# Validate the completed dashboard and write the delivery note\n\n"
        "Do step 4 of the Sales MIS Workflow this cycle only: the dashboard "
        "below was already generated in a prior cycle. Reconcile it against "
        "the staged clean dataset, sheet by sheet, write the validation log, "
        "and draft the delivery note for Business and Leadership.\n\n"
        "## Context\n\n"
        f"- staged_dataset: {STAGED_DATASET}\n"
        f"- completed_dashboard: {COMPLETED_DASHBOARD}\n"
        f"- validation_log_output: {VALIDATION_LOG}\n",
    ),
]


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
    thread is reasoning with. Every reasoning-log record lands in
    `runtime.reasoning_log.records` the instant it's recorded (well before
    the end-of-cycle `flush()` to disk), so a page load or the page's own
    3-second auto-refresh shows state as it happens, mid-cycle, not just
    after a cycle finishes.

    Returns `(thread, server)`. Deliberately *not* a daemon thread pointed
    at the old blocking `serve()`: a daemon thread is killed the instant
    the process exits, which is exactly the bug this fixes -- the
    dashboard used to die the moment `main()`'s cycles finished, even
    though that is precisely when a human wants to sit and read it. The
    caller is expected to keep the process alive after its own work is
    done (see `main()`) until the page's own Shut Down button -- or
    Ctrl-C -- calls `server.shutdown()`."""
    from ear.dashboard import create_server

    server = create_server(runtime, port=port, host="127.0.0.1", refresh=3)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Live dashboard: http://127.0.0.1:{port}/  (auto-refreshes every 3s)")
    return thread, server


def main() -> int:
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

    # Each cycle below is a focused, single-persona ask -- one script plus
    # room for a rerun if the first attempt errors -- not a whole 4-step
    # pipeline in one shot, so a modest, justified budget replaces a
    # blind 40.
    runtime.tool_binder.max_iterations = 15
    attach_progress_printer(runtime)
    dashboard_thread, dashboard_server = start_live_dashboard(
        runtime, port=int(os.environ.get("MIS_DASHBOARD_PORT", "8000"))
    )

    uploads = runtime.sandbox.resolve("uploads")
    uploads.mkdir(parents=True, exist_ok=True)
    shutil.copy(RAW_SOURCE, uploads / RAW_SOURCE.name)
    shutil.copy(DASHBOARD_TEMPLATE, uploads / DASHBOARD_TEMPLATE.name)
    print(f"Seeded sandbox uploads/: {RAW_SOURCE.name}, {DASHBOARD_TEMPLATE.name}")

    intents_dir = STACK / "intents"
    intents_dir.mkdir(exist_ok=True)
    exchange = Exchange(STACK)

    for filename, text in CYCLES:
        (intents_dir / filename).write_text(text, encoding="utf-8")
        print(f"\n=== intents/{filename} dropped into the exchange ===")
        for path in exchange.run(runtime):
            print(f"\n--- {path.relative_to(STACK)} ---")
            print(path.read_text())

    trail = Path(runtime.reasoning_log.path)
    print(f"\nReasoning trail: {trail}")
    print(f"Sandbox workspace: {runtime.sandbox.root}")

    print(
        "\nAll cycles complete. Dashboard stays live -- shut it down from the "
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
