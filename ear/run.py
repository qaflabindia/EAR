"""run -- the in-pod entrypoint: run one EAR cycle from a stack and an
intent handed in through the environment, so a Kubernetes Job (see
`ear/k8s.py`) can execute a runtime instance inside a container.

    python -m ear.run /path/to/stack

Reads the intent from the environment -- `EAR_INTENT` (the text, required)
and `EAR_CONTEXT` (a JSON object of context values, optional) -- loads the
stack, runs one governed cycle, writes the decision to
`EAR_DECISION_PATH` (default `<stack>/decision.md`) and flushes the trail.
The exit code reflects the cycle's outcome, so the Job's success or failure
is the decision's: 0 when a decision was reached, 2 when governance blocked
it (a refusal is a valid, non-crash outcome), 1 on an unexpected error.

Set `EAR_LOG_LEVEL` (DEBUG/INFO/WARNING/...) to see live progress on
stderr as the cycle runs -- every LM call and sandboxed command, as they
happen, not just the final decision. Unset means silent, exactly as
before this existed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional


def main(argv: Optional[list] = None) -> int:
    level = os.environ.get("EAR_LOG_LEVEL")
    if level:
        logging.basicConfig(level=level.upper(), format="%(asctime)s %(name)s %(levelname)s %(message)s")

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m ear.run <stack-dir>", file=sys.stderr)
        return 2
    stack = argv[0]
    text = os.environ.get("EAR_INTENT")
    if not text:
        print("EAR_INTENT is required (the intent text)", file=sys.stderr)
        return 2
    context: dict = {}
    raw = os.environ.get("EAR_CONTEXT")
    if raw:
        try:
            loaded = json.loads(raw)
        except ValueError:
            print("EAR_CONTEXT must be a JSON object", file=sys.stderr)
            return 2
        if isinstance(loaded, dict):
            context = loaded

    from .intent import Intent
    from .loader import load_runtime

    runtime = load_runtime(stack)
    try:
        decision = runtime.reason(Intent(text=text, context=context))
    except PermissionError as blocked:
        print(f"BLOCKED: {blocked}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 -- the pod reports the failure and exits non-zero
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    destination = os.environ.get("EAR_DECISION_PATH") or str(Path(stack) / "decision.md")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(str(decision), encoding="utf-8")
    print(str(decision)[:1000])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
