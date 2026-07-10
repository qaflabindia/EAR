"""conftest -- opt-in live-model testing.

A local `.env` credential (ANTHROPIC_API_KEY) is loaded into the test
process ONLY when `EAR_LIVE_TESTS=1` is set in the environment. Default
`pytest` therefore makes zero API calls and spends zero money -- every
live-LLM test skips. This is deliberate: an earlier version loaded `.env`
unconditionally, and every "offline" full-suite run silently billed real
model calls.

To run the live tests, explicitly:

    EAR_LIVE_TESTS=1 python3 -m pytest tests/ -q

Re-run only what failed last time (pytest's own last-failed cache):

    EAR_LIVE_TESTS=1 python3 -m pytest tests/ -q --lf

Test-only: this file is never part of the shipped `ear` package
(pyproject.toml's package discovery only includes `ear*`). An already-set
environment variable is never overwritten, so a real deployment's
credential always wins over this file's."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent / ".env"


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


if os.environ.get("EAR_LIVE_TESTS") == "1":
    _load_env(_ENV_FILE)
