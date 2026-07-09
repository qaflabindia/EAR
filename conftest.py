"""conftest -- load a local .env into the environment before tests run,
so a credential like ANTHROPIC_API_KEY reaches the test process without
a third-party dotenv dependency. Test-only: this file is never part of
the shipped `ear` package (pyproject.toml's package discovery only
includes `ear*`) and never installed -- pytest collects it from the repo
root by its own convention. An already-set environment variable is never
overwritten, so a real deployment's credential always wins over this
file's."""

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


_load_env(_ENV_FILE)
