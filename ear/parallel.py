"""Parallel -- native, dependency-free parallel map and map-reduce.

EAR's workload is I/O-bound: a cycle spends almost all its time waiting on a
model call over the stdlib HTTPS client, which releases the GIL while it
waits. So *threads* give real overlap, and this module is EAR's own
`joblib`-shaped engine over `concurrent.futures.ThreadPoolExecutor` -- no
third-party dependency, the standard library is the whole of it.

Two primitives:

* `parallel_map(fn, items)` -- run `fn` over `items` concurrently and return
  one `Result` per item **in input order**, regardless of completion order.
  A unit that raises does not fail the batch: its error is captured on its
  `Result`, so the map always returns a full, ordered set of outcomes (the
  same per-unit isolation the Kernel already gives per task).

* `map_reduce(items, map_fn, reduce_fn)` -- scatter `map_fn` over `items` in
  parallel, then fold the successful values with `reduce_fn`. The reduce is
  where parallel branches join, and in EAR it is usually a *judgment*:
  `JudgedReducer` synthesizes N partial results into one with the model
  (`SynthesizeParallel`), and falls back to a deterministic reduce offline --
  code maps in parallel, the model reduces, on the record.

Three properties make this EAR-shaped rather than a bare pool:

1. **Deterministic order.** Results come back indexed by input position, so a
   caller (and the audit trail it writes) sees the same order every run even
   though threads finish out of order.
2. **A backend seam.** `backend="serial"` runs in-process and in order (the
   offline / deterministic-debug floor and the zero-work fast path);
   `"threads"` uses a bounded pool. The seam is where an out-of-process
   backend (the existing Kubernetes dispatcher) attaches later, the same way
   `joblib` swaps threads for processes.
3. **Bounded, nest-safe.** Every unit is capped by `workers`, and nesting is
   depth-guarded: a `map_reduce` whose units themselves `map_reduce` runs the
   inner level serially past `max_depth`, so a bounded thread budget can
   never deadlock or explode.

The single-writer rule still holds: a mapped unit must be **independent** --
it reads the immutable stack and writes only its own result, never shared
runtime state. The reduce is the only place results fold back.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

SERIAL = "serial"
THREADS = "threads"

DEFAULT_WORKERS = 8
DEFAULT_MAX_DEPTH = 2

# Nesting depth of the current thread's parallel work. A unit that runs its
# own parallel_map increments it; past DEFAULT_MAX_DEPTH the inner level runs
# serially, so a bounded pool can never be starved by its own children.
_depth = threading.local()


def _current_depth() -> int:
    return getattr(_depth, "value", 0)


@dataclass
class Result:
    """One unit's outcome: its value or the exception it raised, tagged with
    the input index so the batch stays in input order."""

    index: int
    value: Any = None
    error: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _run_one(fn: Callable[[Any], Any], index: int, item: Any) -> Result:
    """Run one unit, capturing an ordinary failure on the Result rather than
    letting it fail the whole batch. `Exception` only -- a cancellation
    (KeyboardInterrupt / SystemExit, both `BaseException`) still propagates
    and tears the batch down, the exercisable stop the kill switch needs."""
    try:
        return Result(index=index, value=fn(item))
    except Exception as error:  # noqa: BLE001 -- one unit's failure never fails the batch
        return Result(index=index, error=error)


def parallel_map(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    workers: int = DEFAULT_WORKERS,
    backend: str = THREADS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> list[Result]:
    """Map `fn` over `items` concurrently; return one `Result` each, in input
    order. Runs serially (deterministically) when the backend is `serial`,
    when there is nothing to gain (<=1 worker or item), or when already nested
    `max_depth` deep -- the safe floor against nested-pool thread blow-up."""
    materialized = list(items)
    if not materialized:
        return []
    depth = _current_depth()
    if backend == SERIAL or workers <= 1 or len(materialized) == 1 or depth >= max_depth:
        return [_run_one(fn, index, item) for index, item in enumerate(materialized)]

    results: list[Optional[Result]] = [None] * len(materialized)
    _depth.value = depth + 1
    try:
        with ThreadPoolExecutor(max_workers=min(workers, len(materialized))) as pool:
            for result in pool.map(
                lambda pair: _run_one(fn, pair[0], pair[1]),
                list(enumerate(materialized)),
            ):
                results[result.index] = result
    finally:
        _depth.value = depth
    # `pool.map` preserves input order, but index the results anyway so the
    # invariant holds no matter how the backend schedules.
    return [result for result in sorted((r for r in results if r is not None), key=lambda r: r.index)]


def values(results: list[Result], on_error: str = "collect") -> list[Any]:
    """The successful values from a `parallel_map`, in order. `on_error`:
    `"collect"` drops failures (their errors stay on the Results), `"raise"`
    re-raises the first failure -- the caller chooses partial-tolerance."""
    if on_error == "raise":
        for result in results:
            if not result.ok:
                raise result.error  # type: ignore[misc]
    return [result.value for result in results if result.ok]


def errors(results: list[Result]) -> list[BaseException]:
    return [result.error for result in results if not result.ok and result.error is not None]


def map_reduce(
    items: Iterable[Any],
    map_fn: Callable[[Any], Any],
    reduce_fn: Callable[[list[Any]], Any],
    *,
    workers: int = DEFAULT_WORKERS,
    backend: str = THREADS,
    on_error: str = "collect",
) -> Any:
    """Scatter `map_fn` over `items` in parallel, then fold the successful
    values with `reduce_fn`. `reduce_fn` may be deterministic code or a
    `JudgedReducer` (the model synthesizes the branches). With `on_error=
    "raise"` the first mapped failure aborts the reduce; by default failures
    are dropped and the reduce sees the partial set."""
    mapped = parallel_map(map_fn, items, workers=workers, backend=backend)
    return reduce_fn(values(mapped, on_error=on_error))


def _concatenate(parts: list[Any]) -> str:
    return "\n\n".join(str(part) for part in parts)


@dataclass
class JudgedReducer:
    """Fold parallel branch results into one answer by *judgment*: with a
    model bound, the model synthesizes the parts (`SynthesizeParallel`) --
    reconciling disagreement, not merely concatenating; offline, a
    deterministic `fallback` runs and the result says it was a fallback. The
    reduce, like every judgment in EAR, lands on the runtime's audit spine
    when a log is given."""

    task: str
    model_binding: Any = None
    runtime: Any = None
    fallback: Callable[[list[Any]], Any] = _concatenate

    def __call__(self, parts: list[Any]) -> Any:
        if not parts:
            return self.fallback(parts)
        binding = self.model_binding if self.model_binding is not None else getattr(self.runtime, "model_binding", None)
        lm = getattr(binding, "lm", None) if binding is not None else None
        if binding is not None:
            binding.activate()
            lm = getattr(binding, "lm", None)
        if lm is None:
            return self.fallback(parts)

        from .reasoning_log import model_name
        from .signatures import SynthesizeParallel

        rendered = "\n".join(f"- part {index + 1}: {part}" for index, part in enumerate(parts))
        result = SynthesizeParallel.run(lm, task=self.task, parts=rendered)
        synthesis = str(getattr(result, "synthesis", "") or "").strip() or self.fallback(parts)
        log = getattr(self.runtime, "reasoning_log", None)
        if log is not None:
            log.record(
                stage="reduce",
                inputs={"task": self.task, "parts": parts},
                output=synthesis,
                rationale=f"synthesized {len(parts)} parallel parts",
                model=model_name(binding),
            )
        return synthesis
