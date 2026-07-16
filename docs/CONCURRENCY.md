# EAR — Concurrency & Parallelism

**Status:** foundation shipped (`ear/parallel.py`, Kernel fleet pool) ·
the decided model, its rationale, and what is built vs. planned.

## The three facts that decide the model

1. **The workload is I/O-bound.** A cycle spends almost all its wall-clock
   waiting on a model call over the stdlib HTTPS client (`ear/llm.py`,
   `timeout=120`). That call **releases the GIL**, so *threads* give real
   overlap — this is not CPU-bound work that needs processes.
2. **A Runtime owns mutable state.** `memory`, `experience`, `adaptations`,
   `session_store`, the **hash-chained `reasoning_log`**, and the AECC
   `envelope_registry`. The audit spine is a *linear* hash chain — one writer
   per instance is the invariant that makes it tamper-evident, not a
   limitation to engineer around.
3. **Scale-out already exists out-of-process** — the Kernel's `dispatcher`
   seam runs a cycle on Kubernetes (`ear/k8s.py`), one pod per firing.

## The decision: single-writer actors, thread parallelism, one ordered spine

- **A Runtime is an actor.** It processes exactly one cycle at a time, so its
  mutable state needs **no locks**. Concurrency comes from running *many*
  instances, and from *independent* fan-out inside a cycle — never from two
  cycles re-entering one instance.
- **Map-reduce is the engine, not a shared runtime.** `joblib`/map-reduce
  needs *independent units*, which is the actor invariant at a finer grain: a
  mapped unit reads the immutable stack and writes only its own result; the
  **reduce** is the one place results fold back. So we get parallel map-reduce
  *and* keep single-writer instances — no contention, no locks on domain
  state.
- **Threads, via the standard library.** `concurrent.futures.ThreadPoolExecutor`.
  Async is rejected (it would recolor every synchronous function and the HTTP
  client — a rewrite for no gain on thread-overlappable I/O); in-process
  multiprocessing is rejected (it breaks shared state and needs pickling — the
  k8s dispatcher already covers process isolation better).
- **One ordered audit spine.** Parallelism changes wall-clock order, never
  *logical* order: results and their trail records come back **indexed by
  input position**, so the hash chain stays linear and every run's trail is
  reproducible.

## What is built

### `ear/parallel.py` — the native `joblib`-shaped primitive

```python
parallel_map(fn, items, *, workers=8, backend="threads") -> list[Result]
map_reduce(items, map_fn, reduce_fn, *, workers=8) -> reduced
```

- **Deterministic order** — `Result`s come back in input order regardless of
  completion order.
- **Per-unit isolation** — a unit that raises captures its error on its
  `Result`; the batch always returns a full, ordered set (`values(...,
  on_error="collect"|"raise")` chooses partial-tolerance). Only a
  cancellation (`BaseException`) tears the batch down — the exercisable stop a
  kill switch needs.
- **Backend seam** — `"serial"` (offline / deterministic-debug / zero-work
  fast path) and `"threads"` (bounded pool). This is where an out-of-process
  backend (the k8s dispatcher) attaches later, the way `joblib` swaps threads
  for processes.
- **Nest-safe** — nesting is depth-guarded (`max_depth`), so a `map_reduce`
  whose units themselves `map_reduce` runs the inner level serially past the
  limit; a bounded thread budget can never deadlock or explode.
- **Reduce by judgment** — `JudgedReducer` synthesizes N parallel parts into
  one with the model (`SynthesizeParallel`), and falls back to a deterministic
  reduce offline, announced as a fallback. *Code maps in parallel; the model
  reduces; it lands on the spine.*

### Kernel fleet pool — `ear/kernel.py`

`Kernel(max_workers=N)`. With `N > 1` the loop runs work for **different**
instances concurrently on a bounded pool, while holding **at most one
in-flight cycle per instance** (`_take_ready_free` reserves an instance
atomically). So:

- different instances overlap (real throughput on I/O-bound cycles);
- one instance's cycles are serialized (single-writer preserved);
- `max_workers=1` (the default) is the historical **serial** scheduler,
  byte-for-byte unchanged — `tick`/`drain` are untouched, and
  `drain_concurrent` falls back to `drain`.

## What is planned (sequenced)

1. **Examiner + Optimizer** — swap their sequential loops to `parallel_map`
   (read-only, separate contexts: safe, immediate wins).
2. **Intra-cycle map-reduce** — independent workflow branches / data
   partitions run in parallel inside `reason()`, records **buffered and
   flushed in input order** onto the one spine, reduced by judgment.
3. **Parallel ensemble** — map K voices on one question, reduce by synthesis
   (self-consistency), as a parallel variant of the sequential `Panel`.

## The hazards, and how they are handled

| Hazard | Handling |
|---|---|
| Nested-pool deadlock / thread blow-up | depth-guarded nesting (`max_depth`) runs inner levels serially |
| Cost blow-up (parallel token spend) | bounded `workers`; the usage ledger prices a batch before it scales (planned budget check pre-fan-out) |
| Cancellation / kill switch | a unit's ordinary failure is isolated; `BaseException` propagates and tears the batch down; LLM calls are time-boxed at 120s |
| Ungoverned fan-out | `Governor.govern` runs **before** any map; each `reason()`-based unit re-enters the gate on its own instance |
| Non-deterministic trail | results and records are indexed by input position — logical order is stable even as wall-clock order varies |

## The one rule, in this plane

**The model judges; code enforces, records — and here, parallelizes.** The
parallel map, the ordering, the budgets, and the per-instance serialization
are mechanics in code; the reduce, when it reasons, is the model's, on the
record; and offline every path degrades to a deterministic, ordered fallback
that says so.
