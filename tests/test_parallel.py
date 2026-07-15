"""Tests for EAR's native parallelism -- `ear/parallel.py` and the Kernel
fleet pool (`ear/kernel.py`).

All offline and deterministic: the parallel primitive is exercised with pure
functions and short sleeps, and the map-reduce reducer runs its deterministic
fallback (no model bound). Concurrency is asserted by observed overlap, not by
timing luck -- a tracker records how many units are active at once.
"""

from __future__ import annotations

import threading
import time

import pytest

from ear import JudgedReducer, Result, Runtime, map_reduce, parallel_map, parallel_values
from ear.kernel import Kernel
from ear.parallel import SERIAL, THREADS, errors


# ---------------------------------------------------------------------------
# parallel_map
# ---------------------------------------------------------------------------


def test_results_come_back_in_input_order():
    # Later items finish first, but the order is by input position.
    def slow(x):
        time.sleep(0.02 * (5 - x))
        return x * 10

    out = parallel_map(slow, [0, 1, 2, 3, 4], workers=5)
    assert [r.value for r in out] == [0, 10, 20, 30, 40]


def test_units_actually_overlap():
    # Five 0.1s sleeps on five workers finish in ~0.1s, not ~0.5s.
    start = time.time()
    parallel_map(lambda _x: time.sleep(0.1), range(5), workers=5)
    assert time.time() - start < 0.3


def test_a_unit_failure_does_not_fail_the_batch():
    def maybe(x):
        if x == 2:
            raise ValueError("boom")
        return x

    out = parallel_map(maybe, [0, 1, 2, 3], workers=4)
    assert parallel_values(out) == [0, 1, 3]
    assert [str(e) for e in errors(out)] == ["boom"]
    assert not out[2].ok


def test_values_can_reraise():
    out = parallel_map(lambda x: 1 / x, [1, 0, 2], workers=3)
    with pytest.raises(ZeroDivisionError):
        parallel_values(out, on_error="raise")


def test_serial_backend_is_deterministic_and_ordered():
    out = parallel_map(lambda x: x + 1, [1, 2, 3], backend=SERIAL)
    assert [r.value for r in out] == [2, 3, 4]


def test_empty_input():
    assert parallel_map(lambda x: x, [], workers=4) == []


def test_nesting_is_depth_guarded_and_does_not_deadlock():
    # An outer map whose units each run their own map completes without
    # starving a bounded pool.
    def outer(_x):
        return sum(r.value for r in parallel_map(lambda y: y, range(3), workers=3))

    out = parallel_map(outer, range(4), workers=2)
    assert [r.value for r in out] == [3, 3, 3, 3]


def test_result_dataclass():
    r = Result(index=0, value=7)
    assert r.ok and r.value == 7
    assert not Result(index=1, error=ValueError("x")).ok


# ---------------------------------------------------------------------------
# map_reduce with a judged reducer (deterministic fallback offline)
# ---------------------------------------------------------------------------


def test_map_reduce_with_deterministic_fallback():
    reducer = JudgedReducer(task="sum of squares", fallback=lambda parts: sum(int(p) for p in parts))
    total = map_reduce([1, 2, 3, 4], lambda x: x * x, reducer, workers=4)
    assert total == 1 + 4 + 9 + 16


def test_map_reduce_drops_failures_by_default():
    reducer = JudgedReducer(task="sum", fallback=lambda parts: sum(parts))
    total = map_reduce([2, 0, 4], lambda x: 10 // x, reducer, workers=3)  # 10//0 fails, dropped
    assert total == 5 + 2


def test_judged_reducer_offline_uses_fallback_and_says_nothing_false():
    # No model bound -> deterministic fallback, no fabricated synthesis.
    reducer = JudgedReducer(task="join", fallback=lambda parts: "|".join(parts))
    assert reducer(["a", "b", "c"]) == "a|b|c"


# ---------------------------------------------------------------------------
# Kernel fleet pool
# ---------------------------------------------------------------------------


class _Tracker:
    def __init__(self):
        self.active = {}
        self.max_parallel = 0
        self.per_instance_overlap = 0
        self.lock = threading.Lock()

    def enter(self, name):
        with self.lock:
            self.active[name] = self.active.get(name, 0) + 1
            self.max_parallel = max(self.max_parallel, sum(self.active.values()))
            if self.active[name] > 1:
                self.per_instance_overlap += 1

    def exit(self, name):
        with self.lock:
            self.active[name] -= 1


class _FakeRuntime:
    def __init__(self, name, tracker, delay=0.15):
        self.name = name
        self.tracker = tracker
        self.delay = delay

    def reason(self, intent, approval=None, claim=None):
        self.tracker.enter(self.name)
        time.sleep(self.delay)
        self.tracker.exit(self.name)
        return f"{self.name} ok"


def test_different_instances_run_concurrently():
    tracker = _Tracker()
    kernel = Kernel(max_workers=4)
    for name in ("a", "b", "c", "d"):
        kernel.register(name, _FakeRuntime(name, tracker))
        kernel.submit(name, object())
    start = time.time()
    dispatched = kernel.drain_concurrent()
    elapsed = time.time() - start
    assert len(dispatched) == 4
    assert all(d.status == "ran" for d in dispatched)
    assert tracker.max_parallel == 4  # all four overlapped
    assert elapsed < 0.5  # ~0.15s wall, not 0.6s serial


def test_same_instance_is_serialized():
    # Three tasks on ONE instance must never overlap -- the single-writer
    # actor invariant that protects its memory and audit spine.
    tracker = _Tracker()
    kernel = Kernel(max_workers=4)
    kernel.register("solo", _FakeRuntime("solo", tracker))
    for _ in range(3):
        kernel.submit("solo", object())
    dispatched = kernel.drain_concurrent()
    assert len(dispatched) == 3
    assert tracker.per_instance_overlap == 0


def test_default_kernel_stays_serial():
    # max_workers defaults to 1: tick/drain behave exactly as before.
    tracker = _Tracker()
    kernel = Kernel()
    kernel.register("x", _FakeRuntime("x", tracker, delay=0.01))
    kernel.submit("x", object())
    assert len(kernel.drain()) == 1
    assert kernel.tick() is None
    # drain_concurrent on a serial kernel falls back to serial drain.
    kernel.submit("x", object())
    assert len(kernel.drain_concurrent()) == 1


def test_a_failing_task_never_takes_the_pool_down():
    class _Boom:
        def reason(self, intent, approval=None, claim=None):
            raise RuntimeError("kaboom")

    kernel = Kernel(max_workers=2)
    kernel.register("ok", _FakeRuntime("ok", _Tracker(), delay=0.01))
    kernel.register("bad", _Boom())
    kernel.submit("ok", object())
    kernel.submit("bad", object())
    dispatched = {d.instance: d.status for d in kernel.drain_concurrent()}
    assert dispatched["ok"] == "ran"
    assert dispatched["bad"] == "failed"
