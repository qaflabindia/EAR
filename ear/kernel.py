"""Kernel -- EAR as a CPU/OS scheduler: a run loop that dispatches work to
runtime instances when there is work, and sleeps until an interrupt when
there is not.

    while running:
        if there_is_work:
            run_work()          # dispatch the next ready task to its instance
        else:
            sleep_until_interrupt()   # block until a task is submitted or a timer fires

That is the whole of it, and it is exactly a kernel's idle loop. The
Kernel holds a **process table** -- named `Runtime` instances, each with
its own sandbox, memory and trail -- and a **run queue** of tasks. A task
names the instance to run on and the intent (or goal) to run; `submit()`
enqueues one and wakes the loop, the way a syscall raises an interrupt.
`schedule(..., every=…)` makes a task recur, the way a timer interrupt
fires on a period -- so a runtime "stays live for the recurring occurrence
of tasks" without a busy-wait: between firings the kernel genuinely sleeps.

Dispatch runs the target instance's normal cycle (`reason`, or `pursue`
for a goal), so every guarantee still holds -- policies gate it, the
sandbox confines it, the trail records it. A governance stop is not a
crash: an approval gate parks the task as `blocked`, a refusal as
`blocked`, an error as `failed`, and the kernel keeps running the rest.
Nothing here reasons; the Kernel only decides *when* work runs -- the
control plane, hardwired, while the judgment stays in the instances.

Zero dependencies: the loop is a `threading.Event` (the interrupt line)
and `time.monotonic` (the clock). `tick()` / `drain()` advance it
synchronously (the testable heartbeat); `start()` / `run_forever()` drive
it in the background until `stop()`.
"""

from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional, Union

_task_ids = itertools.count(1)


@dataclass
class Task:
    """One unit of scheduled work: which instance (process) to run it on,
    the intent to run, an optional goal to pursue instead, and -- when
    recurring -- the period and the next time it is due.

    `approval` carries a human's verdict released through a network control
    plane (no shared filesystem to drop an `approval.md` beside `intent.md`)
    -- it is threaded straight through to `Runtime.reason(intent, approval=...)`,
    the same parameter `Exchange` uses for the file-drop case.

    `claim` (an `ear.identity.Claim`) is the tenant boundary check for
    scheduled work: threaded to `Runtime.reason(intent, claim=...)` at
    dispatch time, so a task submitted for the wrong org lands `blocked`
    rather than touching the instance's data."""

    instance: str
    intent: Any
    goal: Optional[str] = None
    every: Optional[float] = None
    due: float = 0.0
    id: int = field(default_factory=lambda: next(_task_ids))
    runs: int = 0
    approval: Any = None
    claim: Any = None

    @property
    def recurring(self) -> bool:
        return self.every is not None


@dataclass
class Dispatch:
    """The record of one task run: what happened, and how long it took."""

    task_id: int
    instance: str
    status: str  # "ran" | "blocked" | "failed"
    summary: str = ""
    duration_ms: int = 0


@dataclass
class Kernel:
    """The scheduler: a process table of runtime instances and a run queue,
    driven by the kernel idle loop."""

    instances: dict = field(default_factory=dict)
    queue: list = field(default_factory=list)
    history: list = field(default_factory=list)
    running: bool = False
    idle_waits: int = 0
    # An optional execution seam: `dispatcher(task, runtime) -> (status,
    # summary)`. Unset, work runs in-process (reason/pursue); set to a
    # KubeProvider.as_dispatcher(), each firing runs on the cluster instead,
    # while the Kernel stays the single scheduler.
    dispatcher: Any = None
    # Fleet parallelism: with `max_workers > 1` the loop runs work for
    # *different* instances concurrently on a bounded thread pool, while
    # keeping at most one in-flight cycle *per instance* -- the single-writer
    # actor invariant, so an instance's memory and hash-chained audit spine
    # never have two writers. The default (1) is the historical serial
    # scheduler, unchanged. Threads (not processes) because a cycle is
    # I/O-bound on the model call, which releases the GIL. `0` means
    # hardware-aware: size the pool from the detected machine (cgroup CPU
    # quota, current load, battery state -- see HardwareProfile
    # .recommended_workers), resolved once on first use.
    max_workers: int = 1
    _wake: Any = field(default_factory=threading.Event)
    _lock: Any = field(default_factory=threading.Lock)
    _thread: Any = None
    _pool: Any = None
    _in_flight: set = field(default_factory=set)

    # -- the process table ----------------------------------------------------

    def register(self, name: str, runtime: Any) -> Any:
        """Add a runtime instance to the process table under `name`."""
        with self._lock:
            self.instances[name] = runtime
        return runtime

    def remove(self, name: str) -> None:
        with self._lock:
            self.instances.pop(name, None)

    # -- submitting work (the interrupt) --------------------------------------

    def submit(
        self,
        instance: str,
        intent: Any,
        goal: Optional[str] = None,
        every: Optional[float] = None,
        delay: float = 0.0,
        approval: Any = None,
        claim: Any = None,
    ) -> Task:
        """Enqueue work for an instance and wake the loop -- a syscall
        raising an interrupt. `every` makes it recur on that period;
        `delay` defers its first run. `approval` releases a cycle an
        approval-gated policy previously parked (see `Task.approval`).
        `claim` (an `ear.identity.Claim`) is checked against the target
        instance's tenant at dispatch time (see `Task.claim`)."""
        task = Task(
            instance=instance,
            intent=intent,
            goal=goal,
            every=every,
            due=time.monotonic() + delay,
            approval=approval,
            claim=claim,
        )
        with self._lock:
            self.queue.append(task)
        self._wake.set()  # interrupt: there may be work now
        return task

    def schedule(self, instance: str, intent: Any, every: float, goal: Optional[str] = None) -> Task:
        """A recurring task -- a timer interrupt firing every `every`
        seconds -- so an instance stays live for the recurring occurrence
        of a task without a busy-wait."""
        return self.submit(instance, intent, goal=goal, every=every, delay=every)

    def cancel(self, task: Union[Task, int]) -> bool:
        task_id = task.id if isinstance(task, Task) else task
        with self._lock:
            before = len(self.queue)
            self.queue = [item for item in self.queue if item.id != task_id]
            return len(self.queue) < before

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self.queue)

    # -- the loop, one step at a time -----------------------------------------

    def tick(self) -> Optional[Dispatch]:
        """One turn of the loop: if a task is ready, run it; otherwise
        report idle (return None). Non-blocking -- the synchronous
        heartbeat the live loop is built from and tests drive directly."""
        now = time.monotonic()
        task = self._take_ready(now)
        if task is None:
            self.idle_waits += 1
            return None
        return self._dispatch(task, now)

    def drain(self, max_units: int = 10_000) -> list:
        """Run every task that is ready right now, in due order, and return
        what ran. The way to advance the kernel synchronously."""
        done: list = []
        for _ in range(max_units):
            dispatch = self.tick()
            if dispatch is None:
                break
            done.append(dispatch)
        return done

    def _take_ready(self, now: float) -> Optional[Task]:
        """The soonest-due task whose time has come, removed from the queue
        (or rescheduled if recurring). None when nothing is ready."""
        with self._lock:
            ready = sorted((task for task in self.queue if task.due <= now), key=lambda task: task.due)
            if not ready:
                return None
            task = ready[0]
            if task.recurring:
                task.due = now + task.every
            else:
                self.queue.remove(task)
            return task

    # -- fleet parallelism: many instances at once, one cycle per instance ----

    def drain_concurrent(self, max_units: int = 10_000) -> list:
        """Run every ready task, fanning out across *different* instances on
        the pool while serializing work *within* an instance -- the
        synchronous way to advance a parallel kernel (tests drive it
        directly). Falls back to the serial `drain` when `max_workers <= 1`,
        so the default kernel is unchanged."""
        if self._resolved_workers() <= 1:
            return self.drain(max_units)
        pool = self._ensure_pool()
        dispatched: list = []
        while len(dispatched) < max_units:
            futures = []
            while True:
                task = self._take_ready_free(time.monotonic())
                if task is None:
                    break
                futures.append(pool.submit(self._dispatch_and_release, task))
            if not futures:
                break
            for future in futures:
                dispatched.append(future.result())
        return dispatched

    def _take_ready_free(self, now: float) -> Optional[Task]:
        """The soonest-due ready task *whose instance is not already running*,
        atomically reserving that instance in-flight -- at most one cycle per
        instance keeps each a single writer of its own memory and trail."""
        with self._lock:
            ready = sorted(
                (
                    task
                    for task in self.queue
                    if task.due <= now and task.instance not in self._in_flight
                ),
                key=lambda task: task.due,
            )
            if not ready:
                return None
            task = ready[0]
            if task.recurring:
                task.due = now + task.every
            else:
                self.queue.remove(task)
            self._in_flight.add(task.instance)  # atomic take + reserve
            return task

    def _dispatch_and_release(self, task: Task) -> Dispatch:
        try:
            return self._dispatch(task, time.monotonic())
        finally:
            with self._lock:
                self._in_flight.discard(task.instance)

    def _submit_ready(self) -> int:
        """Submit every ready, free-instance task to the pool without
        blocking, returning how many were submitted. The non-blocking step
        the parallel idle loop is built from."""
        pool = self._ensure_pool()
        submitted = 0
        while True:
            task = self._take_ready_free(time.monotonic())
            if task is None:
                break
            pool.submit(self._dispatch_and_release, task)
            submitted += 1
        return submitted

    def _busy(self) -> bool:
        with self._lock:
            return bool(self._in_flight)

    def _ensure_pool(self):
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self._resolved_workers(), thread_name_prefix="ear-kernel")
        return self._pool

    def _resolved_workers(self) -> int:
        """The pool size, resolving `max_workers=0` to a hardware-aware
        recommendation once (cgroup quota, load, battery)."""
        if self.max_workers == 0:
            from .hardware import HardwareProfile

            self.max_workers = HardwareProfile.detect(probe_gpus=False).recommended_workers()
        return max(1, self.max_workers)

    def _dispatch(self, task: Task, now: float) -> Dispatch:
        """run_work(): run the task on its instance through the normal
        cycle. A governance stop parks it (blocked), an error fails it --
        neither takes the kernel down."""
        from .approval import ApprovalRequired

        runtime = self.instances.get(task.instance)
        started = time.monotonic()
        if runtime is None:
            dispatch = Dispatch(task.id, task.instance, "failed", f"no such instance '{task.instance}'")
        elif self.dispatcher is not None:
            # Hand the work to the execution seam (e.g. run it on Kubernetes)
            # rather than in-process. The Kernel still schedules; the pod runs.
            try:
                status, summary = self.dispatcher(task, runtime)
            except Exception as error:  # noqa: BLE001 -- a dispatch failure never takes the kernel down
                status, summary = "failed", str(error)
            dispatch = Dispatch(
                task.id, task.instance, status, str(summary)[:240], int((time.monotonic() - started) * 1000)
            )
        else:
            try:
                if task.goal is not None:
                    outcome = runtime.pursue(task.goal, task.intent)
                    status, summary = "ran", f"goal {outcome.status}: {outcome.blocker}"
                else:
                    decision = runtime.reason(task.intent, approval=task.approval, claim=task.claim)
                    status, summary = "ran", str(decision)[:240]
            except ApprovalRequired as parked:
                status, summary = "blocked", f"awaiting approval: {parked}"
            except PermissionError as blocked:
                status, summary = "blocked", str(blocked)
            except Exception as error:  # noqa: BLE001 -- one task's failure never takes the kernel down
                status, summary = "failed", str(error)
            dispatch = Dispatch(
                task.id, task.instance, status, summary, int((time.monotonic() - started) * 1000)
            )
        task.runs += 1
        self.history.append(dispatch)
        return dispatch

    def _seconds_until_next(self, now: float) -> Optional[float]:
        """How long until the next scheduled task is due, or None when the
        queue is empty -- the timeout the idle sleep waits for before the
        next timer interrupt."""
        with self._lock:
            future = [task.due for task in self.queue if task.due > now]
        return (min(future) - now) if future else None

    # -- the blocking idle loop -----------------------------------------------

    def run_forever(self) -> None:  # pragma: no cover - blocking loop
        """while running: if there is work, run it; else sleep until an
        interrupt. Serial by default; with `max_workers > 1` it fans ready
        work across the pool (one cycle per instance) and only sleeps when
        nothing is ready and nothing is in flight."""
        self.running = True
        parallel = self._resolved_workers() > 1
        while self.running:
            if parallel:
                if self._submit_ready() == 0 and not self._busy():
                    self._sleep_until_interrupt()
            elif self.tick() is None:
                self._sleep_until_interrupt()

    def _sleep_until_interrupt(self) -> None:  # pragma: no cover - blocks
        """Block until a task is submitted (the Event is set) or the next
        timer is due (the wait times out). No busy-wait: the CPU is idle
        here until something actually needs doing."""
        timeout = self._seconds_until_next(time.monotonic())
        self._wake.wait(timeout=timeout)
        self._wake.clear()

    def start(self) -> "Kernel":
        """Drive the loop in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self.running = True
        self._thread = threading.Thread(target=self.run_forever, name="ear-kernel", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Halt the loop and wake it so it exits promptly."""
        self.running = False
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> "Kernel":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- the process table, for a monitor -------------------------------------

    def snapshot(self) -> dict:
        """A glance at the scheduler for a control-room view: the process
        table, the queue depth, and how the last dispatches went."""
        with self._lock:
            return {
                "instances": list(self.instances),
                "pending": len(self.queue),
                "running": self.running,
                "idle_waits": self.idle_waits,
                "dispatched": len(self.history),
                "recent": [
                    {"instance": d.instance, "status": d.status, "summary": d.summary}
                    for d in self.history[-8:]
                ],
            }
