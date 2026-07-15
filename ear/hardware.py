"""Hardware -- know the machine the runtime is standing on.

EAR's efficiency layer starts from an honest reading of the host: how many
CPUs this process may actually use (a container's cgroup quota, not the bare
`os.cpu_count()`), how much memory it may actually touch, whether it is on
battery power, and whether the hardware exposes real energy counters (RAPL)
so energy can be *measured* rather than estimated (`ear/energy.py`).

Everything here is detection and mechanics -- no judgment. It reads `/proc`,
`/sys` and cgroup files directly from the standard library, degrades
gracefully on hosts that expose none of them (macOS, restricted containers:
the profile simply reports less), and never guesses: a value the host does
not expose is `None`, not an invention -- the same honesty rule as pricing
and energy.

The probe functions take injectable root paths so the parsing is testable
against fixture trees without needing the real `/sys`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_CGROUP2 = Path("/sys/fs/cgroup")
_CGROUP1_CPU = Path("/sys/fs/cgroup/cpu")
_CGROUP1_MEMORY = Path("/sys/fs/cgroup/memory")
_POWER_SUPPLY = Path("/sys/class/power_supply")
_RAPL = Path("/sys/class/powercap")
_MEMINFO = Path("/proc/meminfo")


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def cgroup_cpu_limit(cgroup2: Path = _CGROUP2, cgroup1: Path = _CGROUP1_CPU) -> Optional[float]:
    """The CPU quota this process is confined to, in whole-CPU units
    (e.g. 1.5), or None when unconfined / not exposed. cgroup v2
    (`cpu.max`: 'quota period' or 'max ...') first, v1
    (`cpu.cfs_quota_us` / `cpu.cfs_period_us`) as the fallback."""
    text = _read_text(cgroup2 / "cpu.max")
    if text:
        parts = text.split()
        if parts and parts[0] != "max":
            try:
                quota, period = float(parts[0]), float(parts[1]) if len(parts) > 1 else 100_000.0
                if period > 0:
                    return quota / period
            except ValueError:
                pass
        if parts and parts[0] == "max":
            return None
    quota_text = _read_text(cgroup1 / "cpu.cfs_quota_us")
    period_text = _read_text(cgroup1 / "cpu.cfs_period_us")
    if quota_text and period_text:
        try:
            quota, period = float(quota_text), float(period_text)
            if quota > 0 and period > 0:
                return quota / period
        except ValueError:
            pass
    return None


def cgroup_memory_limit_mb(cgroup2: Path = _CGROUP2, cgroup1: Path = _CGROUP1_MEMORY) -> Optional[int]:
    """The memory cap this process is confined to, in MiB, or None when
    unconfined / not exposed."""
    for path in (cgroup2 / "memory.max", cgroup1 / "memory.limit_in_bytes"):
        text = _read_text(path)
        if text and text != "max":
            try:
                limit = int(text)
            except ValueError:
                continue
            # cgroup v1 reports "no limit" as a huge sentinel; treat
            # anything over 1 PiB as unconfined.
            if 0 < limit < (1 << 50):
                return limit // (1024 * 1024)
    return None


def host_memory_mb(meminfo: Path = _MEMINFO) -> Optional[int]:
    """Total host memory in MiB from /proc/meminfo, or None off-Linux."""
    text = _read_text(meminfo)
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            try:
                return int(parts[1]) // 1024  # kB -> MiB
            except (IndexError, ValueError):
                return None
    return None


@dataclass
class Battery:
    """The host's battery, when it has one: charge percent and whether it
    is currently discharging (i.e. the machine is running on stored
    energy, the moment to be frugal)."""

    percent: Optional[int] = None
    discharging: bool = False


def read_battery(power_supply: Path = _POWER_SUPPLY) -> Optional[Battery]:
    """The first battery under /sys/class/power_supply, or None when the
    host has none (servers, containers, macOS)."""
    try:
        supplies = sorted(power_supply.iterdir())
    except OSError:
        return None
    for supply in supplies:
        if (_read_text(supply / "type") or "").lower() != "battery":
            continue
        capacity = _read_text(supply / "capacity")
        status = (_read_text(supply / "status") or "").lower()
        percent: Optional[int] = None
        if capacity:
            try:
                percent = int(capacity)
            except ValueError:
                percent = None
        return Battery(percent=percent, discharging=status == "discharging")
    return None


def rapl_available(rapl: Path = _RAPL) -> bool:
    """Whether the host exposes RAPL energy counters -- the difference
    between *measuring* a cycle's energy and estimating it from declared
    rates (see `ear/energy.py`)."""
    try:
        return any(
            (zone / "energy_uj").exists()
            for zone in rapl.glob("intel-rapl*")
        ) or any((zone / "energy_uj").exists() for zone in rapl.glob("*/intel-rapl*"))
    except OSError:
        return False


def gpu_names(timeout: float = 3.0) -> list[str]:
    """The host's NVIDIA GPUs by name, via `nvidia-smi` when present --
    the one probe that shells out, and only when the binary exists. Any
    failure reads as 'no GPUs visible', never an exception."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return []
    try:
        result = subprocess.run(
            [binary, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


@dataclass
class HardwareProfile:
    """One honest reading of the machine: what this process may actually
    use, and what the host can tell us about power. Fields the host does
    not expose stay None -- reported, never invented."""

    cpus: int = 1
    cpu_quota: Optional[float] = None
    memory_mb: Optional[int] = None
    memory_limit_mb: Optional[int] = None
    load_1m: Optional[float] = None
    battery: Optional[Battery] = None
    energy_measurable: bool = False
    gpus: list[str] = field(default_factory=list)

    @classmethod
    def detect(cls, probe_gpus: bool = True) -> "HardwareProfile":
        cpus = os.cpu_count() or 1
        quota = cgroup_cpu_limit()
        if quota is not None:
            # The container's quota, not the node's core count, is what this
            # process may actually use.
            cpus = max(1, min(cpus, int(quota) or 1))
        try:
            load_1m: Optional[float] = os.getloadavg()[0]
        except (OSError, AttributeError):
            load_1m = None
        return cls(
            cpus=cpus,
            cpu_quota=quota,
            memory_mb=host_memory_mb(),
            memory_limit_mb=cgroup_memory_limit_mb(),
            load_1m=load_1m,
            battery=read_battery(),
            energy_measurable=rapl_available(),
            gpus=gpu_names() if probe_gpus else [],
        )

    def effective_memory_mb(self) -> Optional[int]:
        """The memory actually available to this process: the cgroup cap
        when confined, the host total otherwise."""
        if self.memory_limit_mb is not None and self.memory_mb is not None:
            return min(self.memory_limit_mb, self.memory_mb)
        return self.memory_limit_mb or self.memory_mb

    def on_battery(self) -> bool:
        """Whether the machine is running on stored energy right now --
        the moment the scheduler should be frugal."""
        return self.battery is not None and self.battery.discharging

    def recommended_workers(self, io_bound: bool = True, cap: int = 32) -> int:
        """A sensible parallel-pool size for this machine. EAR's cycles are
        I/O-bound on the model call (the GIL is released while waiting), so
        the pool may exceed the core count -- but it is tempered by current
        load and halved on battery, because parallel width multiplies power
        draw exactly when stored energy is the constraint. Mechanics, not
        judgment: deterministic from the profile."""
        base = self.cpus * 4 if io_bound else self.cpus
        if self.load_1m is not None and self.cpus > 0 and self.load_1m > self.cpus:
            # The machine is already saturated; do not pile on.
            base = max(1, base // 2)
        if self.on_battery():
            base = max(1, base // 2)
        return max(1, min(cap, base))

    def describe(self) -> str:
        """The profile in one plain-English paragraph -- what goes on the
        trail and into a reasoning context, so the model can weigh the
        machine it is running on like any other fact."""
        parts = [f"{self.cpus} CPU(s)" + (f" (cgroup quota {self.cpu_quota:g})" if self.cpu_quota else "")]
        memory = self.effective_memory_mb()
        if memory is not None:
            parts.append(f"{memory} MiB memory available")
        if self.load_1m is not None:
            parts.append(f"load {self.load_1m:.2f}")
        if self.battery is not None:
            state = "discharging" if self.battery.discharging else "on external power"
            charge = f" at {self.battery.percent}%" if self.battery.percent is not None else ""
            parts.append(f"battery{charge}, {state}")
        parts.append(
            "energy counters available (RAPL)" if self.energy_measurable else "no energy counters (estimates only)"
        )
        if self.gpus:
            parts.append(f"GPUs: {', '.join(self.gpus)}")
        return "; ".join(parts) + "."
