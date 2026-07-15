"""Energy -- meter what a cycle burns, honestly.

Two sources of truth, never confused with each other:

* **Measured** -- on hosts that expose RAPL counters
  (`/sys/class/powercap/.../energy_uj`), `EnergyMeter.start()`/`stop()`
  reads real joules across the interval, wrap-safe. RAPL meters the whole
  package, not just this process, and the reading says so -- an honest
  whole-machine figure beats a fabricated per-process one.
* **Declared estimate** -- when the author wrote an `## Energy` section in
  memory.md ("reasoning costs about 0.4 watt-hours per thousand tokens"),
  the token spend converts at the declared rate, exactly the way Pricing
  converts to dollars. The rate is the author's declaration, never a table
  shipped in code.

With neither, a reading is **unmetered** -- an energy figure nobody measured
or declared is never invented, the same rule as everywhere else in EAR.

`EnergyBudget` enforces a prose-declared daily cap ("keep within 50
watt-hours per day") *before* a cycle starts: budgets are code, refusals are
loud (`EnergyBudgetExceeded`, a `PermissionError` like every governance stop),
and the spend is summed from the one audit spine -- the trail is the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

_RAPL = Path("/sys/class/powercap")


class EnergyBudgetExceeded(PermissionError):
    """A cycle was refused because the declared daily energy budget is
    spent. A PermissionError, like every other governance stop in EAR."""


def _rapl_zones(rapl: Path) -> list[Path]:
    """Top-level RAPL package zones (not subzones -- summing both would
    double-count the cores inside their package)."""
    try:
        return sorted(zone for zone in rapl.glob("intel-rapl:*") if (zone / "energy_uj").exists())
    except OSError:
        return []


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


@dataclass
class EnergyReading:
    """One interval's energy: measured joules (whole machine, when RAPL is
    exposed), an estimated watt-hours figure (when a rate was declared),
    and the basis -- so no reader ever mistakes an estimate for a
    measurement or vice versa."""

    measured_joules: Optional[float] = None
    estimated_wh: Optional[float] = None
    tokens: int = 0
    basis: str = "unmetered"

    @property
    def watt_hours(self) -> Optional[float]:
        """The best available figure: measured first, estimate second,
        None when neither exists."""
        if self.measured_joules is not None:
            return self.measured_joules / 3600
        return self.estimated_wh

    def describe(self) -> str:
        wh = self.watt_hours
        if wh is None:
            return "unmetered -- no RAPL counters and no declared energy rate"
        return f"{wh:.4f} Wh ({self.basis}, {self.tokens} tokens)"


@dataclass
class EnergyMeter:
    """Meter an interval: RAPL joules when the host exposes them, declared
    estimates from the strategy otherwise. `start()` snapshots; `stop()`
    returns the `EnergyReading` and, given a runtime, records it on the one
    audit spine."""

    strategy: Any = None
    rapl_root: Path = _RAPL
    _start_uj: dict[str, int] = field(default_factory=dict)

    def measurable(self) -> bool:
        return bool(_rapl_zones(self.rapl_root))

    def start(self) -> "EnergyMeter":
        self._start_uj = {}
        for zone in _rapl_zones(self.rapl_root):
            value = _read_int(zone / "energy_uj")
            if value is not None:
                self._start_uj[str(zone)] = value
        return self

    def stop(self, tokens: int = 0, runtime: Any = None) -> EnergyReading:
        measured = self._measure()
        estimated = None
        if self.strategy is not None:
            estimated = self.strategy.watt_hours(tokens)
        if measured is not None:
            basis = "measured, whole-machine RAPL over the interval"
        elif estimated is not None:
            basis = "declared estimate from the authored energy rate"
        else:
            basis = "unmetered"
        reading = EnergyReading(
            measured_joules=measured, estimated_wh=estimated, tokens=tokens, basis=basis
        )
        log = getattr(runtime, "reasoning_log", None)
        if log is not None:
            log.record(
                stage="energy",
                inputs={"tokens": tokens},
                output=reading.describe(),
                rationale=basis,
            )
        return reading

    def _measure(self) -> Optional[float]:
        """Joules burned since `start()`, summed across package zones and
        wrap-safe (a counter that wrapped adds its max range back). None
        when the host exposes no counters or start() was never called."""
        if not self._start_uj:
            return None
        total_uj = 0
        seen = False
        for zone in _rapl_zones(self.rapl_root):
            key = str(zone)
            if key not in self._start_uj:
                continue
            end = _read_int(zone / "energy_uj")
            if end is None:
                continue
            delta = end - self._start_uj[key]
            if delta < 0:
                max_range = _read_int(zone / "max_energy_range_uj")
                if max_range is None:
                    continue
                delta += max_range
            total_uj += delta
            seen = True
        return (total_uj / 1_000_000) if seen else None


@dataclass
class EnergyBudget:
    """The prose-declared daily energy cap, enforced in code before a cycle
    starts. The spend is estimated from the trail's own token records at
    the declared rate -- the one audit spine is also the energy ledger."""

    strategy: Any

    @property
    def wh_per_day(self) -> Optional[float]:
        return getattr(self.strategy, "energy_budget_wh", None)

    def spent_today_wh(self, log: Any) -> Optional[float]:
        """Today's estimated spend from the trail, or None when no energy
        rate was declared (a spend nobody can price is never invented)."""
        if getattr(self.strategy, "wh_per_thousand_tokens", None) is None:
            return None
        today = date.today()
        tokens = 0
        for record in getattr(log, "records", []):
            timestamp = getattr(record, "timestamp", None)
            if timestamp is None or timestamp.date() != today:
                continue
            tokens += getattr(record, "input_tokens", 0) + getattr(record, "output_tokens", 0)
        return self.strategy.watt_hours(tokens)

    def check(self, log: Any, runtime: Any = None) -> Optional[float]:
        """Refuse -- loudly, on the record -- when today's spend has reached
        the declared budget; otherwise return the remaining watt-hours (None
        when no budget or no rate was declared: an unbudgeted stack is
        simply not budgeted, never silently capped)."""
        budget = self.wh_per_day
        spent = self.spent_today_wh(log)
        if budget is None or spent is None:
            return None
        remaining = budget - spent
        if remaining <= 0:
            message = (
                f"daily energy budget spent: {spent:.3f} Wh of {budget:.3f} Wh used -- "
                "the cycle is refused until tomorrow or a larger declared budget"
            )
            record_log = getattr(runtime, "reasoning_log", None)
            if record_log is not None:
                record_log.record(stage="energy", inputs={"budget_wh": budget, "spent_wh": spent}, output="REFUSED", rationale=message)
            raise EnergyBudgetExceeded(message)
        return remaining
