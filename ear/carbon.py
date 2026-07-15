"""Carbon -- run heavy work when the energy is clean and plentiful.

The energy plane (`ear/energy.py`) meters and budgets *how much* a cycle
burns; this plane governs *when* it burns. Two levers, both honest:

* **Grid intensity** -- the carbon a kilowatt-hour costs right now
  (gCO2/kWh). A deployment wires a live signal through the `provider` seam
  (WattTime, Electricity Maps, a national grid API -- out-of-band, so the
  zero-dependency core stays clean), or declares a coarse one in prose: a
  clean-hours window ("cleanest between 22:00 and 06:00") and/or a threshold
  ("defer heavy work above 300 gCO2/kWh"). With neither, intensity is
  `None` -- unknown, never invented.
* **Stored energy** -- when the machine is on battery and discharging
  (`HardwareProfile.on_battery`), deferrable heavy work waits, so a laptop or
  an off-grid node spends its stored charge on what is due now, not on what
  can run later.

`GridSignal.deferrable_runs_now()` is the one question the scheduler asks; a
deferrable task the grid says no to is rescheduled to `next_clean()`. The
gate applies only to work explicitly marked deferrable, and only when a
signal is configured -- everything else runs exactly as before. And carbon,
like watt-hours and dollars, only ever appears on the trail when it can
actually be computed (`carbon_grams`): a gram nobody could measure is never
written down.

`now` is injectable throughout, so the time-of-day logic is deterministic
and tested against fixed clocks, not the wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional


def carbon_grams(watt_hours: float, gco2_per_kwh: float) -> float:
    """Grams of CO2 for an energy spend at a grid intensity. Wh -> kWh times
    gCO2/kWh -- the one conversion, applied only when both numbers exist."""
    return (watt_hours / 1000.0) * gco2_per_kwh


def _in_window(hour: int, start: int, end: int) -> bool:
    """Whether `hour` falls in the [start, end) window, wrap-around aware
    (22->6 spans midnight: 22, 23, 0, 1, ... 5)."""
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


@dataclass
class CarbonIntensity:
    """One reading of grid carbon intensity: gCO2/kWh when known, the basis
    it came from, and (given a threshold) whether it counts as clean."""

    gco2_per_kwh: Optional[float] = None
    basis: str = "unknown"
    clean: Optional[bool] = None

    def describe(self) -> str:
        if self.gco2_per_kwh is None:
            state = "clean window" if self.clean else ("dirty window" if self.clean is False else "unknown")
            return f"grid intensity unknown ({self.basis}, {state})"
        state = "clean" if self.clean else ("dirty" if self.clean is False else "unrated")
        return f"{self.gco2_per_kwh:.0f} gCO2/kWh ({self.basis}, {state})"


@dataclass
class GridSignal:
    """When the grid is clean enough to run deferrable heavy work.

    Resolution order for the *number*: a live `provider` first, then a
    declared fixed intensity, else None. Resolution order for the
    *clean/dirty verdict*: a threshold against a known number first, then the
    declared clean-hours window, else None (can't say -- and an unknown
    verdict never defers work)."""

    provider: Optional[Callable[[], float]] = None
    declared_intensity: Optional[float] = None
    threshold: Optional[float] = None
    clean_hours: Optional[tuple[int, int]] = None
    defer_on_battery: bool = True

    def configured(self) -> bool:
        return any(
            (self.provider is not None, self.declared_intensity is not None, self.clean_hours is not None)
        )

    def _now(self, now: Optional[datetime]) -> datetime:
        return now or datetime.now()

    def intensity(self, now: Optional[datetime] = None) -> CarbonIntensity:
        moment = self._now(now)
        number: Optional[float] = None
        basis = "unknown"
        if self.provider is not None:
            try:
                number = float(self.provider())
                basis = "live grid provider"
            except Exception:  # noqa: BLE001 -- a dead provider is unknown, not fatal
                number = None
        if number is None and self.declared_intensity is not None:
            number = self.declared_intensity
            basis = "declared fixed intensity"
        clean = self._verdict(number, moment)
        if basis == "unknown" and self.clean_hours is not None:
            basis = "declared clean-hours window"
        return CarbonIntensity(gco2_per_kwh=number, basis=basis, clean=clean)

    def _verdict(self, number: Optional[float], moment: datetime) -> Optional[bool]:
        if self.threshold is not None and number is not None:
            return number <= self.threshold
        if self.clean_hours is not None:
            return _in_window(moment.hour, self.clean_hours[0], self.clean_hours[1])
        return None

    def is_clean(self, now: Optional[datetime] = None) -> Optional[bool]:
        """Whether now is clean enough to run, or None when the signal can't
        say. An unknown verdict is not 'dirty' -- it never defers work."""
        return self.intensity(now).clean

    def deferrable_runs_now(self, now: Optional[datetime] = None, profile: Any = None) -> tuple[bool, str]:
        """The scheduler's question for a deferrable task: run now, or wait?
        Waits when the grid is explicitly dirty, or when the machine is on
        battery (stored energy is spent on what is due now). Returns
        (run, reason) so the decision rides the record."""
        if self.defer_on_battery and profile is not None and getattr(profile, "on_battery", lambda: False)():
            return False, "on battery -- deferring deferrable work to preserve stored energy"
        verdict = self.is_clean(now)
        if verdict is False:
            reading = self.intensity(now)
            return False, f"grid not clean ({reading.describe()}) -- deferring to the next clean window"
        if verdict is True:
            return True, "grid is clean -- running now"
        return True, "grid intensity unknown -- not deferring on an unknown"

    def next_clean(self, now: Optional[datetime] = None) -> Optional[datetime]:
        """The next moment a deferrable task should be retried: the start of
        the next clean-hours window. None when cleanliness is threshold-based
        with no window (the future intensity can't be predicted -- the
        scheduler falls back to a fixed retry instead)."""
        if self.clean_hours is None:
            return None
        moment = self._now(now)
        start, end = self.clean_hours
        if _in_window(moment.hour, start, end):
            return moment
        probe = moment.replace(minute=0, second=0, microsecond=0)
        for _ in range(48):  # search up to two days of hours
            probe += timedelta(hours=1)
            if _in_window(probe.hour, start, end):
                return probe
        return None
