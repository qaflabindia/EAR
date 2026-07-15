"""Tests for carbon-aware scheduling -- `ear/carbon.py`, the `## Carbon`
strategy section, the EnergyMeter's carbon accounting, and the Kernel's
deferral of deferrable work.

All offline and deterministic: the grid signal takes an injectable `now`, so
the time-of-day logic is tested against fixed clocks, and the live-provider
seam is exercised with a plain callable.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from ear import CarbonIntensity, GridSignal, Runtime, Strategy, carbon_grams
from ear.energy import EnergyMeter
from ear.kernel import Kernel


# ---------------------------------------------------------------------------
# The conversion and the signal.
# ---------------------------------------------------------------------------


def test_carbon_grams_conversion():
    assert carbon_grams(1000, 250) == pytest.approx(250.0)  # 1 kWh @ 250 gCO2/kWh
    assert carbon_grams(2000, 250) == pytest.approx(500.0)


def test_clean_hours_window_wraps_midnight():
    grid = GridSignal(clean_hours=(22, 6))
    assert grid.is_clean(datetime(2026, 7, 15, 3)) is True
    assert grid.is_clean(datetime(2026, 7, 15, 23)) is True
    assert grid.is_clean(datetime(2026, 7, 15, 14)) is False


def test_next_clean_finds_the_window_start():
    grid = GridSignal(clean_hours=(22, 6))
    nxt = grid.next_clean(datetime(2026, 7, 15, 14, 30))
    assert nxt.hour == 22
    # Already inside the window -> now is the answer.
    inside = datetime(2026, 7, 15, 3)
    assert grid.next_clean(inside) == inside


def test_threshold_against_a_live_provider():
    dirty = GridSignal(provider=lambda: 400, threshold=300)
    clean = GridSignal(provider=lambda: 120, threshold=300)
    assert dirty.is_clean() is False
    assert clean.is_clean() is True


def test_declared_fixed_intensity():
    grid = GridSignal(declared_intensity=250, threshold=300)
    reading = grid.intensity()
    assert reading.gco2_per_kwh == 250
    assert reading.clean is True
    assert "declared" in reading.basis


def test_unknown_intensity_is_never_invented_and_never_defers():
    grid = GridSignal()
    assert grid.is_clean() is None
    runs, reason = grid.deferrable_runs_now()
    assert runs is True  # an unknown verdict never holds work back
    assert "unknown" in reason


def test_dead_provider_reads_as_unknown_not_fatal():
    def boom():
        raise RuntimeError("grid API down")

    grid = GridSignal(provider=boom, clean_hours=(22, 6))
    # Falls back to the window verdict rather than raising.
    assert grid.is_clean(datetime(2026, 7, 15, 3)) is True


def test_on_battery_defers_deferrable_work():
    class _OnBattery:
        def on_battery(self):
            return True

    grid = GridSignal(clean_hours=(0, 24))  # always "clean" by window
    runs, reason = grid.deferrable_runs_now(now=datetime(2026, 7, 15, 3), profile=_OnBattery())
    assert runs is False
    assert "battery" in reason


def test_intensity_describe():
    assert "gCO2" in CarbonIntensity(gco2_per_kwh=250, clean=True).describe()
    assert "unknown" in CarbonIntensity().describe()


# ---------------------------------------------------------------------------
# The ## Carbon strategy section.
# ---------------------------------------------------------------------------

CARBON_STACK = """# Memory & Strategy

## Carbon

The grid is cleanest between 22:00 and 06:00. Defer heavy work above 300
gCO2/kWh.
"""


def test_carbon_section_parses_window_and_threshold():
    strategy = Strategy.from_markdown(CARBON_STACK)
    assert strategy.clean_hours == (22, 6)
    assert strategy.carbon_threshold_gco2 == 300.0


def test_grid_signal_from_strategy():
    strategy = Strategy.from_markdown(CARBON_STACK)
    grid = strategy.grid_signal()
    assert grid is not None
    assert grid.is_clean(datetime(2026, 7, 15, 23)) is True


def test_no_carbon_section_no_signal():
    assert Strategy.from_markdown("# Memory & Strategy\n").grid_signal() is None


# ---------------------------------------------------------------------------
# Energy meter records carbon when intensity is known.
# ---------------------------------------------------------------------------


def test_energy_reading_carries_carbon(tmp_path):
    strategy = Strategy.from_markdown(
        "# M\n\n## Energy\n\nReasoning costs 0.4 watt-hours per thousand tokens.\n\n"
        "## Carbon\n\nGrid intensity is about 250 gCO2/kWh.\n"
    )
    runtime = Runtime(name="C")
    meter = EnergyMeter(strategy=strategy, grid=strategy.grid_signal(), rapl_root=tmp_path / "none")
    reading = meter.stop(tokens=5000, runtime=runtime)
    assert reading.watt_hours == pytest.approx(2.0)
    assert reading.carbon_grams == pytest.approx(0.5)  # 2 Wh -> 0.002 kWh * 250
    assert "gCO2" in runtime.reasoning_log.for_stage("energy")[-1].output


def test_carbon_absent_when_intensity_unknown(tmp_path):
    strategy = Strategy.from_markdown(
        "# M\n\n## Energy\n\nReasoning costs 0.4 watt-hours per thousand tokens.\n"
    )
    runtime = Runtime(name="C")
    reading = EnergyMeter(strategy=strategy, rapl_root=tmp_path / "none").stop(tokens=1000, runtime=runtime)
    assert reading.carbon_grams is None  # no grid -> no carbon invented


# ---------------------------------------------------------------------------
# Kernel deferral of deferrable tasks.
# ---------------------------------------------------------------------------


class _Runtime:
    def reason(self, intent, approval=None, claim=None):
        return "ok"


def test_dirty_grid_defers_deferrable_but_runs_normal():
    kernel = Kernel(grid=GridSignal(provider=lambda: 500, threshold=300))
    kernel.register("a", _Runtime())
    deferrable = kernel.submit("a", object(), deferrable=True)
    kernel.submit("a", object(), deferrable=False)
    ran = kernel.drain()
    assert len(ran) == 1 and ran[0].status == "ran"
    assert any(d.status == "deferred" for d in kernel.history)
    assert deferrable.deferrals == 1
    assert kernel.pending == 1  # the deferrable task is still queued


def test_clean_grid_runs_deferrable():
    kernel = Kernel(grid=GridSignal(provider=lambda: 100, threshold=300))
    kernel.register("a", _Runtime())
    kernel.submit("a", object(), deferrable=True)
    assert len(kernel.drain()) == 1


def test_no_grid_ignores_deferrable():
    kernel = Kernel()  # no grid signal
    kernel.register("a", _Runtime())
    kernel.submit("a", object(), deferrable=True)
    assert len(kernel.drain()) == 1


def test_deferred_task_reschedules_to_the_backoff_when_no_window():
    # A threshold-only signal can't predict a future clean window, so the
    # deferred task is pushed to the fixed backoff, not lost.
    kernel = Kernel(grid=GridSignal(provider=lambda: 900, threshold=300), carbon_backoff=1234.0)
    kernel.register("a", _Runtime())
    task = kernel.submit("a", object(), deferrable=True)
    kernel.drain()
    assert task.deferrals == 1
    # due pushed roughly backoff seconds out (monotonic clock)
    import time as _time

    assert task.due > _time.monotonic() + 1000
