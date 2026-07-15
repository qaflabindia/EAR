"""Tests for the resource plane -- hardware awareness (`ear/hardware.py`),
energy metering and budgets (`ear/energy.py` + the `## Energy` strategy
section), and compute-thrift model routing (`ear/thrift.py`).

All offline and deterministic. Hardware probes run against injected fixture
trees (a fake /sys), never the real host's; the meter's RAPL path is
exercised with fake counter files including a wrap; the thrift judge's
deterministic fallback is exercised without a model. One live test covers
the judged thrift path, skipped without a key.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ear import (
    EnergyBudget,
    EnergyBudgetExceeded,
    EnergyMeter,
    HardwareProfile,
    Intent,
    ModelBinding,
    ModelThrift,
    Runtime,
    Strategy,
)
from ear.hardware import Battery, cgroup_cpu_limit, cgroup_memory_limit_mb, read_battery
from ear.thrift import HEAVY, LIGHT

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_TEST_MODEL = os.environ.get("ANTHROPIC_TEST_MODEL", "claude-haiku-4-5")

requires_anthropic_key = pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY is not set in the environment -- live-LLM tests are skipped",
)


# ---------------------------------------------------------------------------
# Hardware -- honest detection, injectable roots.
# ---------------------------------------------------------------------------


def test_detect_returns_a_usable_profile():
    profile = HardwareProfile.detect(probe_gpus=False)
    assert profile.cpus >= 1
    assert profile.recommended_workers() >= 1
    assert profile.describe()  # one plain-English paragraph, always


def test_cgroup_v2_cpu_quota(tmp_path):
    (tmp_path / "cpu.max").write_text("150000 100000")
    assert cgroup_cpu_limit(cgroup2=tmp_path, cgroup1=tmp_path / "none") == 1.5


def test_cgroup_v2_unconfined(tmp_path):
    (tmp_path / "cpu.max").write_text("max 100000")
    assert cgroup_cpu_limit(cgroup2=tmp_path, cgroup1=tmp_path / "none") is None


def test_cgroup_v1_cpu_quota(tmp_path):
    v1 = tmp_path / "cpu"
    v1.mkdir()
    (v1 / "cpu.cfs_quota_us").write_text("200000")
    (v1 / "cpu.cfs_period_us").write_text("100000")
    assert cgroup_cpu_limit(cgroup2=tmp_path / "none", cgroup1=v1) == 2.0


def test_cgroup_memory_limit(tmp_path):
    (tmp_path / "memory.max").write_text(str(512 * 1024 * 1024))
    assert cgroup_memory_limit_mb(cgroup2=tmp_path, cgroup1=tmp_path / "none") == 512


def test_memory_huge_sentinel_reads_as_unconfined(tmp_path):
    (tmp_path / "memory.max").write_text(str(1 << 60))
    assert cgroup_memory_limit_mb(cgroup2=tmp_path, cgroup1=tmp_path / "none") is None


def test_battery_read_and_absence(tmp_path):
    battery_dir = tmp_path / "BAT0"
    battery_dir.mkdir()
    (battery_dir / "type").write_text("Battery")
    (battery_dir / "capacity").write_text("37")
    (battery_dir / "status").write_text("Discharging")
    battery = read_battery(power_supply=tmp_path)
    assert battery == Battery(percent=37, discharging=True)
    assert read_battery(power_supply=tmp_path / "none") is None


def test_workers_halved_on_battery():
    mains = HardwareProfile(cpus=4)
    on_battery = HardwareProfile(cpus=4, battery=Battery(percent=30, discharging=True))
    assert on_battery.recommended_workers() == mains.recommended_workers() // 2
    assert on_battery.on_battery()


def test_workers_tempered_by_load():
    idle = HardwareProfile(cpus=4, load_1m=0.2)
    saturated = HardwareProfile(cpus=4, load_1m=9.0)
    assert saturated.recommended_workers() < idle.recommended_workers()


def test_missing_values_are_none_not_invented():
    profile = HardwareProfile(cpus=2)
    assert profile.effective_memory_mb() is None
    assert "estimates only" in profile.describe()


# ---------------------------------------------------------------------------
# Energy -- the ## Energy section, the meter, the budget.
# ---------------------------------------------------------------------------

ENERGY_STACK = """# Memory & Strategy

## Energy

Reasoning costs about 0.4 watt-hours per thousand tokens. Keep within a
budget of 50 watt-hours per day.
"""


def test_energy_section_parses_decimal_rate_and_budget():
    strategy = Strategy.from_markdown(ENERGY_STACK)
    assert strategy.wh_per_thousand_tokens == 0.4
    assert strategy.energy_budget_wh == 50.0
    assert strategy.watt_hours(2500) == 1.0


def test_undeclared_energy_is_never_invented():
    strategy = Strategy.from_markdown("# Memory & Strategy\n")
    assert strategy.watt_hours(1_000_000) is None


def _fake_rapl(tmp_path: Path, start_uj: int = 1_000_000) -> Path:
    zone = tmp_path / "intel-rapl:0"
    zone.mkdir(parents=True)
    (zone / "energy_uj").write_text(str(start_uj))
    (zone / "max_energy_range_uj").write_text(str(10_000_000))
    return zone


def test_meter_measures_rapl_joules(tmp_path):
    zone = _fake_rapl(tmp_path)
    meter = EnergyMeter(rapl_root=tmp_path).start()
    zone.joinpath("energy_uj").write_text(str(1_000_000 + 3_600_000))  # +3.6 J
    reading = meter.stop()
    assert reading.measured_joules == pytest.approx(3.6)
    assert reading.watt_hours == pytest.approx(0.001)
    assert "measured" in reading.basis


def test_meter_survives_a_counter_wrap(tmp_path):
    zone = _fake_rapl(tmp_path, start_uj=9_500_000)
    meter = EnergyMeter(rapl_root=tmp_path).start()
    zone.joinpath("energy_uj").write_text(str(100_000))  # wrapped past max
    reading = meter.stop()
    assert reading.measured_joules == pytest.approx(0.6)


def test_meter_estimates_from_the_declared_rate(tmp_path):
    strategy = Strategy.from_markdown(ENERGY_STACK)
    meter = EnergyMeter(strategy=strategy, rapl_root=tmp_path / "none")
    reading = meter.stop(tokens=2000)
    assert reading.measured_joules is None
    assert reading.watt_hours == pytest.approx(0.8)
    assert "declared estimate" in reading.basis


def test_meter_is_honest_when_unmetered(tmp_path):
    reading = EnergyMeter(rapl_root=tmp_path / "none").stop(tokens=999)
    assert reading.watt_hours is None
    assert "unmetered" in reading.describe()


def test_meter_records_on_the_spine(tmp_path):
    runtime = Runtime(name="E")
    strategy = Strategy.from_markdown(ENERGY_STACK)
    EnergyMeter(strategy=strategy, rapl_root=tmp_path / "none").stop(tokens=1000, runtime=runtime)
    records = runtime.reasoning_log.for_stage("energy")
    assert len(records) == 1
    assert "Wh" in records[0].output


def test_budget_allows_then_refuses_loudly():
    strategy = Strategy.from_markdown(ENERGY_STACK)
    runtime = Runtime(name="E")
    runtime.reasoning_log.record(
        stage="deliberation", output="x", usage={"input_tokens": 10_000, "output_tokens": 2_500}
    )
    budget = EnergyBudget(strategy=strategy)
    assert budget.spent_today_wh(runtime.reasoning_log) == pytest.approx(5.0)
    assert budget.check(runtime.reasoning_log) == pytest.approx(45.0)

    runtime.reasoning_log.record(
        stage="deliberation", output="x", usage={"input_tokens": 120_000, "output_tokens": 10_000}
    )
    with pytest.raises(EnergyBudgetExceeded):
        budget.check(runtime.reasoning_log, runtime=runtime)
    assert runtime.reasoning_log.for_stage("energy")[-1].output == "REFUSED"


def test_unbudgeted_stack_is_not_silently_capped():
    strategy = Strategy.from_markdown("# Memory & Strategy\n")
    runtime = Runtime(name="E")
    assert EnergyBudget(strategy=strategy).check(runtime.reasoning_log) is None


# ---------------------------------------------------------------------------
# Thrift -- the smallest adequate model.
# ---------------------------------------------------------------------------


def _ladder() -> ModelThrift:
    return ModelThrift(
        light=ModelBinding(provider="anthropic", model="claude-haiku-4-5"),
        heavy=ModelBinding(provider="anthropic", model="claude-opus-4-8"),
    )


def test_fallback_routes_short_work_light_and_long_work_heavy():
    thrift = _ladder()
    short = thrift.choose(Intent(text="format this date", context={}))
    long = thrift.choose(Intent(text="word " * 200, context={}))
    assert short.tier == LIGHT and "fallback" in short.basis
    assert long.tier == HEAVY


def test_choice_lands_on_the_spine():
    runtime = Runtime(name="E")
    _ladder().choose(Intent(text="quick lookup", context={}), runtime=runtime)
    records = runtime.reasoning_log.for_stage("thrift")
    assert len(records) == 1
    assert "light" in records[0].output


def test_bind_sets_a_reachable_binding_for_the_cycle():
    # A reachable tier (here via a local api_base) is set as the cycle's
    # binding; an unreachable one would leave the runtime on its
    # deterministic fallback instead of pointing at an unconfigured endpoint.
    reachable = ModelThrift(
        light=ModelBinding(provider="openai", model="local-small", api_base="http://localhost:11434/v1"),
        heavy=ModelBinding(provider="openai", model="local-big", api_base="http://localhost:11434/v1"),
    )
    runtime = Runtime(name="E")
    choice = reachable.bind(runtime, Intent(text="short", context={}))
    assert runtime.model_binding is choice.binding


def test_bind_degrades_to_deterministic_when_unreachable():
    runtime = Runtime(name="E")
    _ladder().bind(runtime, Intent(text="short", context={}))  # anthropic, no key here
    assert runtime.model_binding is None


@requires_anthropic_key
def test_light_model_judges_complexity_live():
    thrift = _ladder()
    simple = thrift.choose(Intent(text="Extract the year from '2026-07-15'.", context={}))
    hard = thrift.choose(
        Intent(
            text=(
                "Design a phased migration of our core-banking ledger to an "
                "event-sourced architecture, weighing regulatory constraints, "
                "rollback strategy and data-integrity guarantees."
            ),
            context={},
        )
    )
    assert simple.basis == "judged by the light model"
    assert simple.tier == LIGHT
    assert hard.tier == HEAVY


# ---------------------------------------------------------------------------
# End-to-end: the resource plane wired into reason().
# ---------------------------------------------------------------------------


def _energy_stack(tmp_path: Path) -> Path:
    (tmp_path / "process.md").write_text("# R\n\n## Do\n\nHandle the request.\n")
    (tmp_path / "memory.md").write_text(ENERGY_STACK)
    return tmp_path


def test_loader_wires_energy_from_the_section(tmp_path):
    from ear import load_runtime

    runtime = load_runtime(_energy_stack(tmp_path))
    assert runtime.energy_meter is not None
    assert runtime.energy_budget is not None


def test_a_cycle_records_its_energy(tmp_path):
    from ear import load_runtime

    runtime = load_runtime(_energy_stack(tmp_path))
    runtime.reason(Intent(text="hello", context={}))
    energy = runtime.reasoning_log.for_stage("energy")
    assert len(energy) == 1  # one energy record per cycle


def test_a_cycle_is_refused_when_the_daily_budget_is_spent(tmp_path):
    from ear import load_runtime

    runtime = load_runtime(_energy_stack(tmp_path))
    # Pre-load the trail with a spend beyond the 50 Wh/day budget.
    runtime.reasoning_log.record(
        stage="deliberation", output="x", usage={"input_tokens": 130_000, "output_tokens": 10_000}
    )
    with pytest.raises(EnergyBudgetExceeded):
        runtime.reason(Intent(text="again", context={}))


def test_thrift_wired_into_reason_records_and_degrades():
    from ear import Process, Workflow

    runtime = Runtime(name="T")
    process = Process(name="p")
    process.add_workflow(Workflow(name="w"))
    runtime.add_process(process)
    runtime.enable_thrift(
        ModelBinding(provider="anthropic", model="claude-haiku-4-5"),
        ModelBinding(provider="anthropic", model="claude-opus-4-8"),
    )
    runtime.reason(Intent(text="a short routine task", context={}))
    thrift = runtime.reasoning_log.for_stage("thrift")
    assert len(thrift) == 1
    assert "light" in thrift[0].output
    # Unreachable here -> the cycle degraded to deterministic reasoning.
    assert runtime.model_binding is None


# ---------------------------------------------------------------------------
# Kernel auto-size.
# ---------------------------------------------------------------------------


def test_kernel_max_workers_zero_auto_sizes():
    from ear.kernel import Kernel

    kernel = Kernel(max_workers=0)
    resolved = kernel._resolved_workers()
    assert resolved >= 1
    assert kernel.max_workers == resolved  # resolved once, stable after
