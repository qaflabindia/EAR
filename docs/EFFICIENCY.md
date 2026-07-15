# EAR — Hardware, Energy & Compute Efficiency

**Status:** shipped (`ear/hardware.py`, `ear/energy.py`, `ear/thrift.py`,
the `## Energy` strategy section, Kernel auto-sizing).

An Enterprise AGI that runs at scale has to be honest about what it costs the
machine — cores, memory, watt-hours — and frugal by default. EAR already
metered *tokens* and *dollars*; this plane adds the physical layer:
**know the host, measure the energy, and spend the smallest adequate model.**
It follows EAR's one rule throughout — *the model judges; code measures,
enforces, and records; and a figure nobody measured or declared is never
invented.*

## 1. Hardware awareness — `ear/hardware.py`

`HardwareProfile.detect()` reads the machine the way a scheduler should — not
the bare `os.cpu_count()`, but what this process may **actually** use:

- **CPU** confined by the container's cgroup quota (v2 `cpu.max`, v1
  `cpu.cfs_quota_us/period_us`), not the node's core count.
- **Memory** capped by the cgroup (`memory.max` / `memory.limit_in_bytes`),
  min'd against host `MemTotal`.
- **Load** (`getloadavg`), **battery** (`/sys/class/power_supply`: charge and
  whether *discharging*), **energy counters** (RAPL presence), and **GPUs**
  (via `nvidia-smi`, only when the binary exists).

Every field the host does not expose stays `None` — reported, never invented
(macOS, restricted containers simply report less). Probe functions take
injectable root paths, so the parsing is tested against fixture trees, not
the real `/sys`.

`recommended_workers()` turns the profile into a pool size: I/O-bound cycles
(the GIL is released on the model call) may exceed the core count, but the
figure is **tempered by current load** and **halved on battery** — parallel
width multiplies power draw exactly when stored energy is the constraint.
Deterministic mechanics from the profile, no judgment.

**Kernel auto-sizing:** `Kernel(max_workers=0)` resolves the pool size from
`HardwareProfile` once, on first use — a fleet that fits the machine it lands
on without hand-tuning.

## 2. Energy metering & budgets — `ear/energy.py` + `## Energy`

Two sources of truth, never confused:

- **Measured** — on hosts exposing RAPL counters, `EnergyMeter.start()/stop()`
  reads real joules across the interval, **wrap-safe** (a wrapped counter adds
  its max range back), summed across package zones. RAPL meters the whole
  package, and the reading says so — an honest whole-machine figure beats a
  fabricated per-process one.
- **Declared estimate** — an `## Energy` section in `memory.md` ("reasoning
  costs about 0.4 watt-hours per thousand tokens") converts the token spend at
  the author's rate, exactly as `## Pricing` converts to dollars. The rate is
  the author's declaration, never a table shipped in code.

With neither, a reading is **unmetered** — stated as such, never guessed.
Every reading lands on the one audit spine (stage `energy`), labelled
`measured` / `declared estimate` / `unmetered`.

`EnergyBudget` enforces a prose-declared daily cap ("keep within 50 watt-hours
per day") **before a cycle starts**: the spend is summed from the trail's own
token records at the declared rate — the audit spine *is* the energy ledger —
and an exhausted budget raises `EnergyBudgetExceeded` (a `PermissionError`,
like every governance stop), loudly, on the record. An unbudgeted stack is
simply not budgeted — never silently capped.

```markdown
## Energy

Reasoning costs about 0.4 watt-hours per thousand tokens. Keep within a
budget of 50 watt-hours per day.
```

## 3. Compute thrift — `ear/thrift.py`

Most enterprise cycles are light work (extraction, routing, classification,
short summaries). Sending every one to the largest model burns compute,
dollars and energy for no better answer.

`ModelThrift(light=…, heavy=…)` routes each intent by **judged complexity** —
and the routing is judged by the **light** model itself
(`JudgeTaskComplexity`), so the decision costs a cheap call, never an
expensive one. Escalation is honest by instruction: an uncertain judge says
*heavy* (a wasted large call costs money; a botched hard task costs more).

Offline, or when the light model isn't reachable (no credential), the fallback
is deterministic and **labelled**: route by the intent's sheer size. A routing
judgment nobody made is never written down as one. Every choice lands on the
spine (stage `thrift`) with the tier, the basis, and the rationale — so the
saving is measurable from the trail against Pricing dollars and the `## Energy`
rate.

```python
thrift = ModelThrift(
    light=ModelBinding(provider="anthropic", model="claude-haiku-4-5"),
    heavy=ModelBinding(provider="anthropic", model="claude-opus-4-8"),
)
thrift.bind(runtime, intent)   # sets the smallest adequate model for this cycle
```

## How the plane composes

The three layers reinforce each other, all on the one spine:

1. **Hardware** sizes the fleet (`Kernel(max_workers=0)`) and tempers
   parallel width on battery/load.
2. **Thrift** picks the smallest adequate model per intent, cutting the tokens
   at the source.
3. **Energy** meters (or estimates) the watt-hours those tokens cost and
   refuses a cycle that would blow the day's budget.

Fewer tokens (thrift) → fewer watt-hours (energy) → a fleet sized to the metal
(hardware). Measured where the host allows, estimated where the author
declares, and honest — `None` — where neither can speak.

## The one rule, extended once more

**The model judges; code measures, enforces, and records.** Complexity is the
model's call (thrift); the machine reading, the joules, the token-to-watt-hour
conversion, the daily-budget refusal, and the pool sizing are mechanics in
code; and offline every path degrades to a deterministic, labelled fallback —
or to an honest `None` — that says exactly what it is.
