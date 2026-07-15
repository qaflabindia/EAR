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

## Wired into the reasoning loop

The plane is not just a set of seams beside the runtime — it runs *inside*
`Runtime.reason()`, off unless declared:

- **Thrift routes first.** If a runtime has a thrift ladder
  (`runtime.enable_thrift(light, heavy)`), each intent is routed to the
  smallest adequate model *before* the cycle's accounting begins, so token
  and energy figures are read against the binding actually used. When the
  chosen tier isn't reachable, the cycle degrades to deterministic reasoning
  rather than pointing at an unconfigured endpoint; a failed routing call
  degrades the same way — routing never takes the cycle down.
- **The energy budget is a gate.** A cycle that would exceed the declared
  daily watt-hours is refused before any work (`EnergyBudgetExceeded`), on the
  record — the same posture as a governance stop.
- **Every cycle is metered.** With a `## Energy` section declared, the Loader
  wires an `EnergyMeter`/`EnergyBudget` onto the runtime, and each cycle
  records its watt-hours (measured or estimated) against the exact tokens the
  usage record accounts — a blocked or parked cycle's energy is recorded too.

## 4. Carbon-aware scheduling — `ear/carbon.py`

The energy layer governs *how much* a cycle burns; carbon governs *when*.
`GridSignal` answers one question for the scheduler — is now clean enough to
run deferrable heavy work? — from, in order of preference:

- a **live grid provider** (a callable returning gCO2/kWh — WattTime,
  Electricity Maps, a national grid API — wired in code through the seam, so
  the zero-dependency core stays clean);
- a declared **clean-hours window** ("cleanest between 22:00 and 06:00")
  and/or a **threshold** ("defer heavy work above 300 gCO2/kWh") and/or a
  fixed declared intensity, from a `## Carbon` section in `memory.md`;
- otherwise **`None`** — unknown, and an unknown verdict *never* defers work.

**Stored energy** folds in: when the machine is on battery and discharging
(`HardwareProfile.on_battery`), deferrable work waits, so an off-grid or
laptop node spends its charge on what is due now.

Only tasks explicitly marked `deferrable=True` are ever held, and only when
the Kernel has a `grid` signal — a dirty-grid or on-battery deferrable task is
rescheduled to `next_clean()` (or a fixed backoff when no window is
predictable) and recorded as a `deferred` dispatch. Everything else runs when
due, exactly as before.

And carbon rides the energy record: when the watt-hours *and* a grid intensity
are both known, `EnergyMeter` writes the **gCO2** those watt-hours cost onto
the same trail line — a gram nobody could compute is never written down.

```markdown
## Carbon

The grid is cleanest between 22:00 and 06:00. Defer heavy work above 300
gCO2/kWh.
```

## How the plane composes

The four layers reinforce each other, all on the one spine:

1. **Hardware** sizes the fleet (`Kernel(max_workers=0)`) and tempers
   parallel width on battery/load.
2. **Thrift** picks the smallest adequate model per intent, cutting the tokens
   at the source.
3. **Energy** meters (or estimates) the watt-hours those tokens cost and
   refuses a cycle that would blow the day's budget.
4. **Carbon** defers heavy, deferrable work to a clean grid window or an
   off-battery moment, and prices the watt-hours in gCO2.

Fewer tokens (thrift) → fewer watt-hours (energy) → run them when the grid is
clean and the charge is plentiful (carbon) → on a fleet sized to the metal
(hardware). Measured where the host allows, estimated where the author
declares, and honest — `None` — where neither can speak.

## The one rule, extended once more

**The model judges; code measures, enforces, and records.** Complexity is the
model's call (thrift); the machine reading, the joules, the token-to-watt-hour
conversion, the daily-budget refusal, and the pool sizing are mechanics in
code; and offline every path degrades to a deterministic, labelled fallback —
or to an honest `None` — that says exactly what it is.
