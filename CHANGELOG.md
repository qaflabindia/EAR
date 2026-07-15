# Changelog

All notable changes to EAR are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
has not yet made a versioned release, so entries accumulate under
`[Unreleased]` until the first tagged version.

## [Unreleased]

### Added
- **Carbon-aware scheduling** (`ear/carbon.py`, the `## Carbon` strategy
  section, Kernel task deferral) -- run heavy work when the energy is clean
  and plentiful (`docs/EFFICIENCY.md`).
  - `GridSignal` answers "is now clean enough to run deferrable work?" from a
    live grid `provider` seam (gCO2/kWh, wired in code), a declared clean-hours
    window and/or gCO2/kWh threshold and/or fixed intensity (from `## Carbon`
    in memory.md), or `None` -- an unknown verdict never defers work, and a
    dead provider reads as unknown, not fatal. On battery + discharging,
    deferrable work waits (stored energy). `next_clean()` predicts the next
    window; `carbon_grams` converts Wh x gCO2/kWh.
  - Kernel: a task submitted `deferrable=True` is held when the grid is dirty
    or the machine is on battery (only when a `grid` signal is set), pushed to
    the next clean window (or a fixed `carbon_backoff`) and recorded as a
    `deferred` dispatch. Non-deferrable work, and a kernel with no grid, run
    exactly as before.
  - `EnergyMeter` records the **gCO2** a cycle cost on the energy trail line
    when both the watt-hours and a grid intensity are known (`EnergyReading
    .carbon_grams`); absent either, no carbon is invented. `Strategy
    ._read_carbon` / `grid_signal()`; the Loader wires the grid onto the meter.
- **Hardware, energy & compute efficiency** (`ear/hardware.py`,
  `ear/energy.py`, `ear/thrift.py`, the `## Energy` strategy section, Kernel
  auto-sizing) -- the physical resource plane (`docs/EFFICIENCY.md`). Honest
  throughout: a figure nobody measured or declared is never invented.
  - `ear/hardware.py`: `HardwareProfile.detect()` reads what the process may
    *actually* use -- CPU confined by cgroup quota (v2 `cpu.max`, v1
    `cfs_quota/period`), memory capped by the cgroup, load, battery (charge +
    whether discharging), RAPL presence, and GPUs (`nvidia-smi` when present).
    Fields the host does not expose stay `None`. `recommended_workers()` sizes
    an I/O-bound pool, tempered by load and **halved on battery**. Probe
    functions take injectable roots (tested against fixture `/sys` trees).
  - `ear/energy.py`: `EnergyMeter` measures real joules from RAPL counters
    (wrap-safe, summed across package zones) when the host exposes them, or
    estimates watt-hours from a declared rate otherwise, or reports
    `unmetered` -- each reading labelled and recorded on the one audit spine
    (stage `energy`). `EnergyBudget` enforces a prose-declared daily cap
    before a cycle starts, summing spend from the trail's own token records,
    raising `EnergyBudgetExceeded` (a `PermissionError`) loudly on the record.
    A new `## Energy` section in `memory.md` declares the rate and budget the
    way `## Pricing` declares dollar rates (`Strategy._read_energy` /
    `watt_hours`).
  - `ear/thrift.py`: `ModelThrift(light, heavy)` routes each intent to the
    smallest adequate model by complexity judged *on the light model itself*
    (`JudgeTaskComplexity`, new in `ear/signatures.py`), so the routing costs
    a cheap call; an uncertain judge escalates to heavy. Offline / unreachable
    light model degrades to a deterministic, labelled size-based fallback.
    Every choice records on the spine (stage `thrift`).
  - `Kernel(max_workers=0)` auto-sizes the fleet pool from `HardwareProfile`
    once, on first use.
  - **Wired into `Runtime.reason()`** (off unless declared): thrift routes the
    intent's model *before* the cycle's accounting (so tokens/energy are read
    against the binding actually used, degrading to deterministic reasoning
    when a tier is unreachable or a routing call fails); the energy budget
    refuses a cycle before any work if the declared daily watt-hours are spent;
    and every cycle records its energy against the exact tokens the usage
    record accounts (a blocked/parked cycle's energy is recorded too). The
    Loader wires `EnergyMeter`/`EnergyBudget` from a `## Energy` section;
    `Runtime.enable_thrift(light, heavy)` attaches the model ladder.
- **Concurrency & parallelism** (`ear/parallel.py`, Kernel fleet pool) --
  the decided model is *single-writer actors, thread parallelism, one ordered
  spine* (`docs/CONCURRENCY.md`).
  - `ear/parallel.py`: a native, dependency-free `joblib`-shaped engine over
    `concurrent.futures.ThreadPoolExecutor` (threads because a cycle is
    I/O-bound on the model call, which releases the GIL). `parallel_map(fn,
    items)` returns one `Result` per item **in input order** regardless of
    completion order, with per-unit error isolation (a unit that raises
    captures its error; the batch still returns a full ordered set;
    `values(..., on_error=...)` chooses partial-tolerance). A **backend seam**
    (`serial` / `threads`) is where an out-of-process backend attaches later;
    nesting is **depth-guarded** so a bounded pool can never deadlock or
    explode. `map_reduce(items, map_fn, reduce_fn)` scatters in parallel and
    folds; `JudgedReducer` reduces *by judgment* -- the model synthesizes the
    parts (`SynthesizeParallel`, new in `ear/signatures.py`) with a
    deterministic fallback offline.
  - Kernel fleet pool: `Kernel(max_workers=N)`. With `N > 1` the loop runs
    work for **different** instances concurrently on a bounded pool while
    holding **at most one in-flight cycle per instance** (the single-writer
    actor invariant, so an instance's memory and hash-chained audit spine
    never have two writers); `drain_concurrent` advances it synchronously for
    tests. `max_workers=1` (the default) is the historical serial scheduler,
    unchanged -- `tick`/`drain` are untouched.

### Changed
- **Reason-first governance** (correcting Phase 3 to EAR's own rule -- the
  model judges, code enforces and records, and a deterministic path is only
  ever an honest fallback):
  - `EnvelopePolicy` (AECC) is no longer a hardcoded allow-list. It now
    keeps a *deterministic floor* -- uncertified / revoked / suspended /
    tampered / no-envelope, which is absolute and never model-waivable (what
    makes revocation immediate, `CapabilityEnvelope.floor` /
    `EnvelopeRegistry.floor`) -- and *judges scope/tier authorization above
    the floor*: with a model bound it reasons over the envelope's granted
    scopes, tier and standing (injected into the context) via the base
    `Policy.judge`; offline it falls back to the deterministic
    `envelope_authorizes` check and says so. Reason-first, exactly like
    every other policy.
  - `is_flagged` (ATC) is model-judged with a deterministic fallback. A
    factual floor (an explicit high-stakes/irreversible value, or a
    probationary acting agent) still flags on its own; otherwise a bound
    model decides whether the intent warrants scrutiny
    (`FlagForAdversarialReview`, new in `ear/signatures.py`), and offline an
    unremarkable intent is not flagged and announces that -- never a keyword
    rule masquerading as a judgment. The flag decision (including a decision
    *not* to review) lands on the one audit spine.

### Added
- `ear/authority.py` + `ear/adversary.py`: **Enterprise AGI Phase 3** --
  who may act (AECC capability envelopes), and the red team on the intent
  path (ATC adversarial deliberation).
  - **AECC capability-envelope enforcement.** `CapabilityEnvelope` is a
    non-human actor's authority record (certified, scopes, max autonomy
    tier, standing active/probation/suspended/revoked, trust score, and a
    `hashlib` content signature over its authority fields so an on-disk edit
    no longer verifies). `EnvelopeRegistry` loads envelopes from a centre's
    `state/authority_envelopes.json` through the Phase-1 backend and owns
    the transitions -- `certify`, `set_trust`, `probation`, `suspend`,
    `revoke`, `reinstate` -- each recorded on the one audit spine and
    persisted back to state. `enforce_envelopes(runtime, registry)` attaches
    an `EnvelopePolicy` (a `Policy` subclass) at runtime scope whose
    judgment consults the *live* registry: a human-initiated intent (no
    agent in context) is not applicable; an agent-initiated one blocks
    unless the agent holds an active, in-scope, in-tier envelope; and
    **revocation is immediate** -- the next `Governor.govern` fails the gate
    with no reload. A signing secret, when used, is named by environment
    variable only.
  - **ATC adversarial-deliberation hook.** `is_flagged` triggers an
    adversarial pass on a high-stakes/irreversible/adversarial context
    value, or on an acting agent AECC has placed on probation.
    `AdversarialReview.review` argues the case against the action, the
    defense, and a verdict -- `uphold` / `escalate` / `overturn` -- via the
    new native `AdversarialChallenge` judgment when a model is bound, and a
    conservative deterministic fallback offline (a flagged low-confidence
    action escalates rather than being silently upheld; an unparseable
    verdict escalates, never guesses an uphold). Challenge, defense and
    verdict land on the one audit spine; an unflagged intent is never
    delayed. `AdversarialChallenge` added to `ear/signatures.py`.
  - `CommandCentre.bind` gained governance-plane specializations: binding
    AECC attaches envelope enforcement and exposes `runtime.envelope_registry`;
    binding ATC exposes `runtime.adversarial_review`. `Binding` now reports
    both. Fixtures `tests/fixtures/command_centres/{aecc,atc}` and
    `tests/test_authority.py` / `tests/test_adversary.py` (27 tests) cover
    the gate, immediate revocation, signature tampering, the flag predicate
    and the offline fallback, plus one live adversarial judgment.
- `ear/compiler.py` + `ear/mcp_command_centre.py`: **Enterprise AGI Phase 2**
  -- a whole command centre compiles to a runnable EAR stack, and can be
  served out-of-process as a native MCP server.
  - **Centre → EAR-stack compiler.** `StackCompiler` /
    `compile_command_centre` map an acc-skills command centre onto the six
    natural-language stack files an author would otherwise write by hand
    (architecture §3.5): `SKILL.md` mission → `persona.md`; `##
    Capabilities` → `skills.md` (one skill each); `## Procedures` →
    `workflow.md` (steps delegating to the persona); a process wrapping them
    → `process.md`; `references/constitutional_rules.md` → `policy.md` (via
    Phase 1's `Constitution.to_policy_markdown`); the remaining
    `references/*.md` → `knowledge/`; `SKILL.md` frontmatter org context →
    `tenant.md`; and an operating strategy → `memory.md` declaring the
    knowledge sources and one `.ear/reasoning.md` audit spine. Nothing an
    author wrote is dropped -- an unconsumed `##` section folds into the
    persona as prose (flattened out of bullets, which the loader would
    otherwise read as skill references), and `compile(verify=True)` loads
    the result once so every cross-reference resolves or the loader fails
    loudly. `CompiledStack.load()` runs the compiled centre as a first-class
    Runtime; `CompiledStack.mapping` reports which artifact produced each
    file.
  - **MCP packaging.** `CommandCentreServer` serves a centre as a native
    stdio JSON-RPC MCP server (the out-of-process binding, architecture
    §3.4), exposing its script pentad as tools -- `list_state`,
    `load_state`, `update_state`, `evaluate`, `audit` -- over the Phase-1
    `CommandCentreBackend` and `Constitution`. `evaluate` runs the
    constitution's deterministic fallbacks (a plain subprocess, no model
    bound), so an out-of-process centre still enforces its mechanically
    checkable rules. Launch with `python -m ear.mcp_command_centre <dir>`
    and connect via `Runtime.connect_mcp`; the dispatch (`call`) is pure and
    synchronous, so it is testable without a subprocess.
  - Fixture `tests/fixtures/command_centres/afcc` (AFCC, an operational
    finance centre: mission, `## Capabilities`, `## Procedures`,
    `CR-FIN-01..05`, taxonomy/matrix references, budget and vendor state)
    is the worked end-to-end example; `tests/test_compiler.py` compiles it,
    loads the stack, enforces its constitution, and round-trips the MCP
    server over EAR's own `McpClient`.
- `ear/enterprise.py`: the **Enterprise AGI** binding layer -- Phase 1 of
  the framework architecture (`docs/ENTERPRISE_AGI.md`), binding the
  thirteen `acc-skills` constitutional command centres onto EAR's execution
  substrate, "least invasion first."
  - **Constitutions become policies.** `Constitution.from_directory` reads a
    centre's `references/constitutional_rules.md`; each rule compiles to an
    EAR `Policy` (`ConstitutionalRule.to_policy`) -- the rule's prose is the
    `statement` an LLM judges, any mechanically checkable clause is the
    policy's `Fallback:` deterministic expression, and the declared scope
    becomes `Applies to:`. `Constitution.to_policy_markdown` renders a
    `policy.md` the existing `Loader` reads back unchanged, so English stays
    the source of truth and nothing an author wrote is dropped in
    compilation (a compiled constitution round-trips to the same policies it
    started as).
  - **AGCC verdicts map onto the one policy gate.** `Verdict` translates the
    AGCC verdict vocabulary onto `Governor.govern` behaviour: `HALT` is a
    hard, unwaivable block; `DEFER`/`ESCALATE` park the cycle for a human
    (riding the same `Approval`/`ApprovalRequired` machinery, `ESCALATE`
    with a declared deadline); advisory verdicts (`CONSTRAIN`,
    `EXECUTE_WITH_ADVISORY`) ride the reasoning log rather than block.
    Enforcement flows through the same choke point every other intent
    clears -- there is no private governance path.
  - **State behind one store abstraction.** `CommandCentreBackend` exposes a
    centre's `state/*.json` through EAR's `CatalogueBackend` protocol
    (`list/exists/read/write/delete`, `read_json`/`write_json`), satisfied
    structurally. Phase 1 is an adapter: the JSON files stay the source of
    truth, zero changes to acc-skills. The append-only `audit_trail.jsonl`
    is never adapted as state -- `CommandCentre.mirror_audit` folds it onto
    EAR's one audit spine.
  - `CommandCentre.load` / `.bind` load a centre and attach its constitution
    onto a runtime at the declared scope (runtime, tools, or a named
    workflow); `load_command_centres`/`bind_command_centres` load and bind a
    whole acc-skills root, governance plane first (the governance plane
    governs the operational and cognitive planes). `COMMAND_CENTRES`/
    `plane_of` carry the thirteen-centre plane assignment.
  - Fixture `tests/fixtures/command_centres/agcc` (AGCC, CR-AG01..08 with
    state and a ledger) makes the whole binding demonstrable and the suite
    self-contained -- no reach to the acc-skills repo.
- `ear/evolution.py`: governed self-modification, configured, never
  assumed. `EvolutionPolicy` declares which *kinds* of change a runtime
  may make to itself (`allowed_changes`), which it may never make
  (`prohibited_changes` -- the prohibition always wins, even over the
  allow-list), and what every permitted change must carry: a sandbox to
  trial in, an evaluation to pass, an explanation on the record, a human
  approval for the sensitive kinds (`require_human_approval_for`, riding
  the same `Approval`/`ApprovalRequired` machinery as policy.md's gates),
  and a rollback so no change is a one-way door. `Runtime.enable_evolution`
  raises the fence (off by default: a runtime that never enabled evolution
  refuses every proposed change); `Runtime.evolve` walks an
  `EvolutionChange` through the `Evolver`'s gates in order, applies it only
  once every gate passes, rolls back on a failed evaluation or a crashed
  apply, and records every refusal, park and promotion as an `evolution`
  trail record. Enabling evolution also puts the Acquirer's
  `create_tool`/`retire_tool` under the same policy (a `tool_adapter`
  change), so the tools-that-create-tools loop cannot outrun the fence.
  The policy is also authorable in memory.md: an `## Evolution` section
  (`- Allowed:`/`- Prohibited:`/`- Approval required:` bullets, the four
  requirements defaulting on and relaxed only by explicit prose) is read
  by `Strategy` and applied by the Loader on `load_runtime`.
- `ear/caveman.py`: a deterministic, zero-dependency prose compressor
  (ported, MIT license, from github.com/JuliusBrussee/caveman's
  `caveman-shrink` MCP middleware) that always compresses a tool result
  before it re-enters the native tool loop's `gathered` context -- drops
  filler words via `re.sub`, protects fenced/inline code, URLs, paths,
  `CONST_CASE` identifiers, dotted calls and version numbers via sentinel
  substitution, and never touches a bare number (nothing in the rule set
  matches digits). Deterministic by construction: it can only delete
  matched words, never generate replacement text, so it cannot hallucinate
  or garble a fact the way a generative summarizer can.
- Auxiliary Model (memory.md): a second, optional ModelBinding a stack may
  declare for two mechanical jobs layered on top of `ear.caveman`'s
  always-on pass -- squeezing a tool result further
  (`SummarizeToolResult`), and consolidating everything gathered so far
  into one checkpoint every 3 tool calls (`ConsolidateGatheredContext`,
  `Reasoner._checkpoint_gathered_context`) so key facts stay retained
  instead of diluting across a lengthening list of compressed entries.
  Both judgments' instructions are explicit: no fabrication, no
  shallowness, no fluff, no sloppiness, no context loss or distortion of
  any number, path, name or outcome. The full, uncompressed tool result
  always lands on the reasoning trail regardless -- only the copy that
  re-enters the prompt is ever touched. Declaring no Auxiliary Model
  leaves both a no-op; `Strategy._parse_model_prose` now also reads a
  declared `max_output_tokens` ("allow up to N tokens per reply"), shared
  by the primary Model Selection and the Auxiliary Model sections.
- `section.argument_blocks` + a new `"map"` Judgment field kind
  (`ear/judgment.py`, `ear/tool_binder.py`): a tool call's arguments may
  now mix short `- name: value` bullets with a `name:` + blockquote form
  for a value that needs more than one line -- a script's source, a whole
  file's content -- which a single bullet line can never carry.
- `ear/dashboard.py`'s live server gained a `create_server()`/`serve()`
  split (build the `HTTPServer` without blocking, so a caller can run it
  in its own thread and control its lifecycle), a `GET /download/<name>`
  route streaming a file out of the sandbox's `outputs/` (confined by
  `Sandbox.resolve`), an "Outputs" panel listing whatever lands there as
  it appears, and a `POST /shutdown` route (a page button) that stops the
  server -- and only the server; nothing under `uploads/`, `workspace/`
  or `outputs/` is ever touched.
- `examples/sales_mis_stack/` + `examples/sales_mis_guru.py`: a second
  worked example alongside `credit_risk_stack` -- a four-step Sales MIS
  cycle (load, sanity-check, slice-and-dice a dashboard workbook,
  reconcile and validate) driven entirely by the model authoring and
  running its own code inside the Sandbox, never a hand-written script.
  `examples/sales_mis_stack/logs/` keeps the debugging trail of getting a
  real multi-step tool-loop cycle working end to end, including the bugs
  this release's Fixed section below closes.
- Shared-volume support in `ear/k8s.py`: `host_path_volume`/`pvc_volume`/
  `volume_mount` (pure manifest builders, same style as `container_spec`)
  and `volumes`/`volume_mounts` params on `container_spec`/`job_manifest`/
  `cronjob_manifest`. `KubeProvider` gained `host_path`/`pvc_claim` --
  when either is set, every Job/CronJob it creates mounts it at
  `stack_mount` automatically, so a stack's files (and the Sandbox's
  `uploads/`/`outputs/` inside it) move between the host and the
  container as plain file writes, never a copy-through-the-API round
  trip. Neither set: the manifest is byte-identical to before this
  field existed -- purely additive, no behavior change for an existing
  caller.
- `Sandbox.capabilities()` (`ear/sandbox.py`): checks which runtimes
  (`python3`, `node`, `npm`, `pip` by default) are actually reachable on
  the sandbox's PATH and their live `--version`, via `shutil.which` plus
  the same confined `run()` every command goes through -- verified, never
  assumed. Bound as `check_environment` under the `environment_admin`
  toolset. A reference `Dockerfile` at the repo root gives `ear/k8s.py`'s
  `KubeProvider.image` a pod image with both toolchains installed
  (Debian `nodejs`/`npm` alongside the zero-dependency `pip install .`);
  built and its `python3`/`node`/`npm`/`pip` presence confirmed live
  inside the built image, not just claimed.
- `Toolsets` (memory.md), `ear/web.py` (`WebAccess`), `ear/mail.py`
  (`Mail`): ten basic toolsets an operator enables/disables by
  declaration -- `internet_access`, `internet_search`, `read_documents`,
  `write_documents`, `code_executor`, `browser_automation`, `terminal`,
  `email_sender`, `mcp_connector`, `environment_admin` -- for mechanics
  that need no per-call reasoning (fetching a URL, sending mail). `fetch_url`
  and `web_search` (Tavily) and `send_email` are the net-new native
  tools this ships (stdlib `urllib`/`smtplib` only); the other seven names
  in the table map onto tools already shipped elsewhere (Sandbox,
  Acquirer, native MCP client) rather than duplicating them. Defaults
  mirror a Tools-Hub-style posture: internet access, document reads, code
  execution and environment admin on; search, document writes, browser
  automation, terminal and mail off. `Strategy.toolsets` folds loose
  bullet phrasing ("Internet Access (Web Fetch)", "Terminal / Shell") to
  the ten canonical keys; an unrecognized name still becomes its own
  toggle rather than an error.
- `Acquirer` (`ear/acquirer.py`): a runtime's basic tools for managing its
  own toolset -- `list_tools`, `view_tool`, `create_tool`, `retire_tool` --
  themselves exposed as BoundTools (`Acquirer.as_tools`), so a live
  deliberation can declare a brand-new tool for itself mid-cycle the same
  way it calls any other tool. A declared tool persists to `.ear/tools.md`
  (Section codec, reviewable and diffable) and survives past the session;
  the Loader merges it back in on the next load, memory.md's own
  declarations always winning on a name clash. Declaring is not binding: a
  self-declared tool stays context to the model until an MCP server,
  sandbox command, or Python binding gives it a handler. Only tools the
  Acquirer itself declared (`Tool.origin == "acquired"`) can be retired
  through it; a human-authored tool is edited by editing memory.md, never
  rewritten by code. On by default (`Strategy.tool_acquisition`), turned
  off by disabling language under Tools in memory.md ("fixed toolset",
  "no new tools"). `python -m ear.tools_cli list|view|create|retire
  <stack-dir> ...` gives a human the same four operations from the
  command line.
- `Optimizer.refine_skill` / `Optimizer.persist_skill` (`ear/optimizer.py`):
  N1.4's reflective instruction rewrite (`RefineInstruction`), aimed at a
  Skill's prompt instead of a Judgment's instruction, and a round-trip
  writer that replaces just that skill's section in a stacked skills.md --
  a refined skill is reviewable and diffable exactly like any
  human-authored one, and survives past the session.
- `Claim` (`ear/identity.py`): who is calling and which Tenant `org_id`(s)
  they may act as -- the piece `Tenant` explicitly deferred to. Checked at
  `Runtime.reason(intent, claim=...)` before a cycle starts and at
  `Kernel.submit(..., claim=...)`/dispatch time for scheduled work; either
  refuses with `TenantBoundaryViolation` (a `PermissionError`, so a Kernel
  task lands `blocked` rather than `failed`). No `claim` supplied is not a
  violation -- the same "off unless declared" posture as `Tenant` itself.
  `TaskDefinition` (`ear/task.py`) gained an `org_id` field (an optional
  `Org id:` line, round-tripped through `TaskStore`) so a shared catalogue
  can tag which org a stored task belongs to.
- `Tenant` (`ear/tenant.py`): the org a stack belongs to, stacked from an
  optional `tenant.md` the same way every other stacked file works --
  `Org id:`, fiscal year bounds, timezone, secret env var. Absent file
  falls back to a default tenant with calendar-year fiscal bounds. Also
  the basis for `schedule.md`'s workday notation, which resolves
  `q`/`h`/`y`/`a` occurrences against the tenant's fiscal year.
- CI: a GitHub Actions workflow (`.github/workflows/ci.yml`) running the
  test suite on Python 3.10-3.12, plus a dedicated bare-environment job
  that installs EAR with zero third-party packages and asserts none crept
  in before running the suite against that install.
- `LICENSE`: the MIT license text, matching the license already declared
  in `pyproject.toml`.
- Release automation: `.github/workflows/release.yml` builds the sdist
  and wheel, verifies the pushed tag matches `pyproject.toml`'s version,
  runs the full suite, and publishes to PyPI via trusted publishing (OIDC)
  when a `v*.*.*` tag is pushed. Requires a PyPI trusted publisher
  configured for this repository before the first tag -- an account-level
  step, not something CI can do for itself.

### Fixed
- The native tool loop's `ChooseToolAction` arguments were parsed as flat
  `- name: value` bullets, one per line -- a value containing a newline
  (a whole script's source) silently truncated to whatever fit on one
  line, with no error and no signal back to the model. `argument_blocks`
  (see Added) fixes this while staying backward compatible with every
  existing short-value bullet.
- `LM`'s request body (`ear/llm.py`) hardcoded `DEFAULT_MAX_TOKENS = 2048`
  with no way to override it from memory.md; a tool-loop turn writing a
  real script routinely needs far more, and a reply cut off mid-token
  looked identical to a model that simply stopped early. `max_output_tokens`
  is now readable from Model Selection/Auxiliary Model prose (see Added).
- MCP client: a timed-out call used to leave its reader thread alive,
  racing later calls' own reader threads for the same stdout bytes and
  occasionally stealing a later call's response. Replaced with one
  persistent reader thread per connection, dispatching responses by
  JSON-RPC id.
- Retention: a declared retention window ("keep 90 days") only rotated
  the trail when the Journey runner's batch pass crossed it; a plain
  `Runtime.reason()` cycle silently ignored it. `apply_retention()` now
  runs from both `Runtime.reason` (every cycle) and `Journeys.run_all`
  (as a batch-pass backstop).
- Secrets hygiene: `KubeConfig.token`, `ModelBinding.api_key` and
  `LM.api_key` are now `field(repr=False)`, so a stray `print`/`repr`/
  traceback dump can no longer leak a credential.
- Documentation drift: the server's bearer-token guard was documented as
  exempting `/health`; the code (and its test) actually requires the
  token there too. Docs corrected to match the code.

## [0.1.0] - initial native parity milestone

The first complete pass of the native-parity plan
(`docs/IMPLEMENTATION_PLAN.md`, phases N1-N4), plus the runtime substrate
it stands on. Highlights:

### Core authoring & runtime
- Markdown-authored stacks: skills, personas, workflows, processes,
  policies and memory, stacked and loaded by `Loader` with zero required
  Python.
- The full reasoning pipeline (`Runtime.reason`): initialize, discover,
  select, delegate, compose, schedule, govern, deliberate, decide,
  validate, remember, research, explain, audit, store, learn, adapt.
- `Judgment`/`Section`: native structured prompting over markdown, no
  JSON-schema library.
- `ModelBinding` and `LM`: a dependency-free LLM client speaking to
  Anthropic natively and any OpenAI-compatible provider over `api_base`.

### N1 - reasoning & optimization depth (parity: DSPy + GEPA)
- LM retries with backoff, on the record; per-judgment tokens and
  latency; prose-declared `Pricing` turning usage into dollars.
- `Optimizer.search`: iterative, evaluation-graded instruction search
  with worked-example demos and persisted, loader-applied instructions.

### N2 - evaluation & knowledge depth (parity: LangSmith/Phoenix, LlamaIndex)
- `Examiner`: report history with regression diffs, rubric-scored
  criteria, and A/B stack comparison via judged preference.
- `Knowledge`: native BM25 narrowing, a persisted gist index, and URL
  knowledge sources fetched over the native HTTPS client.

### N3 - execution depth (parity: LangGraph, Temporal runtime, AutoGen)
- `Journey`: prose-authored routing with revisit budgets, leg retry
  policies, a batch runner with deadline escalation, and event documents.
- `Panel`: judged next-speaker selection, early consensus, and tools
  usable inside a persona's turn.

### N4 - governance & connectivity depth (parity: Temporal governance, LangChain reach)
- Approver allow-lists and tool-scoped policies.
- A hash-chained, tamper-evident reasoning trail with `verify()`.
- Retention rotation and a native usage ledger.
- `McpClient`: a native MCP client (stdio JSON-RPC from the standard
  library) binding a declared server's tools into the cycle.

### Beyond the original plan
- `Router`: an omni-route, provider-agnostic `ModelBinding` with routing
  strategies, fallback and circuit-breaking across 250+ providers.
- `GoalKeeper`: session goals with typed blockers and a bounded
  autonomous continuation loop.
- `Sandbox`: a confined per-instance workspace and governed command
  runner, nesting per subagent.
- `Delegator`/sub-agent seams: declarative sub-agents (Delegator,
  Synthesizer) over the Sandbox.
- `Kernel` and `Server`: an OS-style scheduler (run-or-sleep idle loop)
  and a stdlib HTTP control plane over it.
- `KubeProvider` (`ear.k8s`): running instances as Kubernetes Jobs/
  CronJobs, spoken natively over the REST API.
- `Dashboard` and `Monitor`: a self-contained HTML runtime board and a
  live ANSI-truecolor TUI of the fleet.
